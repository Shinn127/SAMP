# -*- coding: utf-8 -*-
"""
SAMP - Semantically Aligned Motion Prediction (架构升级版)
Author: Adapted from Shinn Ma
Date: 2026-01-15

核心改进：
1. 后验编码器使用Transformer Encoder + RoPE + RMSNorm + SwiGLU
2. LatentMotionPredictor使用Transformer Decoder + MQA + RoPE + RMSNorm + SwiGLU
3. 适配新数据集的输入维度（13帧动作序列，6帧轨迹特征）
4. 实现next-token prediction（自回归预测4帧latent motion）

Pipeline:
    1. Posterior: q(z | x_{-6:6}, t) → z_post（训练）
    2. Prior: p(Δz | x_{-6:0}^{traj}) → Δz（推理采样）
    3. z = z_base + Δz
    4. LatentMotionPredictor(z, x_{-6:0}) → latent_motion (4帧, 64-dim)  // 自回归预测
    5. Diffusion: p(x_{1:4} | latent_motion) → final motion
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------------------------------------------
# 基础组件：RMSNorm, RoPE, SwiGLU
# --------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embeddings (RoPE)"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_cos_sin_tables(x, seq_dim=1)
        return self.apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)


class SwiGLU(nn.Module):
    """SwiGLU activation with hidden dimension expansion"""
    def __init__(self, in_dim: int, hidden_dim: int = None, out_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 4
        out_dim = out_dim or in_dim
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --------------------------------------------------
# 多头注意力实现 (MHA for Posterior, MQA for Decoder)
# --------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 rope: RotaryPositionalEmbeddings,
                 qkv_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = rope

    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to q and k
        q = self.rope(q)
        k = self.rope(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) with RoPE for Decoder"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 rope: RotaryPositionalEmbeddings,
                 qkv_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # MQA: multiple queries, single key/value head
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias)  # Single head for keys
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias)  # Single head for values
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = rope

    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(x).unsqueeze(1)  # [B, 1, L, D]
        v = self.v_proj(x).unsqueeze(1)  # [B, 1, L, D]
        
        # Apply RoPE to q and k
        q = self.rope(q)
        k = self.rope(k)
        
        # Attention (broadcast single key/value head to all query heads)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, L, D]
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out)


# --------------------------------------------------
# Transformer块：使用新组件
# --------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with MHA + RoPE + RMSNorm + SwiGLU"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=RotaryPositionalEmbeddings(dim=embed_dim // num_heads),
            qkv_bias=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(in_dim=embed_dim, hidden_dim=embed_dim * 4)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block
        x = x + self.dropout1(self.attn(self.norm1(x), attn_mask))
        # FFN block
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block with MQA + RoPE + RMSNorm + SwiGLU"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.self_attn = MultiQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=RotaryPositionalEmbeddings(dim=embed_dim // num_heads),
            qkv_bias=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(in_dim=embed_dim, hidden_dim=embed_dim * 4)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block with causal mask
        x = x + self.dropout1(self.self_attn(self.norm1(x), attn_mask))
        # FFN block
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


# --------------------------------------------------
# 1. 文本编码器（保持不变，但增加维度适配）
# --------------------------------------------------
try:
    import clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

class RealCLIPTextEncoder(nn.Module):
    r"""
    冻结 CLIP 文本编码器，输出 512 维归一化向量。
    公式： t = \text{CLIP}_{\text{text}}(\text{text}) \in \mathbb{R}^{512}, \|t\|_2 = 1
    """
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self._use_clip = False
        if _HAS_CLIP:
            try:
                self.model, _ = clip.load(model_name, device=device, jit=False)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self._use_clip = True
            except:
                pass

        if not self._use_clip:
            self.fallback_proj = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 512)
            )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self._use_clip:
            # 截断超长文本（CLIP 最多 77 tokens）
            processed = [" ".join(t.split()[:70]) for t in texts]
            tokens = clip.tokenize(processed, truncate=True).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens).float()
            return feats / feats.norm(dim=-1, keepdim=True)
        else:
            B = len(texts)
            buf = torch.zeros(B, 768, device=self.device)
            for i, s in enumerate(texts):
                arr = torch.tensor([ord(c) for c in s[:768]], dtype=torch.float32, device=self.device)
                buf[i, :arr.numel()] = arr
            out = self.fallback_proj(buf)
            return out


# --------------------------------------------------
# 2. 后验编码器：q(z | full_motion, text) — 重构为Transformer Encoder
# --------------------------------------------------
class MotionPosteriorEncoder(nn.Module):
    r"""
    后验分布：q_\phi(z | x_{-6:6}, t) = \mathcal{N}(\mu, \sigma^2)
    
    输入序列（16 tokens）：
        [mu_token, logvar_token, text_token, motion_{-6}, ..., motion_{6}]
    
    结构：纯 Transformer Encoder with RoPE + RMSNorm + SwiGLU
    """
    def __init__(self,
                 motion_dim: int,
                 text_dim: int = 512,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 z_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = 2 + 1 + 13  # mu, logvar, text, 13 motions

        # Token 投影
        self.motion_proj = nn.Linear(motion_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 可学习的 μ / logσ² tokens
        self.mu_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # 输出头
        self.mu_head = nn.Linear(embed_dim, z_dim)
        self.logvar_head = nn.Linear(embed_dim, z_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, motion_seq: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = motion_seq.shape[0]
        device = motion_seq.device

        # 构建 tokens
        motion_tokens = self.motion_proj(motion_seq)                     # [B,13,embed_dim]
        text_token = self.text_proj(text_emb).unsqueeze(1)               # [B,1,embed_dim]
        mu_tok = self.mu_token.expand(B, -1, -1)                         # [B,1,embed_dim]
        logvar_tok = self.logvar_token.expand(B, -1, -1)                 # [B,1,embed_dim]

        tokens = torch.cat([mu_tok, logvar_tok, text_token, motion_tokens], dim=1)  # [B,16,embed_dim]

        # Transformer encoding
        for layer in self.layers:
            tokens = layer(tokens)  # No positional encoding needed (RoPE inside attention)
        tokens = self.norm(tokens)

        # 提取 μ, logσ²
        mu = self.mu_head(tokens[:, 0, :])      # [B, z_dim]
        logvar = self.logvar_head(tokens[:, 1, :])  # [B, z_dim]
        return mu, logvar


# --------------------------------------------------
# 3. 先验网络：保持不变（轻量级）
# --------------------------------------------------
class SAMPriorVAE(nn.Module):
    r"""
    先验分布：p_\theta(\Delta z | x_{-6:0}^{traj}) = \mathcal{N}(\mu_\Delta, \sigma_\Delta^2)
    
    输入序列（9 tokens）：
        [mu_token, logvar_token, traj_{-6}, ..., traj_{0}]
    """
    def __init__(self,
                 traj_dim: int = 36,
                 z_dim: int = 256,
                 d_model: int = 128,
                 nhead: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        seq_len = 2 + 7  # mu, logvar, 7 traj frames

        self.traj_proj = nn.Linear(traj_dim, d_model)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, past_traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = past_traj.shape[0]
        device = past_traj.device

        traj_tokens = self.traj_proj(past_traj)                          # [B,7,d]
        mu_tok = self.mu_token.expand(B, -1, -1)
        logvar_tok = self.logvar_token.expand(B, -1, -1)

        tokens = torch.cat([mu_tok, logvar_tok, traj_tokens], dim=1)     # [B,9,d]
        tokens = tokens + self.pos_embed.to(device)
        encoded = self.transformer(tokens)

        mu = self.mu_head(encoded[:, 0, :])
        logvar = self.logvar_head(encoded[:, 1, :])
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# --------------------------------------------------
# 4. Latent Motion 预测器：重构为Transformer Decoder (自回归)
# --------------------------------------------------
class LatentMotionPredictor(nn.Module):
    r"""
    自回归预测未来4帧latent motion (64-dim/帧)
    
    输入序列 (8 tokens):
        [z_token, past_{-6:-1} (6), current_0 (1)]
    
    输出序列 (4 tokens):
        [latent_1, latent_2, latent_3, latent_4]  // 通过自回归生成
    
    结构: Transformer Decoder with MQA + RoPE + RMSNorm + SwiGLU
    """
    def __init__(self,
                 z_dim: int = 256,
                 motion_dim: int = 272,
                 latent_dim: int = 64,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_seq_len = 8 + 4  # 8 context tokens + 4 prediction tokens
        
        # Token 投影
        self.z_proj = nn.Linear(z_dim, embed_dim)
        self.motion_proj = nn.Linear(motion_dim, embed_dim)
        
        # Learnable start token for autoregressive generation
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        
        # Output projection to latent space
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
        # Causal mask for autoregressive generation
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len) * float('-inf'), 
            diagonal=1
        ))
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                z_post: torch.Tensor,
                past_motion: torch.Tensor,
                current_motion: torch.Tensor,
                future_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            z_post: [B, z_dim] - posterior latent
            past_motion: [B, 6, motion_dim] - past 6 frames
            current_motion: [B, 1, motion_dim] - current frame
            future_latent: [B, 4, latent_dim] (optional) - ground truth for teacher forcing
        
        Returns:
            predicted_latent: [B, 4, latent_dim] - predicted latent motion
        """
        B = z_post.shape[0]
        device = z_post.device
        
        # Project context tokens
        z_tok = self.z_proj(z_post).unsqueeze(1)          # [B,1,embed_dim]
        past_tok = self.motion_proj(past_motion)           # [B,6,embed_dim]
        curr_tok = self.motion_proj(current_motion)        # [B,1,embed_dim]
        
        # Build context: [z, past_6, current]
        context = torch.cat([z_tok, past_tok, curr_tok], dim=1)  # [B,8,embed_dim]
        seq_len = context.shape[1]
        
        # Autoregressive generation
        if self.training or future_latent is not None:
            # Teacher forcing during training
            # Prepare input sequence: [context, future_latent_shifted]
            future_shifted = torch.cat([
                self.start_token.expand(B, -1, -1),  # [B,1,embed_dim]
                self.output_proj(future_latent)[:, :-1]  # [B,3,embed_dim]
            ], dim=1)
            input_seq = torch.cat([context, future_shifted], dim=1)  # [B,12,embed_dim]
            
            # Apply causal mask
            causal_mask = self.causal_mask[:input_seq.shape[1], :input_seq.shape[1]]
            
            # Decoder forward pass
            for layer in self.layers:
                input_seq = layer(input_seq, causal_mask)
            output = self.norm(input_seq)
            
            # Only take the last 4 positions for prediction
            latent_pred = self.output_proj(output[:, -4:, :])  # [B,4,latent_dim]
            return latent_pred
            
        else:
            # Autoregressive inference
            input_seq = context  # [B,8,embed_dim]
            predictions = []
            
            for i in range(4):
                # Current sequence length
                curr_len = input_seq.shape[1]
                # Apply causal mask for current length
                causal_mask = self.causal_mask[:curr_len, :curr_len]
                
                # Decode
                x = input_seq
                for layer in self.layers:
                    x = layer(x, causal_mask)
                x = self.norm(x)
                
                # Predict next token
                next_token = self.output_proj(x[:, -1, :])  # [B, latent_dim]
                predictions.append(next_token)
                
                # Prepare next input
                if i < 3:  # Only need to add for next iteration
                    next_embed = self.output_proj(next_token).unsqueeze(1)  # [B,1,embed_dim]
                    input_seq = torch.cat([input_seq, next_embed], dim=1)
            
            return torch.stack(predictions, dim=1)  # [B,4,latent_dim]


# --------------------------------------------------
# 5. 扩散模型：保持不变
# --------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    r"""
    正弦时间嵌入，将标量 timestep t 映射为 dim 维向量。
    公式：
        PE(t, 2i)   = \sin\left( t / 10000^{2i / d} \right)
        PE(t, 2i+1) = \cos\left( t / 10000^{2i / d} \right)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        exponents = torch.arange(half_dim, device=device) * -emb
        emb = torch.exp(exponents.unsqueeze(0) * time.unsqueeze(-1).float())
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class X0Denoiser(nn.Module):
    r"""
    x0-parameterization denoiser: \hat{x}_0 = f(x_t, t, c)
    """
    def __init__(self, motion_dim: int, cond_dim: int = 0, time_emb_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.use_cond = cond_dim > 0
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        in_dim = motion_dim + (cond_dim if self.use_cond else 0) + time_emb_dim
        layers = []
        for i in range(3):
            out_dim = hidden_dim if i < 2 else motion_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_emb(t.float())
        if self.use_cond:
            x = torch.cat([x_t, cond, t_emb], dim=1)
        else:
            x = torch.cat([x_t, t_emb], dim=1)
        return self.mlp(x)


class X0DiffusionHead(nn.Module):
    r"""
    DDPM with cosine schedule and x0-parameterization.
    
    训练损失：
        \mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \| \hat{x}_0(x_t, t) - x_0 \|^2
    
    Cosine schedule:
        \bar{\alpha}_t = \frac{\cos^2\left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2} \right)}{\cos^2\left( \frac{s}{1+s} \cdot \frac{\pi}{2} \right)}
    
    采样：8-step DDPM reverse process.
    """
    def __init__(self, motion_dim: int, cond_dim: int = 0, num_steps: int = 8, s: float = 0.008):
        super().__init__()
        self.motion_dim = motion_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.s = s

        # 预计算 cosine alpha_bar (1000 steps)
        t_cont = torch.linspace(0, 1, 1001)
        alphas_cumprod = self._cosine_alpha_cumprod(t_cont, s)[:-1]
        self.register_buffer('alphas_cumprod', torch.clamp(alphas_cumprod, 1e-8, 1-1e-8))

        # 8-step 采样 timesteps
        self.register_buffer('sample_timesteps', torch.linspace(999, 0, num_steps).long())
        self.denoiser = X0Denoiser(motion_dim, cond_dim)

    def _cosine_alpha_cumprod(self, t: torch.Tensor, s: float) -> torch.Tensor:
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].unsqueeze(-1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def training_loss(self, clean_motion: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = clean_motion.shape[0]
        device = clean_motion.device
        t = torch.randint(0, 1000, (B,), device=device).long()
        noise = torch.randn_like(clean_motion)
        x_t = self.q_sample(clean_motion, t, noise)
        x0_pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(x0_pred, clean_motion)

    @torch.no_grad()
    def sample(self, cond: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        B = cond.shape[0] if cond is not None else batch_size
        device = cond.device if cond is not None else torch.device('cpu')
        x_t = torch.randn(B, self.motion_dim, device=device)
        timesteps = self.sample_timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            x0_pred = self.denoiser(x_t, t_batch, cond)

            if t == 0:
                x_t = x0_pred
                break

            # 下一个 timestep
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else torch.tensor(0, device=device)

            # DDPM reverse process (Eq. 7 in DDPM paper)
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev]
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1 - alpha_t

            # 后验均值系数
            coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
            coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
            posterior_mean = coef1 * x0_pred + coef2 * x_t

            # 后验方差（DDPM 固定）
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            if t_prev > 0:
                noise = torch.randn_like(x_t)
                x_t = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x_t = posterior_mean

        return x_t


# --------------------------------------------------
# 6. SAMP 框架整合（适配新架构）
# --------------------------------------------------
class SAMP_Framework(nn.Module):
    r"""
    完整 SAMP 框架：
        z_base = W_text * t
        z_post ~ q(z | x_{-6:6}, t)          ← 训练
        Δz ~ p(Δz | x_{-6:0}^{traj})          ← 推理
        z = z_base + Δz
        latent = f(z, x_{-6:0})               // 自回归预测
        x_{1:4} ~ diffusion(latent)
    """
    def __init__(self,
                 joint_num: int = 22,
                 clip_model_name: str = "ViT-B/32",
                 z_dim: int = 256,
                 device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_num = joint_num
        self.motion_dim_per_frame = 272  # Fixed for HumanML3D
        self.total_motion_dim = 4 * self.motion_dim_per_frame   # 1088
        self.latent_dim = 64
        self.total_latent_dim = 4 * self.latent_dim  # 256

        # 文本编码器
        self.clip_encoder = RealCLIPTextEncoder(model_name=clip_model_name, device=self.device)
        self.text_to_z_proj = nn.Linear(512, z_dim)

        # 后验编码器 (新架构)
        self.posterior_encoder = MotionPosteriorEncoder(
            motion_dim=self.motion_dim_per_frame,
            embed_dim=256, 
            num_heads=4, 
            num_layers=4, 
            z_dim=z_dim
        )

        # 先验网络（Δz）
        self.motion_prior = SAMPriorVAE(
            traj_dim=36, 
            z_dim=z_dim, 
            d_model=128, 
            nhead=2, 
            num_layers=2
        )

        # Latent motion 预测器 (新架构 - 自回归)
        self.motion_decoder = LatentMotionPredictor(
            z_dim=z_dim,
            motion_dim=self.motion_dim_per_frame,
            latent_dim=self.latent_dim,
            embed_dim=256,
            num_heads=4,
            num_layers=2
        )

        # 扩散头
        self.diffusion_head = X0DiffusionHead(
            motion_dim=self.total_motion_dim,
            cond_dim=self.total_latent_dim,
            num_steps=8
        )

        self.to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_gaussian_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl = 0.5 * torch.sum(logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1, dim=1)
        return kl.mean()

    def forward(self,
                texts: List[str],
                post_input: torch.Tensor,        # [B,13,272] - 13帧条件输入
                x: torch.Tensor,                 # [B,7,272] - 7帧历史动作 (x_{-6:0})
                y: torch.Tensor,                 # [B,7,272] - 7帧未来动作 (x_{0:6}), 但只用x_{1:4}
                traj: torch.Tensor,              # [B,6,36] - 6帧轨迹特征
                mode: str = 'train'):
        B = post_input.shape[0]
        text_emb = self.clip_encoder.encode_texts(texts).to(self.device)
        base_z = self.text_to_z_proj(text_emb)

        if mode == 'train':
            # 准备后验输入
            full_motion = post_input.to(self.device)  # [B,13,272]
            
            # 准备轨迹输入 (需要7帧, 但数据集提供6帧, 补充当前帧)
            # 数据集traj是[B,6,36], 我们需要[B,7,36]
            # 使用x的最后6帧作为历史，当前帧用x的第6帧（索引5）? 
            # 注意: x是7帧: [x_{-6}, x_{-5}, ..., x_0]
            # traj应该对应x_{-6}到x_{-1} (6帧)，我们补充x_0作为第7帧
            # 但traj特征需要重新计算，这里简化：直接复制最后一帧
            past_traj = torch.cat([traj, traj[:, -1:, :]], dim=1)  # [B,7,36] 临时方案
            
            # 后验 z_post
            mu_post, logvar_post = self.posterior_encoder(full_motion, text_emb)
            z_post = self.reparameterize(mu_post, logvar_post)

            # 残差后验
            mu_delta_post = mu_post - base_z
            logvar_delta_post = logvar_post

            # 先验 p(Δz | traj)
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(self.device))

            # 准备decoder输入
            past_motion = x[:, :6, :]  # [B,6,272] - x_{-6:-1}
            current_motion = x[:, -1:, :]  # [B,1,272] - x_0
            
            # 准备ground truth latent for teacher forcing
            # 取y的前4帧作为future motion (x_1 to x_4)
            future_motion = y[:, :4, :]  # [B,4,272]
            # 随机初始化latent作为ground truth (实际训练时会被预测替代)
            future_latent_gt = torch.randn(B, 4, self.latent_dim, device=self.device)
            
            # 预测 latent motion (with teacher forcing)
            latent_motion = self.motion_decoder(
                z_post, 
                past_motion, 
                current_motion,
                future_latent=future_latent_gt
            )  # [B,4,64]
            latent_flat = latent_motion.view(B, -1)

            # 扩散 loss (预测x_1 to x_4)
            clean_future = future_motion.reshape(B, -1).to(self.device)
            diffusion_loss = self.diffusion_head.training_loss(clean_future, latent_flat)

            return {
                'diffusion_loss': diffusion_loss,
                'mu_delta_post': mu_delta_post,
                'logvar_delta_post': logvar_delta_post,
                'mu_delta_prior': mu_delta_prior,
                'logvar_delta_prior': logvar_delta_prior,
                'base_z': base_z
            }

        elif mode == 'inference':
            # 从先验采样 Δz
            # 构建7帧轨迹 (同上)
            past_traj = torch.cat([traj, traj[:, -1:, :]], dim=1)  # [B,7,36]
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(self.device))
            delta_z = self.motion_prior.sample(mu_delta_prior, logvar_delta_prior)
            z_sample = base_z + delta_z

            past_motion = x[:, :6, :]  # [B,6,272]
            current_motion = x[:, -1:, :]  # [B,1,272]
            
            # 自回归预测latent motion (no teacher forcing)
            latent_motion = self.motion_decoder(z_sample, past_motion, current_motion)  # [B,4,64]
            latent_flat = latent_motion.view(B, -1)

            # 采样未来4帧动作
            sampled_flat = self.diffusion_head.sample(latent_flat)
            return sampled_flat.view(B, 4, -1)  # [B,4,272]

        else:
            raise ValueError("mode must be 'train' or 'inference'")

    def loss_function(self,
                      diffusion_loss: torch.Tensor,
                      mu_delta_post: torch.Tensor,
                      logvar_delta_post: torch.Tensor,
                      mu_delta_prior: torch.Tensor,
                      logvar_delta_prior: torch.Tensor,
                      lambda_kl: float = 1.0,
                      lambda_l2: float = 0.1) -> dict:
        kl_delta = self._kl_gaussian_gaussian(
            mu_delta_post, logvar_delta_post,
            mu_delta_prior, logvar_delta_prior
        )
        l2_mean = F.mse_loss(mu_delta_post, mu_delta_prior)
        total_loss = diffusion_loss + lambda_kl * kl_delta + lambda_l2 * l2_mean
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss.item(),
            'kl_delta': kl_delta.item(),
            'l2_mean': l2_mean.item()
        }