import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ----------------------------
# CLIP Text Encoder
# ----------------------------
try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

class CLIPTextEncoder(nn.Module):
    r"""
    使用 OpenAI CLIP 文本编码器，提取语义向量。
    输出维度通常为 512 (ViT-B/32) 或 1024 (ViT-L/14)。
    """
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        super().__init__()
        # 1. 自动处理设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not _HAS_CLIP:
            raise ImportError("请安装 CLIP: pip install git+https://github.com/openai/CLIP.git")

        # 2. 加载模型
        try:
            self.model, _ = clip.load(model_name, device=self.device, jit=False)
            self.model.eval()
            # 3. 彻底冻结
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"CLIP 加载失败: {e}")
            raise

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        输入文本列表，返回归一化后的 CLIP 文本嵌入
        """
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        # CLIP 内部有 tokenize 逻辑，它会自动处理截断（context_length=77）
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # 提取特征
        text_features = self.model.encode_text(tokens).float()
        
        # L2 归一化
        return F.normalize(text_features, p=2, dim=-1)


# ----------------------------
# DistilBERT Text Encoder
# ----------------------------
class DistilBERTTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert/distilbert-base-uncased",
        max_length=64,
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 1. 永久冻结 BERT 主干网络
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # 2. 投影层（默认 requires_grad=True，用于训练）
        self.proj = nn.Linear(768, 512)

    def forward(self, texts):
        """
        训练模式：BERT 冻结，proj 层保留梯度
        """
        device = next(self.parameters()).device
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        # BERT 部分不计算梯度
        with torch.no_grad():
            outputs = self.model(**batch)
            h = outputs.last_hidden_state[:, 0]  # [B, 768]

        return F.normalize(self.proj(h), p=2, dim=-1)  # [B, 512]

    @torch.no_grad()
    def encode_texts(self, texts):
        """
        推理模式：完全不计算梯度，用于提取特征
        """
        # 直接复用 forward 逻辑，但装饰器确保了整个过程无梯度
        return self.forward(texts)

# ----------------------------
# RMSNorm
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return self.weight * x


# ----------------------------
# Rotary Position Embedding (RoPE)
# ----------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [B, L, H, D]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]  # [L, 1, D//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ----------------------------
# SwiGLU MLP
# ----------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(2 * (dim * 4) / 3)
            # 向上取整到 multiple_of 的倍数 (例如 256)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# ----------------------------
# Multi-Head Attention (MHA) with RoPE
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis[:L])

        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
    

# ----------------------------
# Multi-Query Attention (MQA) with RoPE
# ----------------------------
class MultiQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)  # single shared KV
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).unsqueeze(2)  # [B, L, 1, D_h]
        v = self.v_proj(x).unsqueeze(2)

        # RoPE
        q, k_single = apply_rotary_emb(q, k.squeeze(2), freqs_cis[:L])
        k = k_single.unsqueeze(2)

        # reshape
        q = q.transpose(1, 2)                  # [B, H, L, D_h]
        k = k.transpose(1, 2).expand(-1, self.n_heads, -1, -1)
        v = v.transpose(1, 2).expand(-1, self.n_heads, -1, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


# ----------------------------
# Motion Posterior Encoder
# ----------------------------
class MotionPosteriorEncoder(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        z_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        seq_len = 2 + 1 + 13  # mu, logvar, text, 13 motions
        assert seq_len <= max_seq_len

        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Precompute RoPE
        self.freqs_cis = precompute_freqs_cis(d_model // nhead, max_seq_len)

        self.layers = nn.ModuleList([
            PosteriorEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, motion_seq: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = motion_seq.shape[0]
        device = motion_seq.device

        motion_tokens = self.motion_proj(motion_seq)                     # [B,13,d]
        text_token = text_emb.unsqueeze(1)                               # [B,1,d]
        mu_tok = self.mu_token.expand(B, -1, -1)
        logvar_tok = self.logvar_token.expand(B, -1, -1)

        tokens = torch.cat([mu_tok, logvar_tok, text_token, motion_tokens], dim=1)  # [B,16,d]

        freqs_cis = self.freqs_cis.to(device)
        x = tokens
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)

        mu = self.mu_head(x[:, 0, :])
        logvar = self.logvar_head(x[:, 1, :])
        return mu, logvar


# --------------------------
# Posterior Encoder Layer
# --------------------------
class PosteriorEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out = self.attn(x, freqs_cis)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


# ----------------------------
# Latent Motion Predictor (Decoder)
# ----------------------------
class LatentMotionPredictor(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        motion_dim: int = 272,
        latent_motion_dim: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_motion_dim = latent_motion_dim
        self.seq_len = 1 + 6 + 1  # z, past6, curr1
        assert self.seq_len <= max_seq_len

        self.z_proj = nn.Linear(z_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)

        self.freqs_cis = precompute_freqs_cis(d_model // nhead, max_seq_len)

        self.layers = nn.ModuleList([
            LatentDecoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_motion_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        z_post: torch.Tensor,
        past_motion: torch.Tensor,
        current_motion: torch.Tensor,
    ) -> torch.Tensor:
        B = z_post.shape[0]
        device = z_post.device

        z_tok = self.z_proj(z_post).unsqueeze(1)      # [B,1,d]
        past_tok = self.motion_proj(past_motion)     # [B,6,d]
        curr_tok = self.motion_proj(current_motion).unsqueeze(1)  # [B,1,d]

        x = torch.cat([z_tok, past_tok, curr_tok], dim=1)  # [B,8,d]
        L = x.shape[1]

        freqs_cis = self.freqs_cis[:L].to(device)

        # causal mask
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)

        latent_seq = self.output_proj(x)  # [B, L, latent_motion_dim]

        if self.training:
            # training: return full latent sequence
            return latent_seq
        else:
            # inference: only return last step
            return latent_seq[:, -1:, :]


# --------------------------
# Latent Decoder Layer
# --------------------------
class LatentDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x, freqs_cis, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


# ----------------------------
# SAMPriorVAE: Prior Network
# ----------------------------
class SAMPriorVAE(nn.Module):
    r"""
    先验网络：p_\theta(\Delta z | x_{-6:-1}^{traj}) = \mathcal{N}(\mu_\Delta, \sigma_\Delta^2)
    采用 MHA + RoPE + RMSNorm + SwiGLU 架构

    输入序列（8 tokens）：
        [mu_token, logvar_token, traj_{-6}, ..., traj_{-1}]
    """
    def __init__(self,
                 traj_dim: int = 44,
                 z_dim: int = 512,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.seq_len = 2 + 6  # mu, logvar, 6 traj frames (total 8)

        # 1. 输入投影
        self.traj_proj = nn.Linear(traj_dim, d_model)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 2. 预计算 RoPE 频率 (旋转位置编码)
        self.freqs_cis = precompute_freqs_cis(d_model // nhead, self.seq_len)

        # 3. 堆叠自定义 Transformer 层 (Modern LLM Style)
        self.layers = nn.ModuleList([
            PriorLayer(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # 4. 输出头
        self.prior_scale = nn.Parameter(torch.full((1,), 0.01))
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

        # 组装 Tokens
        traj_tokens = self.traj_proj(past_traj)                          # [B, 6, d]
        mu_tok = self.mu_token.expand(B, -1, -1)
        logvar_tok = self.logvar_token.expand(B, -1, -1)
        x = torch.cat([mu_tok, logvar_tok, traj_tokens], dim=1)         # [B, 8, d]

        # 准备 RoPE 频率
        freqs_cis = self.freqs_cis[:self.seq_len].to(device)

        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x, freqs_cis)
        
        x = self.final_norm(x)

        # 提取结果
        mu = self.mu_head(x[:, 0, :]) * self.prior_scale
        logvar = self.logvar_head(x[:, 1, :]) * self.prior_scale
        return mu, logvar


# ----------------------------
# 辅助层结构 (MHA + SwiGLU + RMSNorm)
# ----------------------------
class PriorLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        # Pre-Norm 结构，使用 RMSNorm
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.attention_norm = RMSNorm(d_model)
        
        self.ffn = SwiGLU(d_model)
        self.ffn_norm = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # 注意力路径 (Residual + Pre-Norm)
        h = x + self.dropout(self.attention(self.attention_norm(x), freqs_cis))
        # FFN 路径 (Residual + Pre-Norm)
        out = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return out



import torch
import torch.nn as nn

# ----------------------------
# 辅助组件：时间步嵌入
# ----------------------------
class TimestepEmbedder(nn.Module):
    """将标量时间步 t 转换为特征向量。"""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t, t_emb_input):
        # t_emb_input 是预计算的正弦位置编码
        return self.mlp(t_emb_input)

# ----------------------------
# DiT Block: 带条件注入的 Transformer 层
# ----------------------------
class DiTBlock(nn.Module):
    """
    使用 Adaptive Layer Norm (adaLN) 注入条件的 Transformer 块。
    公式: x = x + scale * Attention(norm(x)) + ...
    """
    def __init__(self, hidden_size, nhead, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, nhead, dropout) # 复用之前的 MHA
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLU(hidden_size)
        
        # adaLN 调制参数投影: 为 norm1 和 norm2 分别生成 scale, shift 和 gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis):
        # 从条件向量 c 中预测调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # 调制 LayerNorm 1 并执行注意力
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis
        )
        # 调制 LayerNorm 2 并执行 FFN
        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ----------------------------
# 主模型：MotionDiffusionTransformer
# ----------------------------
class MotionDiffusionTransformer(nn.Module):
    def __init__(self, 
                 motion_dim=272, 
                 latent_cond_dim=64, 
                 hidden_size=512, 
                 nhead=8, 
                 num_layers=8):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, hidden_size)
        self.cond_proj = nn.Linear(latent_cond_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # RoPE 频率预计算 (假设预测 1 帧，但为了扩展性预留序列长度)
        self.freqs_cis = precompute_freqs_cis(hidden_size // nhead, 1)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, nhead, 0.1) for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, motion_dim)
        )
        # 最后的 adaLN 调制
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x_t, t, latent_motion, t_emb_sinusoidal):
        """
        Args:
            x_t: [B, 1, 272] 当前加噪的动作
            t: [B] 时间步
            latent_motion: [B, 1, 64] 来自步骤 4 的条件
            t_emb_sinusoidal: [B, hidden_size] 预计算的时间编码
        """
        # 1. 嵌入输入与条件
        x = self.motion_proj(x_t) # [B, 1, d]
        
        # 融合时间信息与 latent_motion 作为全局条件 c
        t_feat = self.t_embedder(t, t_emb_sinusoidal) # [B, d]
        c_feat = self.cond_proj(latent_motion).squeeze(1) # [B, d]
        c = t_feat + c_feat # 最终条件向量 [B, d]

        # 2. Transformer 层处理
        freqs_cis = self.freqs_cis.to(x.device)
        for block in self.blocks:
            x = block(x, c, freqs_cis)

        # 3. 最终投影回动作空间
        shift, scale = self.final_modulation(c).chunk(2, dim=1)
        x = self.final_layer[0](x) # Norm
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # Final modulation
        return self.final_layer[1](x) # Linear to 272



class X0DiffusionHead(nn.Module):
    r"""
    基于 x0 预测和 DDIM 采样算法的扩散模型头部。
    支持 Cosine Schedule 和高效的短步数确定性采样。
    """
    def __init__(self, motion_dim: int, cond_dim: int = 0, num_steps: int = 8, s: float = 0.008):
        super().__init__()
        self.motion_dim = motion_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.s = s

        # 1. 预计算 Cosine Alpha_bar (1000 steps 基础)
        # Cosine schedule 相比 Linear schedule 在步数较少时能保留更多信息
        t_cont = torch.linspace(0, 1, 1001)
        alphas_cumprod = self._cosine_alpha_cumprod(t_cont, s)[:-1]
        self.register_buffer('alphas_cumprod', torch.clamp(alphas_cumprod, 1e-8, 1-1e-8))

        # 2. 采样时间步 (例如从 999 到 0 的均匀分布)
        self.register_buffer('sample_timesteps', torch.linspace(999, 0, num_steps).long())
        
        # 3. 内部去噪网络 (由外部定义)
        # self.denoiser = X0Denoiser(motion_dim, cond_dim) 

    def _cosine_alpha_cumprod(self, t: torch.Tensor, s: float) -> torch.Tensor:
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向加噪过程: x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].view(-1, 1) # 适配维度
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def training_loss(self, clean_motion: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 MSE 损失: 直接预测原始信号 x0"""
        B = clean_motion.shape[0]
        device = clean_motion.device
        t = torch.randint(0, 1000, (B,), device=device).long()
        noise = torch.randn_like(clean_motion)
        
        x_t = self.q_sample(clean_motion, t, noise)
        x0_pred = self.denoiser(x_t, t, cond)
        
        return F.mse_loss(x0_pred, clean_motion)

    @torch.no_grad()
    def sample(self, cond: Optional[torch.Tensor] = None, batch_size: int = 1, eta: float = 0.0) -> torch.Tensor:
        r"""
        使用 DDIM 算法进行采样
        Args:
            cond: 条件张量
            batch_size: 如果无条件，指定的 batch 大小
            eta: 随机性缩放因子。eta=0 为确定性 DDIM，eta=1 为随机 DDPM。
        """
        B = cond.shape[0] if cond is not None else batch_size
        device = cond.device if cond is not None else torch.device('cpu')
        
        # 从纯高斯噪声开始
        x_t = torch.randn(B, self.motion_dim, device=device)
        timesteps = self.sample_timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            
            # 模型直接预测 x0
            x0_pred = self.denoiser(x_t, t_batch, cond)

            if t == 0:
                x_t = x0_pred
                break

            # 获取当前和前一个时间点的 alpha_bar
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else torch.tensor(0, device=device)
            
            ab_t = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[t_prev]

            # --- DDIM 核心逻辑 ---
            # 1. 根据 x_t 和预测的 x0 算出隐含的噪声 eps
            # 公式由来: x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps
            eps_theta = (x_t - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1 - ab_t)

            # 2. 计算随机性方差 sigma
            # 若 eta=0，则 sigma=0，采样过程变为确定性
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)

            # 3. 计算指向 x_{t-1} 的预测方向 (Predicted direction to x_t)
            # 这里的方向是基于 eps_theta 指向当前噪声趋势
            dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps_theta
            
            # 4. 组合得到 x_{t-1}
            x_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt
            
            if sigma > 0:
                noise = torch.randn_like(x_t)
                x_t = x_prev + sigma * noise
            else:
                x_t = x_prev

        return x_t












class SAMPFramework(nn.Module):
    def __init__(
        self,
        text_encoder: DistilBERTTextEncoder,        # TextEncoder
        posterior_encoder: MotionPosteriorEncoder,   # MotionPosteriorEncoder
        prior_vae: SAMPriorVAE,           # SAMPriorVAE
        latent_predictor: LatentMotionPredictor,    # LatentMotionPredictor
        diffusion_head: nn.Module,      # DiffusionHead
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.posterior_encoder = posterior_encoder
        self.prior_vae = prior_vae
        self.latent_predictor = latent_predictor
        self.diffusion_head = diffusion_head

    def forward(
        self, 
        motion_seq: torch.Tensor,   # [B, 13, motion_dim] (x_{-6:6})
        past_traj: torch.Tensor,    # [B, 7, traj_dim] (x_{-6:0})
        texts: List[str]
    ):
        """
        训练阶段 Pipeline
        """
        # 1. Base 提取 (模长为 1)
        z_base = self.text_encoder.encode_texts(texts) # [B, 512]

        # 2. 后验采样 (Posterior): q(z | full_motion, text)
        mu_post, logvar_post = self.posterior_encoder(motion_seq, z_base)
        z_post = self.reparameterize(mu_post, logvar_post) # [B, 512]

        # 3. 先验分布 (Prior): p(Δz | past_traj)
        mu_prior, logvar_prior = self.prior_vae(past_traj)

        # 4. 残差融合 (训练时使用后验 z)
        # 这里的目标是让 z_post 逼近 z_base + Δz
        # 换句话说，让 Δz 学习 z_post - z_base
        z_final = z_post 

        # 5. 潜在动作预测
        # 输入：z_final, past_motion (x_{-6:-1}), current_motion (x_0)
        past_motion = motion_seq[:, :6, :]
        curr_motion = motion_seq[:, 7, :]
        
        latent_out = self.latent_predictor(z_final, past_motion, curr_motion)

        return {
            "latent_out": latent_out,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z_base": z_base
        }

    def sample_motion(
        self, 
        past_traj: torch.Tensor, 
        past_motion: torch.Tensor, 
        curr_motion: torch.Tensor, 
        texts: List[str]
    ):
        """
        推理阶段 Pipeline (z = z_base + Δz)
        """
        self.eval()
        with torch.no_grad():
            # 1. z_base
            z_base = self.text_encoder.encode_texts(texts)

            # 2. Δz 采样自先验
            mu_prior, logvar_prior = self.prior_vae(past_traj)
            dz = self.reparameterize(mu_prior, logvar_prior)
            
            # 3. 残差相加
            z_inference = z_base + dz

            # 4. 预测下一帧的 latent (用于给 Diffusion 做 condition)
            latent_motion = self.latent_predictor(z_inference, past_motion, curr_motion)

        return latent_motion

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, model_outputs, target_latent, kl_weight=1.0):
        """
        Loss = 重建损失 + KL散度(后验 || 先验)
        """
        out = model_outputs
        
        # 1. Reconstruction Loss (潜在空间对齐)
        recon_loss = F.mse_loss(out["latent_out"], target_latent)

        # 2. KL Divergence (Alignment Loss)
        # 核心：让先验网络预测出的 Δz 分布 靠近 后验网络相对于 base 的偏移
        # 公式: KL( q(z|x,t) || p(z_base + Δz | x_traj) )
        mu_p = out["mu_prior"]
        var_p = out["logvar_prior"].exp()
        mu_q = out["mu_post"] - out["z_base"] # 后验相对于 base 的期望偏移
        var_q = out["logvar_post"].exp()

        # 计算两个高斯分布之间的 KL
        kl_loss = 0.5 * (torch.log(var_p/var_q) + (var_q + (mu_q - mu_p)**2)/var_p - 1).sum(dim=-1).mean()

        return recon_loss + kl_weight * kl_loss






def test_distilbert_encoder():
    """
    针对 DistilBERTTextEncoder 的单元测试函数
    """
    print("开始测试 DistilBERTTextEncoder...")
    
    # 1. 检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前测试环境设备: {device}")

    try:
        # 2. 初始化模型并移动到指定设备
        # 注意：在我的改进建议中，模型初始化后应通过 .to(device) 显式移动
        text_encoder = DistilBERTTextEncoder().to(device)

        # 3. 准备测试数据
        texts = ["A person is walking", "Jumping in place"]
        
        # 4. 执行编码
        # 注意：在 encode_texts 内部会自动获取模型所在的 device
        embeds = text_encoder.encode_texts(texts)

        # 5. 验证结果
        expected_shape = (len(texts), 768)  # 预期输出维度
        
        # 验证维度
        assert embeds.shape == expected_shape, f"维度错误: 预期 {expected_shape}, 实际 {embeds.shape}"
        # 验证设备
        assert embeds.device.type == device.type, f"设备不匹配: 预期 {device.type}, 实际 {embeds.device.type}"
        # 验证是否已归一化 (L2范数应接近1)
        norms = torch.norm(embeds, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms)), "输出特征未进行单位范数归一化"

        print("--- 测试通过！ ---")
        print(f"输出特征维度: {embeds.shape}")
        print(f"输出特征设备: {embeds.device}")
        print(f"特征范数样例: {norms.detach().cpu().numpy()}")

    except Exception as e:
        print(f"--- 测试失败！ ---")
        print(f"错误信息: {e}")


def test_motion_encoder():
    # 参数设置
    B, motion_dim, seq_len_motion = 4, 272, 13
    text_dim = 512
    z_dim = 256
    
    # 实例化模型
    model = MotionPosteriorEncoder(
        motion_dim=motion_dim, 
        z_dim=z_dim,
        max_seq_len=16
    )
    
    # 模拟输入
    motion_seq = torch.randn(B, seq_len_motion, motion_dim) # [4, 13, 272]
    text_emb = torch.randn(B, text_dim)                    # [4, 512]
    
    try:
        mu, logvar = model(motion_seq, text_emb)
        
        print("✅ 前向传播成功！")
        print(f"输入 Motion 维度: {motion_seq.shape}")
        print(f"输出 Mu 维度:     {mu.shape}     (预期: [{B}, {z_dim}])")
        print(f"输出 Logvar 维度: {logvar.shape} (预期: [{B}, {z_dim}])")
        
        # 简单的数值检查
        assert not torch.isnan(mu).any(), "输出包含 NaN"
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("-" * 30)


def test_latent_motion_predictor():
    print("开始测试 LatentMotionPredictor...")
    
    # 1. 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, z_dim, motion_dim = 4, 256, 272
    latent_motion_dim = 128
    
    # 2. 初始化模型
    model = LatentMotionPredictor(
        z_dim=z_dim,
        motion_dim=motion_dim,
        latent_motion_dim=latent_motion_dim,
        d_model=512,
        max_seq_len=16
    ).to(device)

    # 3. 构造模拟输入
    # z_post: 来自 Encoder 的隐变量
    # past_motion: 过去的 6 帧动作
    # current_motion: 当前的 1 帧动作
    z_post = torch.randn(B, z_dim).to(device)
    past_motion = torch.randn(B, 6, motion_dim).to(device)
    current_motion = torch.randn(B, motion_dim).to(device)

    # 4. 测试训练模式 (Training Mode)
    model.train()
    out_train = model(z_post, past_motion, current_motion)
    # 序列长度 L = 1(z) + 6(past) + 1(curr) = 8
    expected_train_shape = (B, 8, latent_motion_dim)
    
    assert out_train.shape == expected_train_shape, f"训练模式维度错误: {out_train.shape}"
    print(f"✅ 训练模式输出维度正确: {out_train.shape}")

    # 5. 测试推理模式 (Inference Mode)
    model.eval()
    with torch.no_grad():
        out_eval = model(z_post, past_motion, current_motion)
    
    # 推理模式仅返回最后一帧
    expected_eval_shape = (B, 1, latent_motion_dim)
    
    assert out_eval.shape == expected_eval_shape, f"推理模式维度错误: {out_eval.shape}"
    print(f"✅ 推理模式输出维度正确: {out_eval.shape}")

    # 6. 验证因果掩码逻辑 (简单检查数值稳定性)
    assert not torch.isnan(out_eval).any(), "输出包含 NaN"
    print("✅ 数值稳定性检查通过。")
    print("-" * 30)


def test_sam_prior_vae():
    print("开始测试 SAMPriorVAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型 (traj_dim 设为 44)
    model = SAMPriorVAE(traj_dim=44, z_dim=512, d_model=128).to(device)
    model.eval()

    # 2. 构造输入: [Batch, Sequence_Len=6, Traj_Dim=44]
    B = 4
    past_traj = torch.randn(B, 6, 44).to(device)

    # 3. 前向传播
    try:
        mu, logvar = model(past_traj)
        
        # 4. 验证维度
        print(f"输入形状: {past_traj.shape}")
        print(f"输出 Mu 形状: {mu.shape} (预期: [{B}, 512])")
        print(f"输出 Logvar 形状: {logvar.shape} (预期: [{B}, 512])")

        assert mu.shape == (B, 512)
        assert logvar.shape == (B, 512)
        
        # 5. 验证 prior_scale 的初始化效果
        # 由于 prior_scale = 0.01，初始输出应该非常接近 0
        print(f"Mu 的均值 (由于 scale=0.01 应该接近0): {mu.mean().item():.6f}")
        
        print("\n✅ 测试通过：维度对齐且前向传播无误。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")




if __name__ == '__main__':
    # test_distilbert_encoder()
    test_motion_encoder()
    test_latent_motion_predictor()
    test_sam_prior_vae()