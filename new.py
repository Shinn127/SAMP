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


# ----------------------------
# 基础组件
# ----------------------------
class AdaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, shift, scale):
        x = self.norm(x)
        return x * (1 + scale) + shift


class DiffusionMLPBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.adaln1 = AdaLN(hidden_size)
        self.adaln2 = AdaLN(hidden_size)

        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=1)
        x = x + gate1 * self.mlp1(self.adaln1(x, shift1, scale1))      
        x = x + gate2 * self.mlp2(self.adaln2(x, shift2, scale2))      
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_emb_input: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb_input)


class MotionDiffusionMLP(nn.Module):
    def __init__(self, motion_dim=272, latent_cond_dim=128, hidden_size=512, num_layers=4):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_cond_dim = latent_cond_dim
        self.hidden_size = hidden_size

        self.motion_proj = nn.Linear(motion_dim, hidden_size)
        self.cond_proj = nn.Linear(latent_cond_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiffusionMLPBlock(hidden_size) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, motion_dim)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor, latent_motion: torch.Tensor) -> torch.Tensor:
        x = self.motion_proj(x_t)
        t_feat = self.t_embedder(t_embed)
        c_feat = self.cond_proj(latent_motion)
        c = t_feat + c_feat

        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.final_modulation(c).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + scale) + shift
        return self.final_linear(x)


class X0DiffusionHead(nn.Module):
    def __init__(self, motion_dim: int = 272, cond_dim: int = 128, num_steps: int = 8, s: float = 0.008, hidden_size: int = 512):
        super().__init__()
        self.motion_dim = motion_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.s = s
        self.hidden_size = hidden_size

        t_cont = torch.linspace(0, 1, 1001)
        alphas_cumprod = self._cosine_alpha_cumprod(t_cont, s)[:-1]
        self.register_buffer('alphas_cumprod', torch.clamp(alphas_cumprod, 1e-8, 1 - 1e-8))
        self.register_buffer('precomputed_t_emb', self._generate_timestep_embeddings(1000, hidden_size))
        self.register_buffer('sample_timesteps', torch.linspace(999, 0, num_steps).long())

        self.denoiser = MotionDiffusionMLP(
            motion_dim=motion_dim,
            latent_cond_dim=cond_dim,
            hidden_size=hidden_size,
            num_layers=4
        )

    def _cosine_alpha_cumprod(self, t: torch.Tensor, s: float) -> torch.Tensor:
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def _generate_timestep_embeddings(self, num_timesteps: int, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_timesteps, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def forward(self, clean_motion: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = clean_motion.shape
        device = clean_motion.device

        clean_flat = clean_motion.view(B * T, -1)
        cond_flat = cond.view(B * T, -1)

        t_per_batch = torch.randint(0, 1000, (B,), device=device).long()
        t_flat = t_per_batch.repeat_interleave(T)
        t_emb_flat = self.precomputed_t_emb[t_flat]

        noise = torch.randn_like(clean_flat)
        x_t_flat = self.q_sample(clean_flat, t_flat, noise)
        x0_pred_flat = self.denoiser(x_t_flat, t_emb_flat, cond_flat)

        return x0_pred_flat, clean_flat

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        B, T, _ = cond.shape
        device = cond.device
        total_tokens = B * T

        x_t_flat = torch.randn(total_tokens, self.motion_dim, device=device)
        cond_flat = cond.view(total_tokens, -1)

        timesteps = self.sample_timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            t_flat = t_batch.repeat_interleave(T)
            t_emb_flat = self.precomputed_t_emb[t_flat]

            x0_pred_flat = self.denoiser(x_t_flat, t_emb_flat, cond_flat)

            if t == 0:
                x_t_flat = x0_pred_flat
                break

            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            ab_t = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[t_prev]

            eps_theta = (x_t_flat - torch.sqrt(ab_t) * x0_pred_flat) / torch.sqrt(1 - ab_t)
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps_theta
            x_prev = torch.sqrt(ab_prev) * x0_pred_flat + dir_xt

            if sigma > 0:
                x_t_flat = x_prev + sigma * torch.randn_like(x_t_flat)
            else:
                x_t_flat = x_prev

        return x_t_flat.view(B, T, -1)


class SAMPFramework(nn.Module):
    def __init__(
        self,
        text_encoder: DistilBERTTextEncoder,        # TextEncoder
        posterior_encoder: MotionPosteriorEncoder,   # MotionPosteriorEncoder
        prior_vae: SAMPriorVAE,           # SAMPriorVAE
        latent_predictor: LatentMotionPredictor,    # LatentMotionPredictor
        diffusion_head: X0DiffusionHead,      # DiffusionHead
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.posterior_encoder = posterior_encoder
        self.prior_vae = prior_vae
        self.latent_predictor = latent_predictor
        self.diffusion_head = diffusion_head

    def forward(
        self,
        texts: List[str],
        motion_seq: torch.Tensor,   # [B, 13, motion_dim]
        x: torch.Tensor,            # [B, 7, motion_dim]
        y: torch.Tensor,            # [B, 7, motion_dim]
        past_traj: torch.Tensor,    # [B, 6, traj_dim]
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
        past_motion = x[:, :6, :]
        curr_motion = x[:, 6, :]
        
        latent_out = self.latent_predictor(z_final, past_motion, curr_motion)

        x0_pred_flat, clean_flat = self.diffusion_head(y, latent_out[:, 1:, :].contiguous())

        return {
            "latent_out": latent_out,
            "x0_pred_flat": x0_pred_flat,
            "clean_flat": clean_flat,
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

        total_loss = recon_loss + kl_weight * kl_loss

        # Optional diffusion reconstruction term if forward() output includes it.
        if "x0_pred_flat" in out and "clean_flat" in out:
            diff_loss = F.mse_loss(out["x0_pred_flat"], out["clean_flat"])
            total_loss = total_loss + diff_loss

        return total_loss






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
        expected_shape = (len(texts), 512)  # 预期输出维度
        
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


def test_samp_framework():
    print("开始集成测试: SAMPFramework...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 假设统一维度为 512 (根据你的 PriorVAE 设置)
    z_dim = 512
    motion_dim = 272
    traj_dim = 44
    
    # 1. 模拟初始化子模块 (实际使用时替换为你的类实例)
    # 注意：确保这些 mock 对象的输出维度相互匹配
    framework = SAMPFramework(
        text_encoder=DistilBERTTextEncoder(),
        posterior_encoder=MotionPosteriorEncoder(motion_dim=motion_dim, z_dim=z_dim),
        prior_vae=SAMPriorVAE(traj_dim=traj_dim, z_dim=z_dim),
        latent_predictor=LatentMotionPredictor(z_dim=z_dim, motion_dim=motion_dim),
        diffusion_head=X0DiffusionHead(motion_dim=272, cond_dim=128, num_steps=8, hidden_size=512)
    ).to(device)

    # 2. 构造输入
    B = 2
    texts = ["a person walks", "a person jumps"]
    motion_seq = torch.randn(B, 13, motion_dim).to(device)
    x = torch.randn(B, 7, motion_dim).to(device)
    y = torch.randn(B, 7, motion_dim).to(device)
    past_traj = torch.randn(B, 6, traj_dim).to(device)

    # 3. 前向传播
    try:
        output = framework(texts, motion_seq, x, y, past_traj)
        print("✅ 集成前向传播成功!")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"   - {k}: {v.shape}")
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_training_loss(
    model: X0DiffusionHead,
    batch_size: int = 32,
    num_frames: int = 7,
    motion_dim: int = 272,
    cond_dim: int = 128,
    device: torch.device = torch.device("cpu")
) -> float:
    """
    测试训练损失计算是否正常。
    
    Args:
        model: X0DiffusionHead 实例
        batch_size: 批大小
        num_frames: 每个样本的帧数
        motion_dim: 运动数据维度
        cond_dim: 条件维度
        device: 设备
    
    Returns:
        loss_value: 标量损失值
    """
    model.eval()  # 确保在 eval 模式下测试（无 dropout 等）
    with torch.no_grad():
        clean_motion = torch.randn(batch_size, num_frames, motion_dim, device=device)
        cond = torch.randn(batch_size, num_frames, cond_dim, device=device)
        x0_pred_flat, clean_flat = model(clean_motion, cond)
        loss = F.mse_loss(x0_pred_flat, clean_flat)
    return loss.item()


def test_sampling(
    model: X0DiffusionHead,
    batch_size: int = 32,
    num_frames: int = 7,
    cond_dim: int = 128,
    eta: float = 0.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    测试采样过程是否正常。
    
    Args:
        model: X0DiffusionHead 实例
        batch_size: 批大小
        num_frames: 每个样本的帧数
        cond_dim: 条件维度
        eta: DDIM 随机性参数
        device: 设备
    
    Returns:
        generated: 生成的运动数据 [B, T, motion_dim]
    """
    model.eval()
    cond = torch.randn(batch_size, num_frames, cond_dim, device=device)
    with torch.no_grad():
        generated = model.sample(cond, eta=eta)
    return generated






if __name__ == '__main__':
    # test_distilbert_encoder()
    # test_motion_encoder()
    # test_latent_motion_predictor()
    # test_sam_prior_vae()
    test_samp_framework()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # 初始化模型
    # model = X0DiffusionHead(
    #     motion_dim=272,
    #     cond_dim=128,
    #     num_steps=8,
    #     hidden_size=512
    # ).to(device)

    # # 测试训练损失
    # loss_val = test_training_loss(model, batch_size=32, num_frames=7, device=device)
    # print(f"✅ Training loss: {loss_val:.6f}")

    # # 测试采样
    # generated = test_sampling(model, batch_size=32, num_frames=7, eta=0.0, device=device)
    # print(f"✅ Sampling output shape: {generated.shape}")
    # assert generated.shape == (32, 7, 272), "Sampling output shape mismatch!"

    # print("🎉 All tests passed!")
