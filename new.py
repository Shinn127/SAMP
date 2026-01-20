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
            processed = [" ".join(t.split()[:77]) for t in texts]
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
            # out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return out






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
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * dim / 3 * 4)  # common ratio
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
        text_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        z_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        seq_len = 2 + 1 + 13  # mu, logvar, text, 13 motions
        assert seq_len <= max_seq_len

        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
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
        text_token = self.text_proj(text_emb).unsqueeze(1)               # [B,1,d]
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
        z_dim: int = 128,
        motion_dim: int = 272,
        latent_motion_dim: int = 64,
        d_model: int = 256,
        nhead: int = 4,
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
        curr_tok = self.motion_proj(current_motion)  # [B,1,d]

        x = torch.cat([z_tok, past_tok, curr_tok], dim=1)  # [B,9,d]
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
        self.attn = MultiQueryAttention(d_model, nhead, dropout)
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










