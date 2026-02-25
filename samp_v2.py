import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) Text Encoder
# ============================================================
class DistilBERTTextEncoder(nn.Module):
    r"""
    DistilBERT 文本编码器（冻结 backbone，仅训练线性投影）。

    理论：
        给定文本序列 t，先通过冻结语言模型得到句向量 h(t) \in R^{768}，
        再投影到潜空间：
            z_base = normalize(W h(t) + b),  z_base \in R^{d_z}

    说明：
        - backbone 参数不更新；
        - proj 层可训练；
        - 输出做 L2 归一化，便于稳定与后验/先验对齐。
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased",
        max_length: int = 64,
        out_dim: int = 512,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.out_dim = out_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.proj = nn.Linear(768, out_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**batch)
            cls = outputs.last_hidden_state[:, 0]  # [B, 768]

        return F.normalize(self.proj(cls), p=2, dim=-1)  # [B, out_dim]

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self.forward(texts)


# ============================================================
# 1) Shared building blocks: RMSNorm / RoPE / SwiGLU / MHA
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return self.weight * x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [L, D/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex tensor


def apply_rotary_emb(
    xq: torch.Tensor,  # [B, L, H, Dh]
    xk: torch.Tensor,  # [B, L, H, Dh]
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]  # [L, 1, Dh/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim * n_heads != dim:
            raise ValueError("dim must be divisible by n_heads")

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # [B, L, D]
        freqs_cis: torch.Tensor,       # [L, Dh/2]
        mask: Optional[torch.Tensor] = None,  # [L, L]
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis[:seqlen])

        q = q.transpose(1, 2)  # [B, H, L, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.out_proj(out)


# ============================================================
# 2) Posterior: q(z | x_{-6:6}, z_base)
# ============================================================
class PosteriorEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, freqs_cis)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MotionPosteriorEncoder(nn.Module):
    r"""
    后验编码器：
        q_\phi(z | x_{-6:6}, z_base) = N(\mu_post, \sigma_post^2)

    token 序列定义：
        [mu_token, logvar_token, text_token, m_{-6}, ..., m_0, ..., m_{+6}]
        总长度 = 16

    输出：
        mu_post, logvar_post  (shape: [B, z_dim])
    """

    def __init__(
        self,
        motion_dim: int = 272,
        text_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 3,
        z_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len)

        self.layers = nn.ModuleList([
            PosteriorEncoderLayer(d_model, n_heads, dropout) for _ in range(num_layers)
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
        bsz = motion_seq.shape[0]
        device = motion_seq.device

        motion_tokens = self.motion_proj(motion_seq)            # [B,13,d]
        text_token = self.text_proj(text_emb).unsqueeze(1)      # [B,1,d]
        mu_tok = self.mu_token.expand(bsz, -1, -1)              # [B,1,d]
        logvar_tok = self.logvar_token.expand(bsz, -1, -1)      # [B,1,d]
        x = torch.cat([mu_tok, logvar_tok, text_token, motion_tokens], dim=1)  # [B,16,d]

        freqs_cis = self.freqs_cis.to(device)
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)

        mu = self.mu_head(x[:, 0, :])
        logvar = self.logvar_head(x[:, 1, :])
        return mu, logvar


# ============================================================
# 3) Prior: p(Δz | x_{-6:-1}^{traj})
# ============================================================
class PriorLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, freqs_cis)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SAMPriorVAE(nn.Module):
    r"""
    先验网络：
        p_\theta(\Delta z | traj_{-6:-1}) = N(\mu_prior, \sigma_prior^2)

    token 序列：
        [mu_token, logvar_token, traj_{-6}, ..., traj_{-1}]  -> 长度 8
    """

    def __init__(
        self,
        traj_dim: int = 44,
        z_dim: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.traj_proj = nn.Linear(traj_dim, d_model)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, 8)

        self.layers = nn.ModuleList([
            PriorLayer(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)

        # 缩放先验初始幅度，避免训练初期先验不稳定
        self.prior_scale = nn.Parameter(torch.tensor(0.01))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, past_traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = past_traj.shape[0]
        device = past_traj.device

        traj_tok = self.traj_proj(past_traj)  # [B,6,d]
        mu_tok = self.mu_token.expand(bsz, -1, -1)
        logvar_tok = self.logvar_token.expand(bsz, -1, -1)
        x = torch.cat([mu_tok, logvar_tok, traj_tok], dim=1)  # [B,8,d]

        freqs_cis = self.freqs_cis.to(device)
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)

        mu = self.mu_head(x[:, 0, :]) * self.prior_scale
        logvar = self.logvar_head(x[:, 1, :]) * self.prior_scale
        return mu, logvar


# ============================================================
# 4) Latent Motion Predictor
# ============================================================
class LatentDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, freqs_cis, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class LatentMotionPredictor(nn.Module):
    r"""
    条件 latent 预测器：
        输入 [z, m_{-6}, ..., m_0]，输出 [lm_{-5}, ..., lm_{+1}] 共 7 个条件 latent。

    这里使用因果 mask，保证 token i 不看未来 token。
    """

    def __init__(
        self,
        z_dim: int = 512,
        motion_dim: int = 272,
        latent_dim: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 8,  # z + past6 + curr1
    ):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            LatentDecoderLayer(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        z: torch.Tensor,               # [B, z_dim]
        past_motion: torch.Tensor,     # [B, 6, motion_dim]
        curr_motion: torch.Tensor,     # [B, motion_dim] or [B,1,motion_dim]
    ) -> torch.Tensor:
        if curr_motion.dim() == 3:
            curr_motion = curr_motion[:, 0, :]

        bsz = z.shape[0]
        device = z.device

        z_tok = self.z_proj(z).unsqueeze(1)               # [B,1,d]
        past_tok = self.motion_proj(past_motion)          # [B,6,d]
        curr_tok = self.motion_proj(curr_motion).unsqueeze(1)  # [B,1,d]
        x = torch.cat([z_tok, past_tok, curr_tok], dim=1)      # [B,8,d]

        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis[:seqlen].to(device)

        # causal mask: upper triangle = -inf
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        x = self.norm(x)
        latent = self.out_proj(x)  # [B,8,latent_dim]

        # 去掉 z token，仅保留与 motion 对齐的 7 帧条件
        return latent[:, 1:, :]    # [B,7,latent_dim]


# ============================================================
# 5) Diffusion head (x0-parameterization)
# ============================================================
class AdaLN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x * (1 + scale) + shift


class DiffusionMLPBlock(nn.Module):
    def __init__(self, hidden_size: int):
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

        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(c).chunk(6, dim=1)
        x = x + gate1 * self.mlp1(self.adaln1(x, shift1, scale1))
        x = x + gate2 * self.mlp2(self.adaln2(x, shift2, scale2))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class MotionDiffusionMLP(nn.Module):
    r"""
    去噪网络 f_\theta(x_t, t, cond) -> \hat{x}_0
    """

    def __init__(self, motion_dim: int = 272, cond_dim: int = 128, hidden_size: int = 512, num_layers: int = 4):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([DiffusionMLPBlock(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.out = nn.Linear(hidden_size, motion_dim)

    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.motion_proj(x_t)
        c = self.t_embedder(t_emb) + self.cond_proj(cond)
        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.final_mod(c).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + scale) + shift
        return self.out(x)


class X0DiffusionHead(nn.Module):
    r"""
    x0 参数化扩散头

    前向扩散：
        x_t = sqrt(\bar{alpha}_t) x_0 + sqrt(1-\bar{alpha}_t) epsilon

    训练目标：
        L_diff = E || f_\theta(x_t, t, cond) - x_0 ||_2^2

    调度：
        使用 cosine \bar{alpha}_t（Nichol & Dhariwal）
    """

    def __init__(self, motion_dim: int = 272, cond_dim: int = 128, num_steps: int = 8, hidden_size: int = 512, s: float = 0.008):
        super().__init__()
        self.motion_dim = motion_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.s = s

        t_cont = torch.linspace(0, 1, 1001)
        alpha_bar = self._cosine_alpha_cumprod(t_cont, s)[:-1]
        self.register_buffer("alphas_cumprod", torch.clamp(alpha_bar, 1e-8, 1 - 1e-8))
        self.register_buffer("precomputed_t_emb", self._generate_timestep_embeddings(1000, hidden_size))
        self.register_buffer("sample_timesteps", torch.linspace(999, 0, num_steps).long())

        self.denoiser = MotionDiffusionMLP(
            motion_dim=motion_dim,
            cond_dim=cond_dim,
            hidden_size=hidden_size,
            num_layers=4,
        )

    def _cosine_alpha_cumprod(self, t: torch.Tensor, s: float) -> torch.Tensor:
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def _generate_timestep_embeddings(self, num_timesteps: int, dim: int) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000) / max(half - 1, 1)))
        emb = torch.arange(num_timesteps, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def forward(self, clean_motion: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
            clean_motion: [B, T, motion_dim]
            cond:        [B, T, cond_dim]
        返回：
            x0_pred_flat, clean_flat  (均为 [B*T, motion_dim])
        """
        bsz, tlen, _ = clean_motion.shape
        device = clean_motion.device

        clean_flat = clean_motion.reshape(bsz * tlen, -1)
        cond_flat = cond.reshape(bsz * tlen, -1)

        t_per_batch = torch.randint(0, 1000, (bsz,), device=device).long()
        t_flat = t_per_batch.repeat_interleave(tlen)
        t_emb = self.precomputed_t_emb[t_flat]

        noise = torch.randn_like(clean_flat)
        x_t = self.q_sample(clean_flat, t_flat, noise)
        x0_pred = self.denoiser(x_t, t_emb, cond_flat)
        return x0_pred, clean_flat

    def training_loss(self, clean_motion: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x0_pred, clean = self.forward(clean_motion, cond)
        return F.mse_loss(x0_pred, clean)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        条件采样：
            cond: [B, T, cond_dim]
            return: [B, T, motion_dim]
        """
        bsz, tlen, _ = cond.shape
        device = cond.device

        total = bsz * tlen
        cond_flat = cond.reshape(total, -1)
        x_t = torch.randn(total, self.motion_dim, device=device)
        timesteps = self.sample_timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(bsz)
            t_flat = t_batch.repeat_interleave(tlen)
            t_emb = self.precomputed_t_emb[t_flat]

            x0_pred = self.denoiser(x_t, t_emb, cond_flat)
            if t == 0:
                x_t = x0_pred
                break

            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            ab_t = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[t_prev]

            eps = (x_t - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1 - ab_t)
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma ** 2, min=0.0)) * eps
            x_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt

            if sigma > 0:
                x_t = x_prev + sigma * torch.randn_like(x_t)
            else:
                x_t = x_prev

        return x_t.view(bsz, tlen, -1)


# ============================================================
# 6) SAMP Framework (train + inference)
# ============================================================
class SAMPFramework(nn.Module):
    r"""
    整体框架：

    1) 文本基底语义
        z_base = E_text(text)

    2) 后验
        q(z | x_{-6:6}, z_base) = N(mu_post, sigma_post^2)
        z_post = mu_post + sigma_post * epsilon

    3) 先验（对齐 residual）
        p(Delta z | traj_{-6:-1}) = N(mu_prior, sigma_prior^2)
        训练约束 q_delta 与 p_delta 对齐，其中：
            mu_q_delta = mu_post - z_base

    4) latent 条件序列
        latent = G(z_post, x_{-6:0})  -> [B, 7, latent_dim]

    5) 条件扩散
        x_hat = D(y, latent)
    """

    def __init__(
        self,
        text_encoder: DistilBERTTextEncoder,
        posterior_encoder: MotionPosteriorEncoder,
        prior_vae: SAMPriorVAE,
        latent_predictor: LatentMotionPredictor,
        diffusion_head: X0DiffusionHead,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.posterior_encoder = posterior_encoder
        self.prior_vae = prior_vae
        self.latent_predictor = latent_predictor
        self.diffusion_head = diffusion_head

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_gaussian(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        """
        KL(N_q || N_p) = 1/2 * sum(log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1)
        """
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl = 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
        return kl.sum(dim=-1).mean()

    def forward_train(
        self,
        texts: List[str],
        motion_seq: torch.Tensor,    # [B,13,272]
        x_hist: torch.Tensor,        # [B,7,272] = past6 + curr1
        y_target: torch.Tensor,      # [B,7,272]
        past_traj: torch.Tensor,     # [B,6,44]
    ) -> Dict[str, torch.Tensor]:
        # 1) text base
        z_base = self.text_encoder.encode_texts(texts)  # [B, z_dim]

        # 2) posterior
        mu_post, logvar_post = self.posterior_encoder(motion_seq, z_base)
        z_post = self.reparameterize(mu_post, logvar_post)

        # 3) prior
        mu_prior, logvar_prior = self.prior_vae(past_traj)

        # 4) latent condition
        past_motion = x_hist[:, :6, :]
        curr_motion = x_hist[:, 6, :]
        latent_cond = self.latent_predictor(z_post, past_motion, curr_motion)  # [B,7,latent_dim]

        # 5) diffusion prediction
        x0_pred_flat, clean_flat = self.diffusion_head(y_target, latent_cond)

        return {
            "z_base": z_base,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "latent_cond": latent_cond,
            "x0_pred_flat": x0_pred_flat,
            "clean_flat": clean_flat,
        }

    def compute_loss(self, out: Dict[str, torch.Tensor], kl_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        # diffusion reconstruction
        diff_loss = F.mse_loss(out["x0_pred_flat"], out["clean_flat"])

        # residual KL: q_delta vs p_delta
        mu_q_delta = out["mu_post"] - out["z_base"]
        logvar_q_delta = out["logvar_post"]
        kl_loss = self.kl_gaussian(mu_q_delta, logvar_q_delta, out["mu_prior"], out["logvar_prior"])

        total = diff_loss + kl_weight * kl_loss
        return {
            "total_loss": total,
            "diff_loss": diff_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    @torch.no_grad()
    def sample_motion(
        self,
        texts: List[str],
        past_traj: torch.Tensor,      # [B,6,44]
        past_motion: torch.Tensor,    # [B,6,272]
        curr_motion: torch.Tensor,    # [B,272]
        eta: float = 0.0,
    ) -> torch.Tensor:
        # inference latent
        z_base = self.text_encoder.encode_texts(texts)
        mu_prior, logvar_prior = self.prior_vae(past_traj)
        dz = self.reparameterize(mu_prior, logvar_prior)
        z_inf = z_base + dz

        latent_cond = self.latent_predictor(z_inf, past_motion, curr_motion)  # [B,7,latent_dim]
        return self.diffusion_head.sample(latent_cond, eta=eta)  # [B,7,272]


# ============================================================
# 7) Minimal smoke test
# ============================================================
def _smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz = 2
    motion_dim = 272
    traj_dim = 44
    z_dim = 512
    latent_dim = 128

    model = SAMPFramework(
        text_encoder=DistilBERTTextEncoder(out_dim=z_dim),
        posterior_encoder=MotionPosteriorEncoder(
            motion_dim=motion_dim, text_dim=z_dim, z_dim=z_dim
        ),
        prior_vae=SAMPriorVAE(traj_dim=traj_dim, z_dim=z_dim),
        latent_predictor=LatentMotionPredictor(
            z_dim=z_dim, motion_dim=motion_dim, latent_dim=latent_dim
        ),
        diffusion_head=X0DiffusionHead(
            motion_dim=motion_dim, cond_dim=latent_dim, num_steps=8
        ),
    ).to(device)

    texts = ["a person walks", "a person jumps"]
    motion_seq = torch.randn(bsz, 13, motion_dim, device=device)
    x_hist = torch.randn(bsz, 7, motion_dim, device=device)
    y_target = torch.randn(bsz, 7, motion_dim, device=device)
    past_traj = torch.randn(bsz, 6, traj_dim, device=device)

    out = model.forward_train(texts, motion_seq, x_hist, y_target, past_traj)
    losses = model.compute_loss(out)
    print("smoke test ok:", {k: float(v) for k, v in losses.items()})


if __name__ == "__main__":
    _smoke_test()
