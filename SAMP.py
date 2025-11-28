# -*- coding: utf-8 -*-
"""
SAMP - Semantically Aligned Motion Prediction (完整实现 - 改进版)
Author: Shinn Ma (adapted and extended)
Date: 2025-11-25

Pipeline:
    1. text + full_motion(6 past + 1 current + 6 future = 13 frames) -> z_post (posterior)
    2. z_post + text + past_motion + current_motion -> CausalMotionDecoder -> latent_motion (4 frames)
    3. noised_motion + latent_motion -> X0 diffusion -> final motion (next 1 + future 3)

This file contains:
    - RealCLIPTextEncoder (frozen CLIP; fallback implemented)
    - MotionPosteriorEncoder (3-layer Transformer Decoder posterior)
    - CausalMotionDecoder (improved: learnable query, norm after FFN, GELU, memory includes past+current)
    - SAMPriorVAE (trajectory-conditioned Δz prior)
    - X0Denoiser and X0DiffusionHead (cosine schedule)
    - SAMP_Framework class that ties everything + loss functions and example training/inference skeleton.

Each significant line has an explanatory comment and the module docstring includes the main formulas.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional CLIP import; if missing, fall back to simple encoder (for testing)
try:
    import clip  # type: ignore
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False


# ---------------------------
# Utility: Sinusoidal time embedding
# ---------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps.
    Formula (standard):
        PE_{(t,2i)} = sin(t / 10000^{2i/d})
        PE_{(t,2i+1)} = cos(t / 10000^{2i/d})
    Implementation returns a vector of dimension `dim`.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: [B] or [B,1] integer or float timesteps
        device = time.device
        half_dim = self.dim // 2
        # log-scale factor
        emb = math.log(10000) / (half_dim - 1)
        # multiplier factors
        exponents = torch.arange(half_dim, device=device) * -emb  # [half_dim]
        # shape: [B, half_dim]
        emb = torch.exp(exponents.unsqueeze(0) * time.unsqueeze(-1).float())
        # produce sin/cos pairs
        emb_sin = torch.sin(emb)
        emb_cos = torch.cos(emb)
        return torch.cat([emb_sin, emb_cos], dim=-1)  # [B, dim]


# ---------------------------
# 1) Text encoder: RealCLIPTextEncoder (frozen) with fallback
# ---------------------------
class RealCLIPTextEncoder(nn.Module):
    r"""
    Frozen CLIP text encoder. Maps text -> 512-d embedding.
    If CLIP not available, uses a simple MLP fallback (for testing).

    Formula (conceptual):
        t = CLIP_text(text)  \in R^{512},  with ||t||_2 = 1 (we normalize)
    """
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        if _HAS_CLIP:
            # Use OpenAI CLIP, load model (frozen)
            try:
                self.model, _ = clip.load(model_name, device=device, jit=False)
                self.model.eval()
                # freeze params
                for p in self.model.parameters():
                    p.requires_grad = False
                self._use_clip = True
            except Exception:
                # Safe fallback if load fails
                self._use_clip = False
        else:
            self._use_clip = False

        if not self._use_clip:
            # fallback: a small learned text encoder (not normalized)
            # NOTE: replace during real experiments with CLIP
            self.fallback_proj = nn.Sequential(
                nn.Linear(768, 512),  # assumes some token-vector aggregation elsewhere
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 512)
            )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], len=B
        returns: [B, 512] normalized feature
        """
        if self._use_clip:
            with torch.no_grad():
                # 🔥 新增：截断超长文本（CLIP 最多支持 77 tokens）
                max_len = 77
                processed_texts = []
                for t in texts:
                    # 简单按空格截断（粗略估计，实际 token 数可能略不同）
                    words = t.split()
                    if len(words) > 70:  # 保守估计
                        t = " ".join(words[:70])
                    processed_texts.append(t)

                tokens = clip.tokenize(processed_texts, truncate=True).to(self.device)  # ⬅️ 关键：truncate=True
                feats = self.model.encode_text(tokens).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats
        else:
            # Simple fallback: hash strings to deterministic vectors (for quick tests)
            # WARNING: this is NOT semantically meaningful; used only for debugging.
            B = len(texts)
            # create deterministic pseudo-random features from text length & chars
            buf = torch.zeros(B, 768, device=self.device)
            for i, s in enumerate(texts):
                # fill with ascii codes (cyclic) up to 768
                arr = torch.tensor([ord(c) for c in s[:768]], device=self.device, dtype=torch.float32)
                buf[i, :arr.numel()] = arr
            out = self.fallback_proj(buf)  # [B,512]
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return out


# ---------------------------
# 2) Motion Posterior Encoder (3-layer Transformer Decoder)
# ---------------------------
class MotionPosteriorEncoder(nn.Module):
    r"""
    Posterior: q_\phi(z | x_full, text), where x_full = [past6, current1, future6] => 13 frames.
    Architecture:
        - motion_proj: frame-wise linear projection into d_model
        - motion_encoder: TransformerEncoder over 13 frames -> motion memory
        - text query (1 token) attends to motion memory using a small TransformerDecoder (dec_layers=3)
        - output: μ, logvar for Gaussian posterior

    Formula:
        q_\phi(z|x,t) = N( μ(x,t), diag(σ^2(x,t)) )
    """
    def __init__(self,
                 motion_dim: int,
                 text_dim: int = 512,
                 d_model: int = 256,
                 nhead: int = 4,
                 enc_layers: int = 4,
                 dec_layers: int = 3,
                 z_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, d_model)   # project per-frame motion to d_model
        self.text_proj = nn.Linear(text_dim, d_model)       # project text embedding to d_model

        # Transformer encoder for motion memory (processes 13 frames)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.motion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Transformer decoder: query (text) -> attends to motion memory
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, z_dim)          # outputs posterior mean
        self.logvar_head = nn.Linear(d_model, z_dim)      # outputs posterior log-variance

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # small initialization (Xavier for linear layers)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, motion_seq: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        motion_seq: [B, 13, motion_dim]  (6 past + 1 current + 6 future)
        text_emb: [B, text_dim]
        returns: mu [B,z_dim], logvar [B,z_dim]
        """
        B, T, _ = motion_seq.shape
        # project frames into d_model
        motion_emb = self.motion_proj(motion_seq)            # [B,13,d_model]
        motion_mem = self.motion_encoder(motion_emb)         # [B,13,d_model]
        # project text -> single query token
        query = self.text_proj(text_emb).unsqueeze(1)        # [B,1,d_model]
        # decoder: query attends to motion memory
        decoded = self.transformer_decoder(tgt=query, memory=motion_mem)  # [B,1,d_model]
        fused = self.norm(decoded.squeeze(1))                # [B,d_model]
        mu = self.mu_head(fused)                             # [B,z_dim]
        logvar = self.logvar_head(fused)                     # [B,z_dim] (log σ^2)
        return mu, logvar


# ---------------------------
# 3) LightweightTrajectory Encoder + SAMPriorVAE (Δz prior conditioned on trajectory)
# ---------------------------
class LightweightTransformerEncoder(nn.Module):
    """Encoder for short root trajectory; outputs pooled representation."""
    def __init__(self, input_dim: int = 36, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # learnable positional embedding (small sequence)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # init
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B, T, _ = x.shape
        x = self.input_proj(x) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        # mean-pool over time -> global traj representation
        return x.mean(dim=1)


class SAMPriorVAE(nn.Module):
    r"""
    Models residual prior Δz conditioned on past trajectory:
        p(Δz | x_traj) = N( μ(x_traj), diag(σ^2(x_traj)) )
    We then use: z = z_base + Δz
    """
    def __init__(self, traj_dim: int = 36, z_dim: int = 256, d_model: int = 128):
        super().__init__()
        self.encoder = LightweightTransformerEncoder(input_dim=traj_dim, d_model=d_model)
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)

    def forward(self, past_traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(past_traj)            # [B, d_model]
        return self.mu_head(h), self.logvar_head(h)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL( N(mu, sigma^2) || N(0,I) ) summed over batch
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------
# 4) Causal Decoder Layer + Improved CausalMotionDecoder
# ---------------------------
class CausalDecoderLayer(nn.Module):
    r"""
    One layer:
      - Self-attn (causal mask)
      - Add & Norm
      - Cross-attn (to memory)
      - Add & Norm
      - FFN (GELU) + Add & Norm  <-- ensure norm after FFN (norm3)

    Using LayerNorm *after* residual sum (post-norm).
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Feed-forward with GELU activation (better than ReLU empirically)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (causal) -- MultiheadAttention returns (attn_output, attn_weights)
        sa_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        x = self.norm1(tgt + self.dropout(sa_out))   # Add & Norm

        # Cross-attention to memory
        ca_out = self.cross_attn(x, memory, memory)[0]
        x = self.norm2(x + self.dropout(ca_out))     # Add & Norm

        # FFN + Add & Norm (norm3 ensures stability)
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class CausalMotionDecoder(nn.Module):
    r"""
    Autoregressive decoder producing latent_motion (4 frames).

    Memory tokens (concatenated along sequence dim):
      [ text_token(1), z_token(1), past_motion(6), current_motion(1) ]  => total 9 tokens

    Target (tgt) is a learnable query sequence of length 4 (learnable queries + positional embeddings).
    Uses registered buffer for causal mask to avoid recreation each forward.
    """
    def __init__(self,
                 z_dim: int = 256,
                 text_dim: int = 512,
                 motion_dim: int = 272,
                 latent_motion_dim: int = 64,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 tgt_len: int = 4):
        super().__init__()
        self.d_model = d_model
        self.motion_dim = motion_dim
        self.tgt_len = tgt_len

        # Projections for memory tokens
        self.z_proj = nn.Linear(z_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)

        # learnable queries (one vector per predicted frame)
        self.query = nn.Parameter(torch.randn(1, tgt_len, d_model) * 0.02)
        # positional embedding for queries (helpful for autoregressive order)
        self.pos_emb = nn.Parameter(torch.zeros(1, tgt_len, d_model))

        # stack of causal decoder layers
        self.layers = nn.ModuleList([CausalDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])

        # final projection from d_model back to motion_dim per frame
        self.output_proj = nn.Linear(d_model, latent_motion_dim)

        # precompute causal mask (upper triangular with -inf) and register it as buffer for fast reuse
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()  # upper triangle (excluding diagonal)
        # MultiheadAttention in PyTorch expects float mask where masked positions are float('-inf')
        attn_mask = torch.zeros(tgt_len, tgt_len)
        attn_mask[mask] = float('-inf')
        self.register_buffer('tgt_mask', attn_mask)  # [tgt_len, tgt_len]

        # init positional embedding
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self,
                z_post: torch.Tensor,             # [B, z_dim]
                text_emb: torch.Tensor,           # [B, text_dim]
                past_motion: torch.Tensor,        # [B, 6, motion_dim]
                current_motion: torch.Tensor) -> torch.Tensor:  # [B, 1, motion_dim]
        """
        Returns: latent_motion [B, tgt_len, motion_dim]
        """

        B = z_post.shape[0]
        device = z_post.device

        # Build initial target: repeat learnable queries for batch + add pos emb
        tgt = self.query.expand(B, -1, -1).to(device)   # [B, tgt_len, d_model]
        tgt = tgt + self.pos_emb.to(device)             # add positional bias

        # Project memory tokens
        text_mem = self.text_proj(text_emb).unsqueeze(1)   # [B,1,d_model]
        z_mem = self.z_proj(z_post).unsqueeze(1)           # [B,1,d_model]
        past_mem = self.motion_proj(past_motion)           # [B,6,d_model]
        curr_mem = self.motion_proj(current_motion)        # [B,1,d_model]

        # concatenate memory sequence: shape -> [B, 1+1+6+1 = 9, d_model]
        memory = torch.cat([text_mem, z_mem, past_mem, curr_mem], dim=1)

        # feed through causal layers (using the same causal mask)
        tgt_mask = self.tgt_mask.to(device)  # [tgt_len, tgt_len]
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask)

        # project back to motion_dim per frame
        out = self.output_proj(tgt)  # [B, tgt_len, motion_dim]
        return out


# ---------------------------
# 5) X0 Denoiser and Diffusion Head (cosine schedule)
# ---------------------------
class X0Denoiser(nn.Module):
    r"""
    Predict clean motion x0 from noisy x_t, timestep t, and latent_motion condition.
    Architecture: MLP that ingests [x_t_flat, latent_motion_flat, t_emb] and outputs x0_flat.
    """
    def __init__(self, motion_dim: int, latent_motion_dim: int, time_emb_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        # time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # build MLP
        in_dim = motion_dim + latent_motion_dim + time_emb_dim  # x_t + latent_proj + time_emb
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else motion_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, latent_motion_flat: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, motion_dim]  (flat noisy motion)
        t: [B]  (timesteps)
        latent_motion_flat: [B, latent_motion_dim]
        returns: x0_pred: [B, motion_dim]
        """
        t_emb = self.time_emb(t.float())                 # [B, time_emb_dim]
        combined = torch.cat([x_t, latent_motion_flat, t_emb], dim=1)  # [B, motion_dim + latent_motion_dim + time_emb_dim]
        return self.mlp(combined)  # predicted clean flat motion


class X0DiffusionHead(nn.Module):
    r"""
    Cosine schedule diffusion with x0-parameterization (Nichol & Dhariwal, 2021).
    - Registers precomputed alphas_cumprod (1000 steps)
    - Provides training_loss() to compute ⟂ MSE between predicted x0 and clean x0
    - sample() uses deterministic 4-step DDIM-like sampling for speed (as an example)
    """
    def __init__(self, motion_dim: int, latent_motion_dim: int, num_steps: int = 4, s: float = 0.008):
        super().__init__()
        self.motion_dim = motion_dim                    # dimension of flat 4-frame motion
        self.latent_motion_dim = latent_motion_dim
        self.num_steps = num_steps
        self.s = s
        # compute cosine schedule for 1000 discrete timesteps
        timesteps_cont = torch.linspace(0, 1, 1001)
        alphas_cumprod = self._cosine_alpha_cumprod(timesteps_cont, s)  # length 1001
        alphas_cumprod = alphas_cumprod[:-1]  # keep 1000 steps
        alphas_cumprod = torch.clamp(alphas_cumprod, 1e-8, 1.0 - 1e-8)
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # [1000]

        # choose discrete timesteps for sampling (e.g., 4-step schedule)
        self.register_buffer('sample_timesteps', torch.linspace(999, 0, num_steps).long())

        # denoiser net
        self.denoiser = X0Denoiser(motion_dim, latent_motion_dim)

    def _cosine_alpha_cumprod(self, t: torch.Tensor, s: float) -> torch.Tensor:
        # Continuous function f(t) = cos^2( (t+s)/(1+s) * pi/2 ), normalized by f(0)
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0_val = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0_val

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t given x0 and t using closed form:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        x0: [B, motion_dim]
        t: [B] int in [0, 999]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].unsqueeze(1)  # [B, 1]
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def training_loss(self, clean_motion: torch.Tensor, latent_motion_flat: torch.Tensor) -> torch.Tensor:
        """
        Training objective:
            - sample t ~ Uniform([0,999])
            - compute x_t by q_sample(clean_motion, t)
            - predict x0_pred = denoiser(x_t, t, latent_motion)
            - loss = MSE(x0_pred, clean_motion)
        Inputs:
            clean_motion: [B, motion_dim]  (flat clean motion we want to regress to)
            latent_motion_flat: [B, latent_dim]  (conditioning)
        returns: MSE loss scalar
        """
        B = clean_motion.shape[0]
        device = clean_motion.device
        t = torch.randint(0, 1000, (B,), device=device).long()  # random timesteps
        noise = torch.randn_like(clean_motion)
        x_t = self.q_sample(clean_motion, t, noise)
        # denoiser returns predicted x0 (flat)
        x0_pred = self.denoiser(x_t, t, latent_motion_flat)
        return F.mse_loss(x0_pred, clean_motion)

    @torch.no_grad()
    def sample(self, latent_motion_flat: torch.Tensor) -> torch.Tensor:
        """
        Simple sampling scheme using a small number of steps (self.num_steps).
        This is NOT a fully optimized DDIM sampler, but demonstrates the integration.
        Returns: motion in shape [B, 4, 272]
        """
        B = latent_motion_flat.shape[0]
        device = latent_motion_flat.device
        # start from pure noise in flat motion space
        x_t = torch.randn(B, self.motion_dim, device=device)
        timesteps = self.sample_timesteps.to(device)   # e.g., [999, 666, 333, 0] if num_steps=4
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B).to(device)
            # predict x0
            x0_pred = self.denoiser(x_t, t_batch, latent_motion_flat)  # [B, motion_dim(4*272)]
            # compute alpha_t and alpha_prev
            a_t = self.alphas_cumprod[t_batch].unsqueeze(1)            # [B,1]
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1].repeat(B).to(device)
            else:
                t_prev = torch.zeros_like(t_batch)
            a_prev = self.alphas_cumprod[t_prev].unsqueeze(1)
            # derive epsilon: eps = (x_t - sqrt(a_t)*x0) / sqrt(1 - a_t)
            eps = (x_t - torch.sqrt(a_t) * x0_pred) / (torch.sqrt(1 - a_t) + 1e-8)
            # update x_{t_prev} = sqrt(a_prev) * x0 + sqrt(1 - a_prev) * eps
            x_t = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps
        # reshape final flat motion to [B, 4, per_frame_motion_dim]
        out = x_t.view(B, 4, int(self.motion_dim / 4))
        return out


# ---------------------------
# 6) 修改后的 SAMP Framework
# ---------------------------
class SAMP_Framework(nn.Module):
    r"""
    Full SAMP framework with explicit residual structure:
        z_post = z_base + Δz_post   (enforced via loss, not architecture)
        Constrain: Δz_post ≈ Δz_prior ~ p(Δz | past_traj)

    Train forward usage:
        outputs: diffusion_loss, kl_delta, (optional l2_mean), etc.

    Inference usage:
        outputs: sampled motion [B, 4, 272]
    """
    def __init__(self,
                 joint_num: int = 22,
                 clip_model_name: str = "ViT-B/32",
                 z_dim: int = 256,
                 device: Optional[str] = None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_num = joint_num
        self.motion_dim_per_frame = joint_num * 4 * 3 + 2 + 6
        self.total_motion_dim = 4 * self.motion_dim_per_frame
        self.latent_motion_dim = 64
        self.total_latent_motion_dim = 4 * self.latent_motion_dim
        self.z_dim = z_dim

        # text encoder (frozen CLIP or fallback)
        self.clip_encoder = RealCLIPTextEncoder(model_name=clip_model_name, device=self.device)
        self.text_to_z_proj = nn.Linear(512, z_dim)  # base z from text (z_base)

        # posterior encoder: maps full_motion (13 frames) + text -> z_post (mu, logvar)
        self.posterior_encoder = MotionPosteriorEncoder(
            motion_dim=self.motion_dim_per_frame,
            text_dim=512,
            d_model=256,
            nhead=4,
            enc_layers=4,
            dec_layers=3,
            z_dim=z_dim
        )

        # prior for Δz conditioned on trajectory
        self.motion_prior = SAMPriorVAE(traj_dim=36, z_dim=z_dim, d_model=128)

        # causal motion decoder
        self.motion_decoder = CausalMotionDecoder(
            z_dim=z_dim,
            text_dim=512,
            motion_dim=self.motion_dim_per_frame,
            latent_motion_dim=self.latent_motion_dim,
            d_model=256,
            nhead=4,
            num_layers=2,
            tgt_len=4
        )

        # diffusion head
        self.diffusion_head = X0DiffusionHead(
            motion_dim=self.total_motion_dim,
            latent_motion_dim=self.total_latent_motion_dim,
            num_steps=4
        )

        self.to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_gaussian_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        """KL(N(mu_q, var_q) || N(mu_p, var_p))"""
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl = 0.5 * torch.sum(
            logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1,
            dim=1
        )
        return kl.mean()

    def forward(self,
                texts: List[str],
                x_motion: torch.Tensor,        # [B, 13, 272]
                future_motion: torch.Tensor,   # [B, 4, 272]
                past_motion: torch.Tensor,     # [B, 6, 272]
                current_motion: torch.Tensor,  # [B, 1, 272]
                past_traj: torch.Tensor,
                mode: str = 'train'):
        B = x_motion.shape[0]
        device = self.device

        text_emb = self.clip_encoder.encode_texts(texts).to(device)  # [B, 512]
        base_z = self.text_to_z_proj(text_emb)  # [B, z_dim]

        if mode == 'train':
            full_motion_flat = x_motion.view(B, x_motion.shape[1], -1).to(device)
            past_motion_flat = past_motion.view(B, past_motion.shape[1], -1).to(device)
            current_motion_flat = current_motion.view(B, current_motion.shape[1], -1).to(device)

            # Posterior: z_post ~ N(mu_post, var_post)
            mu_post, logvar_post = self.posterior_encoder(full_motion_flat, text_emb)
            z_post = self.reparameterize(mu_post, logvar_post)

            # Define residual: Δz_post = z_post - z_base
            # Distribution: N(mu_post - base_z, var_post)
            mu_delta_post = mu_post - base_z
            logvar_delta_post = logvar_post

            # Prior: p(Δz | traj)
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(device))

            # Decode latent motion using z_post
            latent_motion = self.motion_decoder(
                z_post=z_post,
                text_emb=text_emb,
                past_motion=past_motion_flat,
                current_motion=current_motion_flat
            )  # [B, 4, latent_motion_dim]
            latent_motion_flat = latent_motion.view(B, -1)

            clean_future4 = future_motion.view(B, -1).to(device)
            diffusion_loss = self.diffusion_head.training_loss(clean_future4, latent_motion_flat)

            return {
                'diffusion_loss': diffusion_loss,
                'mu_delta_post': mu_delta_post,
                'logvar_delta_post': logvar_delta_post,
                'mu_delta_prior': mu_delta_prior,
                'logvar_delta_prior': logvar_delta_prior,
                'base_z': base_z,
                'latent_motion': latent_motion
            }

        elif mode == 'inference':
            # Sample Δz from prior
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(device))
            delta_z = self.motion_prior.sample(mu_delta_prior, logvar_delta_prior)
            z_sample = base_z + delta_z  # z = z_base + Δz

            past_motion_flat = past_motion.view(B, past_motion.shape[1], -1).to(device)
            current_motion_flat = current_motion.view(B, current_motion.shape[1], -1).to(device)
            latent_motion = self.motion_decoder(
                z_post=z_sample,
                text_emb=text_emb,
                past_motion=past_motion_flat,
                current_motion=current_motion_flat
            )
            latent_motion_flat = latent_motion.view(B, -1)
            sampled_motion = self.diffusion_head.sample(latent_motion_flat)
            return sampled_motion

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
        """
        Total loss = diffusion_loss + λ_kl * KL(q(Δz) || p(Δz)) + λ_l2 * ||μ_q - μ_p||²
        """
        # Main: KL between posterior residual and trajectory-conditioned prior
        kl_delta = self._kl_gaussian_gaussian(
            mu_delta_post, logvar_delta_post,
            mu_delta_prior, logvar_delta_prior
        )

        # Auxiliary: L2 on means (optional but recommended with small weight)
        l2_mean = F.mse_loss(mu_delta_post, mu_delta_prior)

        total_loss = diffusion_loss + lambda_kl * kl_delta + lambda_l2 * l2_mean

        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss.item(),
            'kl_delta': kl_delta.item(),
            'l2_mean': l2_mean.item(),
            'lambda_kl': lambda_kl,
            'lambda_l2': lambda_l2
        }

    def add_geometry_constraints(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def bind_constraint_hook(self, fn):
        self.add_geometry_constraints = fn



class SAMP_Framework_wo_diff(nn.Module):
    r"""
    SAMP without diffusion. Uses a 3-layer Transformer Encoder to directly predict future motion.

    Input memory tokens (flattened per-frame):
        [ text_emb (1), z (1), past_motion (6), current_motion (1) ] → total 9 tokens

    Target: future_motion (4 frames)

    Architecture:
        - Project all inputs to d_model
        - Add learnable positional encoding (length=9)
        - 3-layer Transformer Encoder
        - Global pooling or query-based readout → project to 4 * motion_dim_per_frame
        - Reshape to [B, 4, motion_dim_per_frame]

    Training: uses z_post from posterior.
    Inference: uses z = z_base + Δz ~ prior.
    """
    def __init__(self,
                 joint_num: int = 22,
                 clip_model_name: str = "ViT-B/32",
                 z_dim: int = 256,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 device: Optional[str] = None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_num = joint_num
        self.motion_dim_per_frame = joint_num * 4 * 3 + 2 + 6  # e.g., 272
        self.future_frames = 4
        self.z_dim = z_dim
        self.d_model = d_model

        # ===== Shared components (same as original SAMP) =====
        self.clip_encoder = RealCLIPTextEncoder(model_name=clip_model_name, device=self.device)
        self.text_to_z_proj = nn.Linear(512, z_dim)  # base z

        self.posterior_encoder = MotionPosteriorEncoder(
            motion_dim=self.motion_dim_per_frame,
            text_dim=512,
            d_model=256,
            nhead=4,
            enc_layers=4,
            dec_layers=3,
            z_dim=z_dim
        )

        self.motion_prior = SAMPriorVAE(traj_dim=36, z_dim=z_dim, d_model=128)

        # ===== New: Direct motion predictor (Transformer Encoder) =====
        # Projections
        self.text_proj = nn.Linear(512, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.motion_proj = nn.Linear(self.motion_dim_per_frame, d_model)

        # Learnable positional encoding for fixed-length input (9 tokens)
        self.max_tokens = 1 + 1 + 6 + 1  # text + z + past6 + current1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predict 4 future frames
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, self.future_frames * self.motion_dim_per_frame)
        )

        self.to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _build_memory_tokens(self,
                            z: torch.Tensor,
                            text_emb: torch.Tensor,
                            past_motion: torch.Tensor,
                            current_motion: torch.Tensor) -> torch.Tensor:
        """
        Build input token sequence: [text, z, past_6, current_1] → [B, 9, d_model]
        All inputs assumed already on correct device and shape.
        """
        B = z.shape[0]
        text_tok = self.text_proj(text_emb).unsqueeze(1)          # [B,1,d_model]
        z_tok = self.z_proj(z).unsqueeze(1)                       # [B,1,d_model]
        past_tok = self.motion_proj(past_motion)                  # [B,6,d_model]
        curr_tok = self.motion_proj(current_motion)               # [B,1,d_model]
        tokens = torch.cat([text_tok, z_tok, past_tok, curr_tok], dim=1)  # [B,9,d_model]
        tokens = tokens + self.pos_embed  # add positional encoding
        return tokens

    def forward(self,
                texts: List[str],
                x_motion: torch.Tensor,        # [B,13,272] – full motion (train only)
                future_motion: torch.Tensor,   # [B,4,272] – ground truth future
                past_motion: torch.Tensor,     # [B,6,272]
                current_motion: torch.Tensor,  # [B,1,272]
                past_traj: torch.Tensor,       # [B, T, 36]
                mode: str = 'train'):
        B = x_motion.shape[0]
        device = self.device

        # Encode text
        text_emb = self.clip_encoder.encode_texts(texts).to(device)  # [B,512]
        base_z = self.text_to_z_proj(text_emb)  # [B,z_dim]

        # Flatten motion per frame
        past_flat = past_motion.view(B, 6, -1).to(device)
        current_flat = current_motion.view(B, 1, -1).to(device)
        future_flat_gt = future_motion.view(B, 4, -1).to(device) if future_motion is not None else None

        if mode == 'train':
            full_flat = x_motion.view(B, 13, -1).to(device)
            # Posterior encoder
            mu_post, logvar_post = self.posterior_encoder(full_flat, text_emb)
            z_post = self.reparameterize(mu_post, logvar_post)

            # Build tokens and predict
            tokens = self._build_memory_tokens(z_post, text_emb, past_flat, current_flat)
            encoded = self.transformer(tokens)  # [B,9,d_model]

            # Global mean pooling over tokens → [B,d_model]
            pooled = encoded.mean(dim=1)
            pred_flat = self.output_proj(pooled)  # [B, 4 * motion_dim]
            pred_motion = pred_flat.view(B, 4, -1)  # [B,4,272]

            # Prior for Δz (for KL loss)
            mu_delta, logvar_delta = self.motion_prior(past_traj.to(device))

            return {
                'pred_motion': pred_motion,
                'future_gt': future_flat_gt,
                'mu_post': mu_post,
                'logvar_post': logvar_post,
                'base_z': base_z,
                'mu_delta': mu_delta,
                'logvar_delta': logvar_delta
            }

        elif mode == 'inference':
            # Sample Δz from prior
            mu_delta, logvar_delta = self.motion_prior(past_traj.to(device))
            delta_z = self.motion_prior.sample(mu_delta, logvar_delta)
            z_sample = base_z + delta_z

            tokens = self._build_memory_tokens(z_sample, text_emb, past_flat, current_flat)
            encoded = self.transformer(tokens)
            pooled = encoded.mean(dim=1)
            pred_flat = self.output_proj(pooled)
            pred_motion = pred_flat.view(B, 4, -1)  # [B,4,272]
            return pred_motion

        else:
            raise ValueError("mode must be 'train' or 'inference'")

    def loss_function(self,
                      pred_motion: torch.Tensor,
                      future_gt: torch.Tensor,
                      mu_post: torch.Tensor,
                      logvar_post: torch.Tensor,
                      base_z: torch.Tensor,
                      mu_delta: torch.Tensor,
                      logvar_delta: torch.Tensor,
                      lambda_kl_post: float = 1.0,
                      lambda_kl_prior: float = 0.01) -> dict:
        """
        Total loss = MSE(pred, gt) + λ1 * KL(q(z|x,t) || N(z_base, I)) + λ2 * KL(p(Δz|traj) || N(0,I))
        """
        motion_loss = torch.nn.functional.mse_loss(pred_motion, future_gt)

        # KL: posterior vs N(z_base, I)
        kl_post = -0.5 * torch.sum(1 + logvar_post - (mu_post - base_z).pow(2) - logvar_post.exp())
        kl_post = kl_post / mu_post.size(0)

        # KL: prior Δz vs N(0, I)
        kl_prior = self.motion_prior.kl_divergence(mu_delta, logvar_delta)
        kl_prior = kl_prior / mu_delta.size(0)

        total_loss = motion_loss + lambda_kl_post * kl_post + lambda_kl_prior * kl_prior

        return {
            'total_loss': total_loss,
            'motion_loss': motion_loss.item(),
            'kl_post': kl_post.item(),
            'kl_prior': kl_prior.item(),
            'lambda_kl_post': lambda_kl_post,
            'lambda_kl_prior': lambda_kl_prior
        }

    def add_geometry_constraints(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def bind_constraint_hook(self, fn):
        self.add_geometry_constraints = fn





# ---------------------------
# 6) SAMP Framework itself (ties everything together)
# ---------------------------
class SAMP_Framework_old(nn.Module):
    r"""
    Full SAMP framework that implements:
        - posterior q(z|full_motion, text)
        - prior p(Δz|traj)
        - causal decoder -> latent_motion
        - diffusion conditioned on latent_motion

    Train forward usage:
        outputs: diffusion_loss, mu_post, logvar_post, base_z, mu_delta, logvar_delta

    Inference usage:
        outputs: sampled motion [B, 4, 272]
    """
    def __init__(self,
                 joint_num: int = 22,
                 clip_model_name: str = "ViT-B/32",
                 z_dim: int = 256,
                 device: Optional[str] = None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_num = joint_num
        self.motion_dim_per_frame = joint_num * 4 * 3 + 2 + 6
        self.total_motion_dim = 4 * self.motion_dim_per_frame
        self.latent_motion_dim = 64
        self.total_latent_motion_dim = 4 * self.latent_motion_dim
        self.z_dim = z_dim

        # text encoder (frozen CLIP or fallback)
        self.clip_encoder = RealCLIPTextEncoder(model_name=clip_model_name, device=self.device)
        self.text_to_z_proj = nn.Linear(512, z_dim)  # base z from text (z_base)

        # posterior encoder: maps full_motion (13 frames) + text -> z_post (mu, logvar)
        self.posterior_encoder = MotionPosteriorEncoder(
            motion_dim=self.motion_dim_per_frame,
            text_dim=512,
            d_model=256,
            nhead=4,
            enc_layers=4,
            dec_layers=3,
            z_dim=z_dim
        )

        # prior for Δz conditioned on trajectory
        self.motion_prior = SAMPriorVAE(traj_dim=36, z_dim=z_dim, d_model=128)

        # causal motion decoder
        self.motion_decoder = CausalMotionDecoder(
            z_dim=z_dim,
            text_dim=512,
            motion_dim=self.motion_dim_per_frame,
            latent_motion_dim=self.latent_motion_dim,
            d_model=256,
            nhead=4,
            num_layers=2,
            tgt_len=4
        )

        # diffusion head (x0 parameterization) conditioned on latent_motion
        self.diffusion_head = X0DiffusionHead(
            motion_dim=self.total_motion_dim,
            latent_motion_dim=self.total_latent_motion_dim,
            num_steps=4
        )

        # bind helper: let diffusion use framework geometry constraints (optionally overridden)
        self.diffusion_head.add_geometry_constraints = self.add_geometry_constraints

        # to device
        self.to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from posterior:
            z = mu + eps * sigma,    sigma = exp(0.5 * logvar)
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                texts: List[str],
                x_motion: torch.Tensor,  # [B, 13, 272] ← 这是 x
                future_motion: torch.Tensor,  # [B, 4, 272] ← 这是 y，真实目标
                past_motion: torch.Tensor,  # [B, 6, 272]
                current_motion: torch.Tensor,  # [B, 1, 272]
                past_traj: torch.Tensor,
                mode: str = 'train'):
        """
        mode == 'train':
            * compute posterior z_post from full_motion & text
            * compute latent_motion from (z_post, text, past_motion, current_motion)
            * compute diffusion training_loss given clean future (we will pass future separately)
        mode == 'inference':
            * sample delta_z from prior, build z = z_base + delta_z
            * decode latent_motion and sample via diffusion.sample()
        """
        B = x_motion.shape[0]
        device = self.device

        # 1) encode text -> [B,512] normalized
        text_emb = self.clip_encoder.encode_texts(texts).to(device)  # [B, 512]

        # 2) base z from text embedding (deterministic)
        base_z = self.text_to_z_proj(text_emb)  # [B, z_dim]

        if mode == 'train':
            # Expect full_motion: [B, 13, 272]
            # Convert frames to per-frame flattened vectors [B, 13, motion_dim_per_frame]
            full_motion_flat = x_motion.view(B, x_motion.shape[1], -1).to(device)  # [B,13,dim]
            past_motion_flat = past_motion.view(B, past_motion.shape[1], -1).to(device)  # [B,6,dim]
            current_motion_flat = current_motion.view(B, current_motion.shape[1], -1).to(device)  # [B,1,dim]

            # posterior q(z|full_motion, text)
            mu_post, logvar_post = self.posterior_encoder(full_motion_flat, text_emb)  # [B,z_dim] each
            z_post = self.reparameterize(mu_post, logvar_post)  # sample z_post

            # prior for Δz: from past trajectory
            mu_delta, logvar_delta = self.motion_prior(past_traj.to(device))

            # decode latent_motion from (z_post, text_emb, past_motion, current_motion)
            latent_motion = self.motion_decoder(
                z_post=z_post,
                text_emb=text_emb,
                past_motion=past_motion_flat,
                current_motion=current_motion_flat
            )  # [B, 4, latent_motion_dim]

            # flatten latent motion as diffusion condition
            latent_motion_flat = latent_motion.view(B, -1)  # [B, total_latent_motion_dim]

            # diffusion loss expects clean target motion (which frame-range?)
            # We defined pipeline to predict the future 4 frames (or the appropriate target)
            # Here: choose clean_motion_flat as the *target* we want to reconstruct by diffusion.
            # In your original description the diffusion final motion is "final motion" (e.g., future 4 frames)
            # For training we'll assume clean target is full_motion[:, :4] (past->target mapping depends on dataset design).
            clean_future4 = future_motion.view(B, -1).to(device)  # [B, total_motion_dim]

            diffusion_loss = self.diffusion_head.training_loss(clean_future4, latent_motion_flat)

            return {
                'diffusion_loss': diffusion_loss,
                'mu_post': mu_post,
                'logvar_post': logvar_post,
                'base_z': base_z,
                'mu_delta': mu_delta,
                'logvar_delta': logvar_delta,
                'latent_motion': latent_motion
            }

        elif mode == 'inference':
            # sample delta_z from prior
            mu_delta, logvar_delta = self.motion_prior(past_traj.to(device))
            delta_z = self.motion_prior.sample(mu_delta, logvar_delta)  # [B,z_dim]
            z_sample = base_z + delta_z  # combine base text z with trajectory residual

            # decode latent motion
            # prepare past & current flattened (assuming already provided in correct shape)
            past_motion_flat = past_motion.view(B, past_motion.shape[1], -1).to(device)
            current_motion_flat = current_motion.view(B, current_motion.shape[1], -1).to(device)
            latent_motion = self.motion_decoder(
                z_post=z_sample,
                text_emb=text_emb,
                past_motion=past_motion_flat,
                current_motion=current_motion_flat
            )  # [B, 4, latent_motion_dim]
            latent_motion_flat = latent_motion.view(B, -1)

            # sample final motion via diffusion
            sampled_motion = self.diffusion_head.sample(latent_motion_flat)  # [B,4,272]
            return sampled_motion

        else:
            raise ValueError("mode must be 'train' or 'inference'")

    def loss_function(self,
                      diffusion_loss: torch.Tensor,
                      mu_post: torch.Tensor,
                      logvar_post: torch.Tensor,
                      base_z: torch.Tensor,
                      mu_delta: torch.Tensor,
                      logvar_delta: torch.Tensor,
                      lambda_kl_post: float = 1.0,
                      lambda_kl_prior: float = 0.01) -> dict:
        """
        Total loss:
           L = L_diff + λ1 * KL(q(z|x,t) || N(z_base, I)) + λ2 * KL(p(Δz|traj) || N(0,I))
        KL(q || N(z_base, I)) = -0.5 * sum( 1 + logvar - (mu - base_z)^2 - exp(logvar) )
        KL_prior = KL(p(Δz|traj) || N(0,I)) computed in SAMPriorVAE.kl_divergence
        """
        # kl for posterior w.r.t N(base_z, I)
        kl_post = -0.5 * torch.sum(1 + logvar_post - (mu_post - base_z).pow(2) - logvar_post.exp())
        kl_post = kl_post / mu_post.size(0)   # mean over batch

        # kl for prior (Δz) w.r.t N(0,I)
        kl_prior = self.motion_prior.kl_divergence(mu_delta, logvar_delta)
        kl_prior = kl_prior / mu_delta.size(0)

        total_loss = diffusion_loss + lambda_kl_post * kl_post + lambda_kl_prior * kl_prior
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss.item(),
            'kl_post': kl_post.item(),
            'kl_prior': kl_prior.item()
        }

    def add_geometry_constraints(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Hook for geometry constraints. By default returns motion unchanged.
        You can override this (e.g., enforce bone lengths, remove root drift, etc.)
        Expected motion shape: [B, 6, joint_num, 3]
        """
        return motion

    def bind_constraint_hook(self, fn):
        """
        Allow external binding of geometry constraint function, e.g.:
            framework.bind_constraint_hook(my_constraint_fn)
        where my_constraint_fn(motion) -> constrained_motion
        """
        self.add_geometry_constraints = fn
