import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.9999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients by timestep and reshape for broadcasting."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder for text conditioning c."""

    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(clip_model, device=device, jit=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_dim = self.model.text_projection.shape[1]

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if str(self.model.dtype).startswith("torch.float16"):
            self.model.float()
        self.model = self.model.to(device)
        tokens = clip.tokenize(texts, truncate=True).to(device)
        return self.model.encode_text(tokens).float()


class MDMOriginalDenoiser(nn.Module):
    """
    Paper-style MDM denoiser (encoder-only):
    - Build single condition-time token z_t^c = f_t(t) + f_c(c)
    - Project each noised frame x_t^i into token space
    - Feed [z_t^c, x_t^1, ..., x_t^N] to Transformer encoder
    - Discard first output token and project others to motion dim to predict x0
    """

    def __init__(
        self,
        motion_dim: int = 272,
        seq_len: int = 196,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        clip_model: str = "ViT-B/32",
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob

        self.text_encoder = FrozenCLIPTextEncoder(clip_model=clip_model, device="cpu")
        self.cond_dim = self.text_encoder.out_dim

        self.input_proj = nn.Linear(motion_dim, latent_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Positional embedding for noised motion frame tokens only.
        # `seq_len` is treated as max sequence length and sliced at runtime.
        self.pos_emb = nn.Parameter(torch.randn(seq_len, latent_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, motion_dim)

    def _get_condition(
        self,
        batch_size: int,
        device: torch.device,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        if cond_emb is None:
            if texts is None:
                cond_emb = torch.zeros(batch_size, self.cond_dim, device=device)
            else:
                cond_emb = self.text_encoder.encode(texts, device=device)
        else:
            cond_emb = cond_emb.to(device)

        if force_uncond:
            return torch.zeros_like(cond_emb)

        if self.training and self.cond_drop_prob > 0.0:
            keep_mask = (torch.rand(batch_size, 1, device=device) > self.cond_drop_prob).float()
            cond_emb = cond_emb * keep_mask

        return cond_emb

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, N, motion_dim] noised motion at step t
            t: [B] timesteps
            texts or cond_emb: condition c
        Returns:
            pred_x0: [B, N, motion_dim]
        """
        b, n, _ = x_t.shape
        if n > self.seq_len:
            raise ValueError(f"Input seq_len={n} exceeds max seq_len={self.seq_len}")

        device = x_t.device

        c = self._get_condition(
            batch_size=b,
            device=device,
            texts=texts,
            cond_emb=cond_emb,
            force_uncond=force_uncond,
        )

        # z_t^c = f_t(t) + f_c(c)
        t_emb = timestep_embedding(t, self.latent_dim)
        z_tc = self.time_proj(t_emb) + self.cond_proj(c)  # [B, latent_dim]
        z_tc = z_tc.unsqueeze(1)  # [B, 1, latent_dim]

        # Frame tokens from x_t
        x_tokens = self.input_proj(x_t) + self.pos_emb[:n].unsqueeze(0)  # [B, N, latent_dim]

        tokens = torch.cat([z_tc, x_tokens], dim=1)  # [B, 1+N, latent_dim]
        hidden = self.encoder(tokens)

        pred_x0 = self.output_proj(hidden[:, 1:, :])  # discard z_t^c output token
        return pred_x0


class MotionDDPMOriginal(nn.Module):
    """DDPM wrapper for paper-style x0 prediction with L2 loss only."""

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 50,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def p_losses(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        pred_x0 = self.denoiser(x_t=x_t, t=t, texts=texts, cond_emb=cond_emb)
        return F.mse_loss(pred_x0, x0)

    def forward(
        self,
        x0: torch.Tensor,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        return self.p_losses(x0=x0, t=t, texts=texts, cond_emb=cond_emb, noise=noise)

    def forward_generation_batch(
        self,
        batch,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compatible with TextMotionGenerationDataset dataloader batch:
        batch = (captions: List[str], motion_batch: Tensor[B, 196, motion_dim])
        """
        captions, motion_batch = batch
        if not torch.is_tensor(motion_batch):
            motion_batch = torch.as_tensor(motion_batch)
        motion_batch = motion_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        return self.forward(x0=motion_batch, texts=captions)

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_x0 = self.denoiser(x_t=x_t, t=t, texts=texts, cond_emb=cond_emb)
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: torch.Size,
        device: torch.device,
        texts: Optional[List[str]] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        b = shape[0]
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t=x_t, t=t, texts=texts, cond_emb=cond_emb)
        return x_t


__all__ = [
    "MDMOriginalDenoiser",
    "MotionDDPMOriginal",
    "cosine_beta_schedule",
    "extract",
    "timestep_embedding",
]
