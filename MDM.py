import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
import math

# ==============================
# 1. Cosine Noise Schedule (Nichol & Dhariwal, 2021)
# ==============================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta_t.
    Returns betas of shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """Extract t-th value from a (for broadcasting)."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# ==============================
# 2. Transformer-based Denoising Network
# ==============================
class MotionDiffusionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,           # y 的帧数
        cond_frames=13,         # x 的帧数
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="openai/clip-vit-base-patch32",
        max_text_len=77,        # CLIP default
        use_x0_pred=True        # x0-parameterization
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.cond_frames = cond_frames
        self.latent_dim = latent_dim
        self.use_x0_pred = use_x0_pred

        # === CLIP Text Encoder (frozen) ===
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False  # 冻结 CLIP

        self.text_emb_dim = self.text_encoder.config.hidden_size  # e.g., 512

        # === Projection layers ===
        self.motion_proj_in = nn.Linear(motion_dim, latent_dim)
        self.text_proj = nn.Linear(self.text_emb_dim, latent_dim)
        self.cond_proj = nn.Linear(motion_dim, latent_dim)  # for x (13 frames)

        # === Time embedding ===
        self.time_emb_dim = latent_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # === Transformer Encoder ===
        self.sequence_pos_emb = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.cond_pos_emb = nn.Parameter(torch.randn(cond_frames, latent_dim))
        self.text_pos_emb = nn.Parameter(torch.randn(max_text_len, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Output projection ===
        self.motion_proj_out = nn.Linear(latent_dim, motion_dim)

    def encode_text(self, text):
        """Encode text with CLIP (frozen), always return [B, 77, D]."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",  # 👈 强制 padding 到 max_length
                max_length=77,  # 👈 CLIP 的标准最大长度
                truncation=True
            ).to(self.text_encoder.device)
            text_emb = self.text_encoder(**inputs).last_hidden_state  # [B, 77, D]
        return text_emb

    def forward(self, y_noisy, t, x_cond, text):
        """
        Args:
            y_noisy: [B, 4, 272] - noisy target motion (to denoise)
            t: [B] - diffusion time steps
            x_cond: [B, 13, 272] - condition motion
            text: List[str] - text captions
        Returns:
            pred: [B, 4, 272] - predicted x0 (if x0-parameterization)
        """
        B, F, D = y_noisy.shape
        device = y_noisy.device

        # 1. Encode text with CLIP (frozen)
        text_emb = self.encode_text(text)  # [B, L, 512]

        # 2. Project motion and condition
        y_emb = self.motion_proj_in(y_noisy)  # [B, 4, latent_dim]
        x_emb = self.cond_proj(x_cond)        # [B, 13, latent_dim]
        text_emb = self.text_proj(text_emb)   # [B, L, latent_dim]

        # 3. Time embedding
        t_emb = timestep_embedding(t, self.latent_dim).to(device)  # [B, latent_dim]
        t_emb = self.time_mlp(t_emb)  # [B, latent_dim]

        # 4. Add positional embeddings
        y_emb = y_emb + self.sequence_pos_emb[:F].unsqueeze(0)  # [B, 4, D]
        x_emb = x_emb + self.cond_pos_emb.unsqueeze(0)          # [B, 13, D]
        text_emb = text_emb + self.text_pos_emb.unsqueeze(0)    # [B, L, D]

        # 5. Concatenate all tokens: [text; x_cond; y_noisy]
        # Note: y_noisy is the "query" we want to denoise
        tokens = torch.cat([text_emb, x_emb, y_emb], dim=1)  # [B, L+13+4, D]

        # 6. Add time embedding to all tokens
        t_emb_expanded = repeat(t_emb, 'b d -> b n d', n=tokens.shape[1])
        tokens = tokens + t_emb_expanded

        # 7. Pass through transformer
        hidden = self.transformer(tokens)  # [B, L+13+4, D]

        # 8. Extract y part (last 4 tokens)
        y_out = hidden[:, -F:]  # [B, 4, D]

        # 9. Project back to motion space
        pred = self.motion_proj_out(y_out)  # [B, 4, 272]

        if self.use_x0_pred:
            return pred  # directly predict x0
        else:
            # could return noise if needed (but we use x0)
            raise NotImplementedError("Only x0-parameterization is implemented.")

# ==============================
# 3. Helper: Timestep embedding (sinusoidal)
# ==============================
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: [B]
        dim: embedding dimension
    Returns:
        embeddings: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# ==============================
# 4. DDPM Model (training & sampling logic)
# ==============================
class MotionDDPM(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        loss_type='l2'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Cosine noise schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x0, x_cond, text, t, noise=None):
        """
        Training loss (x0-parameterization).
        Predicts x0 directly, so loss = || model(y_noisy, t, x_cond, text) - x0 ||^2
        """
        if noise is None:
            noise = torch.randn_like(x0)

        y_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        pred_x0 = self.model(y_noisy, t, x_cond, text)

        if self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_x0, x0)
        elif self.loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(pred_x0, x0)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample_loop(self, x_cond, text, shape, device):
        """Sampling using x0-parameterization (DDPM reverse process)."""
        b = shape[0]
        y = torch.randn(shape, device=device)  # [B, 4, 272]

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            pred_x0 = self.model(y, t, x_cond, text)

            # Compute posterior mean (Eq. 11 in DDPM paper, adapted for x0-param)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, y.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, y.shape)

            # x0 estimate
            x0_recon = pred_x0

            # Compute mean of q(x_{t-1} | x_t, x0)
            mean = (
                x0_recon * extract(self.betas, t, y.shape) / sqrt_one_minus_alphas_cumprod_t
                + torch.sqrt(extract(self.alphas, t, y.shape)) * y
            )
            mean = mean / extract(1.0 - self.alphas_cumprod, t, y.shape).sqrt()

            if i == 0:
                y = mean
            else:
                posterior_var = extract(self.betas, t, y.shape)
                noise = torch.randn_like(y)
                y = mean + torch.sqrt(posterior_var) * noise

        return y

    def forward(self, x0, x_cond, text, noise=None):
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.p_losses(x0, x_cond, text, t, noise)