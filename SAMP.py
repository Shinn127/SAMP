# -*- coding: utf-8 -*-
"""
SAMP - Semantically Aligned Motion Prediction (完整重构版)
Author: Adapted from Shinn Ma
Date: 2025-11-28

核心思想：
    z = z_base + Δz
    z_base = f_text(text)
    Δz ~ p(Δz | past_traj)      ← 先验
    z_post ~ q(z | full_motion, text) ← 后验（训练时用）

Pipeline:
    1. Posterior: q(z | x_{-6:6}, t) → z_post（训练）
    2. Prior: p(Δz | x_{-6:0}^{traj}) → Δz（推理采样）
    3. z = z_base + Δz
    4. LatentMotionPredictor(z, x_{-6:0}) → latent_motion (4帧, 64-dim)
    5. Diffusion: p(x_{1:4} | latent_motion) → final motion

理论公式：

1. 后验（训练）：
   q_\phi(z | x_{-6:6}, t) = \mathcal{N}(\mu_q, \sigma_q^2)

2. 先验（推理）：
   p_\theta(\Delta z | x_{-6:0}^{traj}) = \mathcal{N}(\mu_p, \sigma_p^2)
   z = z_base + \Delta z,  z_base = W_{text} t

3. 扩散模型（DDPM, x₀-parameterization）：
   x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
   \text{Loss} = \mathbb{E}_{x_0,t,\epsilon} \| \hat{x}_0(x_t, t) - x_0 \|^2

4. Cosine schedule (Nichol & Dhariwal, 2021):
   \bar{\alpha}_t = \frac{\cos^2\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)}{\cos^2\left( \frac{s}{1 + s} \cdot \frac{\pi}{2} \right)}
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# 工具：正弦时间嵌入（用于扩散 timestep 编码）
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


# --------------------------------------------------
# 1. 文本编码器（冻结 CLIP，带 fallback）
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
            # out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return out


# --------------------------------------------------
# 2. 后验编码器：q(z | full_motion, text) — 纯 Transformer Encoder
# --------------------------------------------------
class MotionPosteriorEncoder(nn.Module):
    r"""
    后验分布：q_\phi(z | x_{-6:6}, t) = \mathcal{N}(\mu, \sigma^2)
    
    输入序列（16 tokens）：
        [mu_token, logvar_token, text_token, motion_{-6}, ..., motion_{6}]
    
    结构：纯 Transformer Encoder，从第 0、1 位置读取 μ, logσ²。
    """
    def __init__(self,
                 motion_dim: int,
                 text_dim: int = 512,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 4,
                 z_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        seq_len = 2 + 1 + 13  # mu, logvar, text, 13 motions

        # Token 投影
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

        # 可学习的 μ / logσ² tokens
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.logvar_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 可学习位置编码（16 位置）
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # 纯 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
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

        # 构建 tokens
        motion_tokens = self.motion_proj(motion_seq)                     # [B,13,d]
        text_token = self.text_proj(text_emb).unsqueeze(1)               # [B,1,d]
        mu_tok = self.mu_token.expand(B, -1, -1)                         # [B,1,d]
        logvar_tok = self.logvar_token.expand(B, -1, -1)                 # [B,1,d]

        tokens = torch.cat([mu_tok, logvar_tok, text_token, motion_tokens], dim=1)  # [B,16,d]
        tokens = tokens + self.pos_embed.to(device)

        # Transformer 编码
        encoded = self.transformer(tokens)  # [B,16,d]

        # 提取 μ, logσ²
        mu = self.mu_head(encoded[:, 0, :])      # [B, z_dim]
        logvar = self.logvar_head(encoded[:, 1, :])  # [B, z_dim]
        return mu, logvar


# --------------------------------------------------
# 3. 先验网络：p(Δz | past_traj) — 轻量纯 Transformer Encoder
# --------------------------------------------------
class SAMPriorVAE(nn.Module):
    r"""
    先验分布：p_\theta(\Delta z | x_{-6:0}^{traj}) = \mathcal{N}(\mu_\Delta, \sigma_\Delta^2)
    
    输入序列（9 tokens）：
        [mu_token, logvar_token, traj_{-6}, ..., traj_{0}]
    """
    def __init__(self,
                 traj_dim: int = 36,
                 z_dim: int = 512,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.0):
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
# 4. Latent Motion 预测器（取代 CausalMotionDecoder）
# --------------------------------------------------
class LatentMotionPredictor(nn.Module):
    r"""
    预测未来 4 帧 latent motion（64-dim/帧）。
    
    输入序列（12 tokens）：
        [z_token, past_{-6:-1} (6), current_0 (1), future_{1:4} (4 learnable)]
    
    注意：不再使用 text_emb（语义已编码进 z）。
    """
    def __init__(self,
                 z_dim: int = 256,
                 motion_dim: int = 272,
                 latent_motion_dim: int = 64,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_motion_dim = latent_motion_dim
        seq_len = 1 + 6 + 1 + 4  # z, past6, curr1, future4

        self.z_proj = nn.Linear(z_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.future_tokens = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, latent_motion_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                z_post: torch.Tensor,
                past_motion: torch.Tensor,
                current_motion: torch.Tensor) -> torch.Tensor:
        B = z_post.shape[0]
        device = z_post.device

        z_tok = self.z_proj(z_post).unsqueeze(1)          # [B,1,d]
        past_tok = self.motion_proj(past_motion)          # [B,6,d]
        curr_tok = self.motion_proj(current_motion)       # [B,1,d]
        future_tok = self.future_tokens.expand(B, -1, -1) # [B,4,d]

        tokens = torch.cat([z_tok, past_tok, curr_tok, future_tok], dim=1)  # [B,12,d]
        tokens = tokens + self.pos_embed.to(device)
        encoded = self.transformer(tokens)

        future_latent = self.output_proj(encoded[:, -4:, :])  # [B,4,64]
        return future_latent


# --------------------------------------------------
# 5. 扩散模型：DDPM + Cosine + 8-step + x0-parameterization
# --------------------------------------------------
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
# 6. SAMP 框架整合
# --------------------------------------------------
class SAMP_Framework(nn.Module):
    r"""
    完整 SAMP 框架：
        z_base = W_text * t
        z_post ~ q(z | x_{-6:6}, t)          ← 训练
        Δz ~ p(Δz | x_{-6:0}^{traj})          ← 推理
        z = z_base + Δz
        latent = f(z, x_{-6:0})
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
        self.motion_dim_per_frame = joint_num * 4 * 3 + 2 + 6  # 假设为 272
        self.total_motion_dim = 4 * self.motion_dim_per_frame   # 1088
        self.latent_motion_dim = 64
        self.total_latent_motion_dim = 4 * self.latent_motion_dim  # 256

        # 文本编码器
        self.clip_encoder = RealCLIPTextEncoder(model_name=clip_model_name, device=self.device)
        self.text_to_z_proj = nn.Linear(512, z_dim)

        # 后验编码器
        self.posterior_encoder = MotionPosteriorEncoder(
            motion_dim=self.motion_dim_per_frame,
            d_model=256, nhead=4, num_layers=4, z_dim=z_dim
        )

        # 先验网络（Δz）
        self.motion_prior = SAMPriorVAE(
            traj_dim=36, z_dim=z_dim, d_model=128, nhead=2, num_layers=2
        )

        # Latent motion 预测器
        self.motion_decoder = LatentMotionPredictor(
            z_dim=z_dim,
            motion_dim=self.motion_dim_per_frame,
            latent_motion_dim=self.latent_motion_dim,
            d_model=256, nhead=4, num_layers=2
        )

        # 扩散头
        self.diffusion_head = X0DiffusionHead(
            motion_dim=self.total_motion_dim,
            cond_dim=self.total_latent_motion_dim,
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
                x_motion: torch.Tensor,        # [B,13,272]
                future_motion: torch.Tensor,   # [B,4,272]
                past_motion: torch.Tensor,     # [B,6,272]
                current_motion: torch.Tensor,  # [B,1,272]
                past_traj: torch.Tensor,       # [B,7,36]
                mode: str = 'train'):
        B = x_motion.shape[0]
        text_emb = self.clip_encoder.encode_texts(texts).to(self.device)
        base_z = self.text_to_z_proj(text_emb)

        if mode == 'train':
            full_motion = x_motion.view(B, 13, -1).to(self.device)
            past_flat = past_motion.view(B, 6, -1).to(self.device)
            curr_flat = current_motion.view(B, 1, -1).to(self.device)

            # 后验 z_post
            mu_post, logvar_post = self.posterior_encoder(full_motion, text_emb)
            z_post = self.reparameterize(mu_post, logvar_post)

            # 残差后验
            mu_delta_post = mu_post - base_z
            logvar_delta_post = logvar_post

            # 先验 p(Δz | traj)
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(self.device))

            # 预测 latent motion
            latent_motion = self.motion_decoder(z_post, past_flat, curr_flat)  # [B,4,64]
            latent_flat = latent_motion.view(B, -1)

            # 扩散 loss
            clean_future = future_motion.view(B, -1).to(self.device)
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
            mu_delta_prior, logvar_delta_prior = self.motion_prior(past_traj.to(self.device))
            delta_z = self.motion_prior.sample(mu_delta_prior, logvar_delta_prior)
            z_sample = base_z + delta_z

            past_flat = past_motion.view(B, 6, -1).to(self.device)
            curr_flat = current_motion.view(B, 1, -1).to(self.device)
            latent_motion = self.motion_decoder(z_sample, past_flat, curr_flat)
            latent_flat = latent_motion.view(B, -1)

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