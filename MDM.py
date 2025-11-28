import torch
import torch.nn as nn
import numpy as np
import clip  # ✅ 使用 OpenAI 官方 CLIP，而非 transformers
from einops import repeat
import math

# ==============================
# 1. Cosine Noise Schedule (Nichol & Dhariwal, 2021)
# ==============================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    使用余弦策略生成噪声调度（更平滑的早期去噪）。
    返回 beta_t 序列，shape: (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """
    从张量 a 中提取第 t 个时间步的值，并广播到 x_shape 的形状。
    用于扩散过程中的系数索引。
    """
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# ==============================
# 2. Transformer-based Denoising Network（使用 CLIP 全局文本嵌入）
# ==============================
class MotionDiffusionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,           # y 的帧数（预测目标）
        cond_frames=13,         # x 的帧数（条件输入）
        latent_dim=512,         # Transformer 隐藏维度
        ff_size=1024,           # FFN 中间层维度
        num_layers=6,           # Transformer 层数
        num_heads=8,            # 注意力头数
        dropout=0.1,            # dropout 率
        clip_model="ViT-B/32",  # OpenAI CLIP 模型名称（返回 512 维全局嵌入）
        use_x0_pred=True        # 使用 x0-parameterization（直接预测干净数据）
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.cond_frames = cond_frames
        self.latent_dim = latent_dim
        self.use_x0_pred = use_x0_pred
        self.device = None  # 将在 forward 中自动获取

        # === 1. OpenAI CLIP 文本编码器（冻结）===
        # 注意：不再使用 HuggingFace 的 CLIPTextModel，而是 OpenAI 官方实现
        self.clip_model, _ = clip.load(clip_model, device="cpu", jit=False)  # 初始加载到 CPU
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False  # 冻结所有参数
        self.text_emb_dim = self.clip_model.text_projection.shape[1]  # 512

        # === 2. 投影层 ===
        self.text_proj = nn.Linear(self.text_emb_dim, latent_dim)      # 文本嵌入投影到 latent 空间
        self.motion_proj_in = nn.Linear(motion_dim, latent_dim)        # 噪声 y 投影
        self.cond_proj = nn.Linear(motion_dim, latent_dim)             # 条件 x 投影

        # === 3. 位置编码（仅对 motion tokens）===
        # 总 token 数 = 1 (文本) + cond_frames + num_frames
        total_tokens = 1 + cond_frames + num_frames
        self.pos_emb = nn.Parameter(torch.randn(total_tokens, latent_dim))  # 可学习位置编码

        # === 4. 时间步嵌入 MLP ===
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # === 5. Transformer 编码器 ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === 6. 输出投影层 ===
        self.motion_proj_out = nn.Linear(latent_dim, motion_dim)

    def encode_text(self, text, device):
        """
        使用 OpenAI CLIP 编码文本为 L2 归一化的全局嵌入。
        输入: List[str] (长度 B)
        输出: [B, 512] 归一化向量
        """
        with torch.no_grad():
            # tokenize 并截断超长文本（CLIP 最大支持 77 tokens）
            tokens = clip.tokenize(text, truncate=True).to(device)  # [B, 77]
            # encode_text 返回 [B, 512] 全局句子嵌入（基于 [EOS] token）
            text_emb = self.clip_model.encode_text(tokens).float()  # [B, 512]
            # 归一化（与 CLIP 对比学习一致）
            # text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb

    def forward(self, y_noisy, t, x_cond, text):
        """
        前向传播：去噪网络预测 x0。
        Args:
            y_noisy: [B, 4, 272] - 加噪后的目标动作（待去噪）
            t: [B] - 扩散时间步（0 ~ timesteps-1）
            x_cond: [B, 13, 272] - 条件动作（过去13帧）
            text: List[str] - 文本描述（长度 B）
        Returns:
            pred_x0: [B, 4, 272] - 预测的干净目标动作（x0）
        """
        B, F, D = y_noisy.shape
        device = y_noisy.device

        # 自动将 CLIP 模型移到当前设备（首次调用时）
        if self.device != device:
            self.clip_model = self.clip_model.to(device)
            self.device = device

        # === 1. 文本编码：获取 [B, 512] 全局嵌入 ===
        text_emb = self.encode_text(text, device)  # [B, 512]

        # === 2. 投影到 latent 空间 ===
        text_emb = self.text_proj(text_emb)        # [B, latent_dim]
        x_emb = self.cond_proj(x_cond)             # [B, 13, latent_dim]
        y_emb = self.motion_proj_in(y_noisy)       # [B, 4, latent_dim]

        # === 3. 将文本嵌入扩展为 token 形式 [B, 1, latent_dim] ===
        text_token = text_emb.unsqueeze(1)  # 增加序列维度

        # === 4. 拼接所有 tokens: [text; x_cond; y_noisy] ===
        tokens = torch.cat([text_token, x_emb, y_emb], dim=1)  # [B, 1+13+4=18, latent_dim]

        # === 5. 添加可学习位置编码 ===
        tokens = tokens + self.pos_emb.unsqueeze(0)  # 广播到 batch

        # === 6. 时间步嵌入（广播到所有 tokens）===
        t_emb = timestep_embedding(t, self.latent_dim).to(device)  # [B, latent_dim]
        t_emb = self.time_mlp(t_emb)  # [B, latent_dim]
        t_emb = repeat(t_emb, 'b d -> b n d', n=tokens.shape[1])  # [B, 18, latent_dim]
        tokens = tokens + t_emb

        # === 7. Transformer 编码 ===
        hidden = self.transformer(tokens)  # [B, 18, latent_dim]

        # === 8. 提取 y 部分（最后 num_frames 个 tokens）===
        y_out = hidden[:, -self.num_frames:]  # [B, 4, latent_dim]

        # === 9. 投影回动作空间 ===
        pred = self.motion_proj_out(y_out)  # [B, 4, 272]

        # ✅ 使用 x0-parameterization：直接返回预测的干净动作
        return pred

# ==============================
# 3. 时间步嵌入（正弦位置编码）
# ==============================
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    生成正弦时间步嵌入，用于注入时间信息。
    Args:
        timesteps: [B] - 时间步索引
        dim: 嵌入维度
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
# 4. DDPM 模型（x0-parameterization + Cosine Schedule）
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

        # === 使用 cosine 策略生成噪声调度 ===
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):
        """
        前向扩散过程：给干净数据 x0 添加噪声。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x0, x_cond, text, t, noise=None):
        """
        计算训练损失（x0-parameterization）。
        模型直接预测 x0，损失为预测值与真实 x0 的 L2 距离。
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # 1. 生成加噪样本 y_noisy
        y_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        # 2. 模型预测 x0
        pred_x0 = self.model(y_noisy, t, x_cond, text)

        # 3. 计算损失
        if self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_x0, x0)
        elif self.loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(pred_x0, x0)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not supported.")
        return loss

    @torch.no_grad()
    def p_sample_loop(self, x_cond, text, shape, device):
        """
        采样过程（反向扩散）。
        使用 x0-parameterization 的采样公式（Nichol & Dhariwal）。
        """
        b = shape[0]
        y = torch.randn(shape, device=device)  # 初始化为纯噪声 [B, 4, 272]

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            pred_x0 = self.model(y, t, x_cond, text)  # 预测 x0

            # 提取当前时间步的扩散系数
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, y.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, y.shape)
            betas_t = extract(self.betas, t, y.shape)
            alphas_t = extract(1.0 - self.betas, t, y.shape)
            alphas_cumprod_t = extract(self.alphas_cumprod, t, y.shape)

            # 计算后验均值（基于 x0 预测）
            mean = (
                betas_t * pred_x0 / sqrt_one_minus_alphas_cumprod_t +
                torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_t / alphas_t) / sqrt_one_minus_alphas_cumprod_t * y
            )
            # 简化公式（等价于上述）：
            # mean = (pred_x0 * betas_t / sqrt_one_minus_alphas_cumprod_t
            #         + torch.sqrt(alphas_t) * y)

            if i == 0:
                y = mean
            else:
                # 添加噪声（除最后一步）
                posterior_var = betas_t
                noise = torch.randn_like(y)
                y = mean + torch.sqrt(posterior_var) * noise

        return y

    def forward(self, x0, x_cond, text, noise=None):
        """
        标准训练前向接口。
        随机采样时间步，计算去噪损失。
        """
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.p_losses(x0, x_cond, text, t, noise)