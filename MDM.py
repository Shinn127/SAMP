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
    

class MotionDiffusionTransformerDecoder(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,
        cond_frames=13,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="ViT-B/32",
        use_x0_pred=True
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.cond_frames = cond_frames
        self.latent_dim = latent_dim
        self.use_x0_pred = use_x0_pred
        self.device = None

        # === 1. CLIP Text Encoder (frozen) ===
        self.clip_model, _ = clip.load(clip_model, device="cpu", jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.text_emb_dim = self.clip_model.text_projection.shape[1]  # 512

        # === 2. Projection layers ===
        self.text_proj = nn.Linear(self.text_emb_dim, latent_dim)      # text -> latent
        self.motion_proj_in = nn.Linear(motion_dim, latent_dim)        # noisy motion -> latent
        self.cond_proj = nn.Linear(motion_dim, latent_dim)             # condition x -> latent
        self.time_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )  # time embedding -> time token

        # === 3. Positional embeddings ===
        # Tokens: [time_token (1); x_cond (cond_frames); y_noisy (num_frames)]
        self.total_query_tokens = 1 + cond_frames + num_frames
        self.pos_emb = nn.Parameter(torch.randn(self.total_query_tokens, latent_dim))

        # === 4. Transformer Decoder ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # === 5. Output projection ===
        self.motion_proj_out = nn.Linear(latent_dim, motion_dim)

    def encode_text(self, text, device):
        with torch.no_grad():
            tokens = clip.tokenize(text, truncate=True).to(device)
            text_emb = self.clip_model.encode_text(tokens).float()
        return text_emb

    def forward(self, y_noisy, t, x_cond, text):
        """
        Args:
            y_noisy: [B, 4, 272]
            t: [B]
            x_cond: [B, 13, 272]
            text: List[str]
        Returns:
            pred_x0: [B, 4, 272]
        """
        B = y_noisy.shape[0]
        device = y_noisy.device

        if self.device != device:
            self.clip_model = self.clip_model.to(device)
            self.device = device

        # === 1. Encode text → [B, latent_dim] → memory ===
        text_emb = self.encode_text(text, device)  # [B, 512]
        text_emb = self.text_proj(text_emb)        # [B, latent_dim]
        memory = text_emb.unsqueeze(1)             # [B, 1, latent_dim] ← memory for cross-attn

        # === 2. Project motion inputs ===
        x_emb = self.cond_proj(x_cond)             # [B, 13, latent_dim]
        y_emb = self.motion_proj_in(y_noisy)       # [B, 4, latent_dim]

        # === 3. Time embedding as a token ===
        t_emb = timestep_embedding(t, self.latent_dim).to(device)  # [B, latent_dim]
        time_token = self.time_proj(t_emb).unsqueeze(1)            # [B, 1, latent_dim]

        # === 4. Concatenate query tokens: [time; x_cond; y_noisy] ===
        query_tokens = torch.cat([time_token, x_emb, y_emb], dim=1)  # [B, 1+13+4=18, latent_dim]

        # === 5. Add positional encoding to query tokens ===
        query_tokens = query_tokens + self.pos_emb.unsqueeze(0)  # [B, 18, latent_dim]

        # === 6. Transformer Decoder ===
        # query: [B, 18, latent_dim]
        # memory: [B, 1, latent_dim]
        hidden = self.transformer_decoder(query_tokens, memory)    # [B, 18, latent_dim]

        # === 7. Extract y part (last num_frames tokens) ===
        y_out = hidden[:, -self.num_frames:]  # [B, 4, latent_dim]

        # === 8. Project to motion space ===
        pred = self.motion_proj_out(y_out)    # [B, 4, 272]
        return pred
    

# ==============================
# Diffusion Head: 接收 y_noisy + time + context
# ==============================
class DiffusionHead(nn.Module):
    def __init__(self, motion_dim, num_frames, context_dim, hidden_dim=1024, num_layers=4):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        input_dim = motion_dim * num_frames + context_dim + context_dim  # y_flat + context + t_emb
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU()
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, motion_dim * num_frames))
        self.net = nn.Sequential(*layers)

    def forward(self, y_noisy, context, t_emb):
        # y_noisy: [B, F, D] → [B, F*D]
        # context: [B, C]
        # t_emb:   [B, C] （与 context 同维）
        B = y_noisy.shape[0]
        y_flat = y_noisy.view(B, -1)  # [B, F*D]
        x = torch.cat([y_flat, context, t_emb], dim=-1)  # [B, F*D + C + C]
        out = self.net(x)  # [B, F*D]
        return out.view(B, self.num_frames, self.motion_dim)
    

# ==============================
# 条件编码器：仅处理 text + x_cond
# ==============================
class MotionConditionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        cond_frames=13,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="ViT-B/32"
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.cond_frames = cond_frames
        self.latent_dim = latent_dim
        self.device = None

        # CLIP text encoder (frozen)
        self.clip_model, _ = clip.load(clip_model, device="cpu", jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.text_emb_dim = self.clip_model.text_projection.shape[1]

        # Projections
        self.text_proj = nn.Linear(self.text_emb_dim, latent_dim)
        self.cond_proj = nn.Linear(motion_dim, latent_dim)

        # Positional embedding for cond tokens + text token
        total_tokens = 1 + cond_frames
        self.pos_emb = nn.Parameter(torch.randn(total_tokens, latent_dim))

        # Time is NOT in this encoder (handled in diffusion head)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional: global pooling or use [CLS]-like token
        # We'll use average pooling over all tokens for context
        # Or you can use the first (text) token as context

    def encode_text(self, text, device):
        with torch.no_grad():
            tokens = clip.tokenize(text, truncate=True).to(device)
            text_emb = self.clip_model.encode_text(tokens).float()
        return text_emb

    def forward(self, x_cond, text):
        """
        Encode condition (x_cond + text) into a global context vector.
        Returns: context [B, latent_dim]
        """
        B = x_cond.shape[0]
        device = x_cond.device

        if self.device != device:
            self.clip_model = self.clip_model.to(device)
            self.device = device

        # Text encoding
        text_emb = self.encode_text(text, device)  # [B, 512]
        text_emb = self.text_proj(text_emb)        # [B, latent_dim]
        text_token = text_emb.unsqueeze(1)         # [B, 1, latent_dim]

        # Condition encoding
        x_emb = self.cond_proj(x_cond)             # [B, 13, latent_dim]

        # Concat: [text; x_cond]
        tokens = torch.cat([text_token, x_emb], dim=1)  # [B, 14, latent_dim]

        # Add positional embedding
        tokens = tokens + self.pos_emb.unsqueeze(0)

        # Transformer
        hidden = self.transformer(tokens)  # [B, 14, latent_dim]

        # Option 1: Use text token as context (index 0)
        context = hidden[:, 0]  # [B, latent_dim]

        # Option 2: Mean pooling (uncomment if preferred)
        # context = hidden.mean(dim=1)  # [B, latent_dim]

        return context
    

# ==============================
# 新的去噪网络：组合 condition encoder + diffusion head
# ==============================
class MotionDenoisingNetwork(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,
        cond_frames=13,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="ViT-B/32"
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        # Condition encoder (no y_noisy!)
        self.condition_encoder = MotionConditionTransformer(
            motion_dim=motion_dim,
            cond_frames=cond_frames,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            clip_model=clip_model
        )

        # Time embedding MLP (output dim = latent_dim for alignment)
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # Diffusion head
        self.diffusion_head = DiffusionHead(
            motion_dim=motion_dim,
            num_frames=num_frames,
            context_dim=latent_dim,
            hidden_dim=1024,
            num_layers=4
        )

    def forward(self, y_noisy, t, x_cond, text):
        """
        y_noisy: [B, F, D] — current noisy target
        t: [B] — time steps
        x_cond: [B, 13, D] — past motion
        text: List[str]
        Returns: pred_x0 [B, F, D]
        """
        B = y_noisy.shape[0]
        device = y_noisy.device

        # 1. Encode condition (text + x_cond) → context
        context = self.condition_encoder(x_cond, text)  # [B, latent_dim]

        # 2. Time embedding
        t_emb_raw = timestep_embedding(t, self.latent_dim).to(device)  # [B, latent_dim]
        t_emb = self.time_mlp(t_emb_raw)  # [B, latent_dim]

        # 3. Denoise with diffusion head
        pred_x0 = self.diffusion_head(y_noisy, context, t_emb)  # [B, F, D]

        return pred_x0
    

# ==============================
# Transformer-based Diffusion Head (2-layer)
# ==============================
class TransformerDiffusionHead(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,
        context_dim=512,
        latent_dim=512,        # token dim inside head
        num_layers=2,
        num_heads=8,
        ff_size=1024,
        dropout=0.1
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        # Project y_noisy to latent space
        self.motion_proj_in = nn.Linear(motion_dim, latent_dim)

        # We'll add a "condition token" that fuses context + t_emb
        self.cond_token_proj = nn.Linear(context_dim * 2, latent_dim)

        # Positional embedding for y tokens (num_frames)
        self.pos_emb_y = nn.Parameter(torch.randn(num_frames, latent_dim))
        # No pos emb for cond token (it's global)

        # 2-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.motion_proj_out = nn.Linear(latent_dim, motion_dim)

    def forward(self, y_noisy, context, t_emb):
        """
        y_noisy: [B, F, D]
        context: [B, C]
        t_emb:   [B, C]
        Returns: [B, F, D] — predicted x0
        """
        B = y_noisy.shape[0]
        device = y_noisy.device

        # 1. Project y_noisy to latent tokens
        y_tokens = self.motion_proj_in(y_noisy)  # [B, F, latent_dim]
        y_tokens = y_tokens + self.pos_emb_y.unsqueeze(0)  # add pos emb

        # 2. Create a single conditioning token from [context; t_emb]
        cond_token = torch.cat([context, t_emb], dim=-1)  # [B, 2*C]
        cond_token = self.cond_token_proj(cond_token)     # [B, latent_dim]
        cond_token = cond_token.unsqueeze(1)              # [B, 1, latent_dim]

        # 3. Concatenate: [cond_token; y_tokens]
        tokens = torch.cat([cond_token, y_tokens], dim=1)  # [B, 1+F, latent_dim]

        # 4. Transformer encoder (2 layers)
        hidden = self.transformer(tokens)  # [B, 1+F, latent_dim]

        # 5. Extract y part (skip cond token)
        y_out = hidden[:, 1:]  # [B, F, latent_dim]

        # 6. Project back to motion space
        pred = self.motion_proj_out(y_out)  # [B, F, motion_dim]

        return pred
    

class TransformerMotionDenoisingNetwork(nn.Module):
    def __init__(
        self,
        motion_dim=272,
        num_frames=4,
        cond_frames=13,
        latent_dim=512,        # for condition encoder
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="ViT-B/32"
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        # Condition encoder (text + x_cond → context)
        self.condition_encoder = MotionConditionTransformer(
            motion_dim=motion_dim,
            cond_frames=cond_frames,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            clip_model=clip_model
        )

        # Time embedding MLP (output dim = latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # ✅ Replace MLP head with 2-layer Transformer head
        self.diffusion_head = TransformerDiffusionHead(
            motion_dim=motion_dim,
            num_frames=num_frames,
            context_dim=latent_dim,
            latent_dim=latent_dim,      # same as encoder for simplicity
            num_layers=2,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout
        )

    def forward(self, y_noisy, t, x_cond, text):
        B = y_noisy.shape[0]
        device = y_noisy.device

        # Encode condition
        context = self.condition_encoder(x_cond, text)  # [B, latent_dim]

        # Time embedding
        t_emb_raw = timestep_embedding(t, self.latent_dim).to(device)
        t_emb = self.time_mlp(t_emb_raw)  # [B, latent_dim]

        # Denoise with transformer head
        pred_x0 = self.diffusion_head(y_noisy, context, t_emb)  # [B, F, D]

        return pred_x0
    

