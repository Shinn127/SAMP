import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
from datetime import datetime
import logging  # 👈 新增

from dataset_v2 import TextMotionPredictionDataset, DATALoader
from MDM import MotionDiffusionTransformer, MotionDDPM

# ======================
# 设置日志
# ======================
def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # 创建 logger
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（尤其在 Jupyter 或多次运行时）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def validate(model, val_loader, device, logger):  # 👈 新增 logger 参数
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for captions, x_cond, y_gt, traj in val_loader:
            x_cond = x_cond.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                loss = model(x0=y_gt, x_cond=x_cond, text=captions)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def main():
    logger = setup_logger()  # 👈 初始化日志

    # ======================
    # 超参数设置
    # ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lr = 2e-4
    weight_decay = 1e-4
    num_epochs = 100
    timesteps = 1000
    save_dir = './checkpoints_ddpm_t2m'
    os.makedirs(save_dir, exist_ok=True)

    # ======================
    # 数据加载：Train & Val
    # ======================
    logger.info("Loading training dataset (split=train)...")
    train_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)
        )
    )
    logger.info(f"Training set: {len(train_dataset)} samples")

    logger.info("Loading validation dataset (split=val)...")
    val_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='val'
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)
        )
    )
    logger.info(f"Validation set: {len(val_dataset)} samples")

    # ======================
    # 模型初始化
    # ======================
    logger.info("Initializing DDPM model...")
    denoise_net = MotionDiffusionTransformer(
        motion_dim=272,
        num_frames=4,
        cond_frames=7,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    ddpm_model = MotionDDPM(
        model=denoise_net,
        timesteps=timesteps,
        loss_type='l2'
    ).to(device)

    # ======================
    # 优化器 & 混合精度
    # ======================
    optimizer = optim.AdamW(
        ddpm_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.99)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    # ======================
    # 训练循环 + 验证
    # ======================
    logger.info("Start training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        ddpm_model.train()
        epoch_loss = 0.0

        for batch_idx, (captions, x_cond, y_gt, traj) in enumerate(train_loader):
            x_cond = x_cond.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                loss = ddpm_model(x0=y_gt, x_cond=x_cond, text=captions)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx} | Train Loss: {loss.item():.6f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # --- Validation ---
        logger.info("Running validation...")
        avg_val_loss = validate(ddpm_model, val_loader, device, logger)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "ddpm_t2m_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, best_ckpt_path)
            logger.info(f"New best model saved (Val Loss: {avg_val_loss:.6f})")

        # --- 定期保存 ---
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(save_dir, f"ddpm_t2m_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, ckpt_path)
            logger.info(f"Periodic checkpoint saved: {ckpt_path}")

    logger.info("Training completed! Best validation loss: {:.6f}".format(best_val_loss))


if __name__ == "__main__":
    main()