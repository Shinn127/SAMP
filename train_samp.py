# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
from datetime import datetime
import logging

# 👉 替换为您的 SAMP 模型（确保路径正确）
from SAMP import SAMP_Framework  # 假设您的模型保存在 SAMP.py
from dataset_nfp import TextMotionPredictionDataset, DATALoader

# ======================
# 设置日志
# ======================
def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"samp_train_{timestamp}.log")

    logger = logging.getLogger("SAMP_TrainLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def validate(model, val_loader, device, logger):
    """在验证集上评估 SAMP 模型"""
    model.eval()
    total_loss = 0.0
    total_diff_loss = 0.0
    total_kl_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            # 解包数据: (text, full_motion, future_motion, past_motion, current_motion, past_traj)
            captions, x_full, y_future, x_past, x_curr, traj = batch
            x_full = x_full.to(device, non_blocking=True)      # [B,13,272]
            y_future = y_future.to(device, non_blocking=True)  # [B,4,272]
            x_past = x_past.to(device, non_blocking=True)      # [B,6,272]
            x_curr = x_curr.to(device, non_blocking=True)      # [B,1,272]
            traj = traj.to(device, non_blocking=True)          # [B,7,36]

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(
                    texts=captions,
                    x_motion=x_full,
                    future_motion=y_future,
                    past_motion=x_past,
                    current_motion=x_curr,
                    past_traj=traj,
                    mode='train'
                )
                loss_dict = model.loss_function(
                    diffusion_loss=outputs['diffusion_loss'],
                    mu_delta_post=outputs['mu_delta_post'],
                    logvar_delta_post=outputs['logvar_delta_post'],
                    mu_delta_prior=outputs['mu_delta_prior'],
                    logvar_delta_prior=outputs['logvar_delta_prior'],
                    lambda_kl=1.0,
                    lambda_l2=0.1
                )
                total_loss += loss_dict['total_loss'].item()
                total_diff_loss += loss_dict['diffusion_loss']
                total_kl_loss += loss_dict['kl_delta']
                count += 1

    avg_total = total_loss / count
    avg_diff = total_diff_loss / count
    avg_kl = total_kl_loss / count
    logger.info(f"  → Val Avg Loss: total={avg_total:.6f}, diff={avg_diff:.6f}, kl={avg_kl:.6f}")
    return avg_total


def main():
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ======================
    # 超参数
    # ======================
    batch_size = 32
    lr = 2e-4
    weight_decay = 1e-4
    num_epochs = 100
    save_dir = './checkpoints_samp'
    os.makedirs(save_dir, exist_ok=True)

    # ======================
    # 数据加载
    # ======================
    logger.info("Loading datasets...")
    train_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='train'
    )
    val_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='val'
    )

    # Collate 函数：确保输出 6 个元素
    def collate_fn(batch):
        # 假设 dataset 返回: (text, full_motion, future_motion, past_motion, current_motion, past_traj)
        # full_motion: 13帧, future: 4帧, past:6, curr:1, traj:7×36
        texts = [item[0] for item in batch]
        x_full = torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0)
        y_future = torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0)
        traj = torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)
        return texts, x_full, y_future, x_full[:, :6], x_full[:, 6:7], traj

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ======================
    # 初始化 SAMP 模型
    # ======================
    logger.info("Initializing SAMP model...")
    samp_model = SAMP_Framework(
        joint_num=22,
        clip_model_name="ViT-B/32",
        z_dim=256,
        device=device
    ).to(device)

    # ======================
    # 优化器 & 混合精度
    # ======================
    optimizer = optim.AdamW(
        samp_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.99)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    # ======================
    # 训练循环
    # ======================
    logger.info("Start SAMP training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        samp_model.train()
        epoch_total_loss = 0.0
        epoch_diff_loss = 0.0
        epoch_kl_loss = 0.0

        for batch_idx, (captions, x_full, y_future, x_past, x_curr, traj) in enumerate(train_loader):
            # 移至 GPU
            x_full = x_full.to(device, non_blocking=True)
            y_future = y_future.to(device, non_blocking=True)
            x_past = x_past.to(device, non_blocking=True)
            x_curr = x_curr.to(device, non_blocking=True)
            traj = traj.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # 前向传播（训练模式）
                outputs = samp_model(
                    texts=captions,
                    x_motion=x_full,
                    future_motion=y_future,
                    past_motion=x_past,
                    current_motion=x_curr,
                    past_traj=traj,
                    mode='train'
                )

                # 计算复合损失
                loss_dict = samp_model.loss_function(
                    diffusion_loss=outputs['diffusion_loss'],
                    mu_delta_post=outputs['mu_delta_post'],
                    logvar_delta_post=outputs['logvar_delta_post'],
                    mu_delta_prior=outputs['mu_delta_prior'],
                    logvar_delta_prior=outputs['logvar_delta_prior'],
                    lambda_kl=1.0,
                    lambda_l2=1.0
                )
                total_loss = loss_dict['total_loss']

            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(samp_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_diff_loss += loss_dict['diffusion_loss']
            epoch_kl_loss += loss_dict['kl_delta']
            global_step += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx} | "
                    f"Total: {total_loss.item():.6f} | Diff: {loss_dict['diffusion_loss']:.6f} | KL: {loss_dict['kl_delta']:.6f} | L2: {loss_dict['l2_mean']:.6f}"
                )

        # Epoch 结束
        avg_train_total = epoch_total_loss / len(train_loader)
        avg_train_diff = epoch_diff_loss / len(train_loader)
        avg_train_kl = epoch_kl_loss / len(train_loader)
        scheduler.step()

        logger.info(f"Epoch {epoch+1} | Train Avg → Total: {avg_train_total:.6f}, Diff: {avg_train_diff:.6f}, KL: {avg_train_kl:.6f}")

        # 验证
        logger.info("Running validation...")
        avg_val_loss = validate(samp_model, val_loader, device, logger)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "samp_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': samp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_total_loss': avg_train_total,
            }, best_ckpt_path)
            logger.info(f"New best model saved (Val Loss: {avg_val_loss:.6f})")

        # 定期保存
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(save_dir, f"samp_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': samp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_total_loss': avg_train_total,
            }, ckpt_path)
            logger.info(f"Periodic checkpoint saved: {ckpt_path}")

    logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()