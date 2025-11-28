import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from datetime import datetime

from dataset_v2 import TextMotionPredictionDataset, DATALoader  # 或直接使用 DataLoader
from MDM import MotionDiffusionTransformer, MotionDDPM

def validate(model, val_loader, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for captions, x_cond, y_gt, traj in val_loader:
            x_cond = x_cond.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                loss = model(x0=y_gt, x_cond=x_cond, text=captions)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss

def main():
    # ======================
    # 超参数设置
    # ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 4
    lr = 2e-4
    weight_decay = 1e-4
    num_epochs = 100
    timesteps = 1000
    save_dir = './checkpoints_ddpm_t2m'
    os.makedirs(save_dir, exist_ok=True)

    # ======================
    # 数据加载：Train & Val
    # ======================
    print("🔧 Loading training dataset (split=train)...")
    train_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='train'  # 👈 关键：使用训练集
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)
        )
    )
    print(f"✅ Training set: {len(train_dataset)} samples")

    print("🔧 Loading validation dataset (split=val)...")
    val_dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        min_seq_length=30,
        split='val'  # 👈 关键：使用验证集
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时无需打乱
        num_workers=num_workers,
        drop_last=False,  # 验证时保留最后不完整 batch
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)
        )
    )
    print(f"✅ Validation set: {len(val_dataset)} samples")

    # ======================
    # 模型初始化
    # ======================
    print("🧠 Initializing DDPM model...")
    denoise_net = MotionDiffusionTransformer(
        motion_dim=272,
        num_frames=4,
        cond_frames=13,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        clip_model="openai/clip-vit-base-patch32"
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
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ======================
    # 训练循环 + 验证
    # ======================
    print("🚀 Start training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        ddpm_model.train()
        epoch_loss = 0.0

        # --- Training ---
        for batch_idx, (captions, x_cond, y_gt, traj) in enumerate(train_loader):
            x_cond = x_cond.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                loss = ddpm_model(x0=y_gt, x_cond=x_cond, text=captions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx} | Train Loss: {loss.item():.6f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # --- Validation ---
        print("🔍 Running validation...")
        avg_val_loss = validate(ddpm_model, val_loader, device)
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

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
            print(f"🏆 New best model saved (Val Loss: {avg_val_loss:.6f})")

        # --- 定期保存（可选）---
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(save_dir, f"ddpm_t2m_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, ckpt_path)
            print(f"💾 Periodic checkpoint saved: {ckpt_path}")

    print("🎉 Training completed! Best validation loss: {:.6f}".format(best_val_loss))

if __name__ == "__main__":
    main()