import argparse
import logging
import os
from datetime import datetime

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import DATALoaderGeneration
from MDM_ori import MDMOriginalDenoiser, MotionDDPMOriginal


def setup_logger(log_dir: str = "./logs", name: str = "train_mdm_ori") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def build_args():
    parser = argparse.ArgumentParser(description="Train MDM (MDM_ori.py) on TextMotionGenerationDataset")
    parser.add_argument("--dataset_name", type=str, default="t2m_272")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--min_seq_length", type=int, default=30)
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--motion_dim", type=int, default=272)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--lambda_pos", type=float, default=0.0)
    parser.add_argument("--lambda_vel", type=float, default=0.0)
    parser.add_argument("--lambda_foot", type=float, default=0.0)
    parser.add_argument("--foot_contact_threshold", type=float, default=0.0025)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_mdm_ori")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    train_loader = DATALoaderGeneration(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
        min_seq_length=args.min_seq_length,
        max_motion_length=args.max_motion_length,
        unit_length=args.unit_length,
    )
    val_loader = DATALoaderGeneration(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
        min_seq_length=args.min_seq_length,
        max_motion_length=args.max_motion_length,
        unit_length=args.unit_length,
    )
    return train_loader, val_loader


def build_model(args, device: torch.device):
    denoiser = MDMOriginalDenoiser(
        motion_dim=args.motion_dim,
        seq_len=args.max_motion_length,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        clip_model=args.clip_model,
        cond_drop_prob=args.cond_drop_prob,
    ).to(device)
    model = MotionDDPMOriginal(
        denoiser=denoiser,
        timesteps=args.timesteps,
        lambda_pos=args.lambda_pos,
        lambda_vel=args.lambda_vel,
        lambda_foot=args.lambda_foot,
        foot_contact_threshold=args.foot_contact_threshold,
    ).to(device)
    return model


@torch.no_grad()
def validate(model, val_loader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    sum_stats = {"total": 0.0, "simple": 0.0, "pos": 0.0, "vel": 0.0, "foot": 0.0}
    total_steps = 0

    for batch in val_loader:
        with autocast(device_type="cuda", enabled=use_amp):
            _, stats = model.forward_generation_batch(batch=batch, device=device, return_components=True)
        for k in sum_stats:
            sum_stats[k] += float(stats[k].item())
        total_steps += 1

    if total_steps == 0:
        return {k: float("inf") for k in sum_stats}
    return {k: v / total_steps for k, v in sum_stats.items()}


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, scaler, best_val_loss: float):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)


def maybe_resume(path: str, model, optimizer, scheduler, scaler, logger: logging.Logger):
    if not path:
        return 0, float("inf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    logger.info("Resumed from %s (start_epoch=%d, best_val=%.6f)", path, start_epoch, best_val_loss)
    return start_epoch, best_val_loss


def main():
    args = build_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Args: %s", vars(args))

    logger.info("Building dataloaders...")
    train_loader, val_loader = build_loaders(args)
    logger.info("Train steps/epoch: %d | Val steps: %d", len(train_loader), len(val_loader))

    logger.info("Building MDM model...")
    model = build_model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_loss = maybe_resume(
        path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger,
    )

    logger.info("Start training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sums = {"total": 0.0, "simple": 0.0, "pos": 0.0, "vel": 0.0, "foot": 0.0}

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                loss, stats = model.forward_generation_batch(batch=batch, device=device, return_components=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in train_sums:
                train_sums[k] += float(stats[k].item())
            if step % 50 == 0:
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Train total %.6f | Lsimple %.6f | Lpos %.6f | Lvel %.6f | Lfoot %.6f",
                    epoch + 1,
                    args.epochs,
                    step,
                    len(train_loader),
                    float(stats["total"].item()),
                    float(stats["simple"].item()),
                    float(stats["pos"].item()),
                    float(stats["vel"].item()),
                    float(stats["foot"].item()),
                )

        scheduler.step()
        num_train_steps = max(len(train_loader), 1)
        train_avg = {k: v / num_train_steps for k, v in train_sums.items()}
        val_avg = validate(model=model, val_loader=val_loader, device=device, use_amp=use_amp)
        logger.info(
            "Epoch %d | Train total %.6f (Lsimple %.6f, Lpos %.6f, Lvel %.6f, Lfoot %.6f) | "
            "Val total %.6f (Lsimple %.6f, Lpos %.6f, Lvel %.6f, Lfoot %.6f)",
            epoch + 1,
            train_avg["total"],
            train_avg["simple"],
            train_avg["pos"],
            train_avg["vel"],
            train_avg["foot"],
            val_avg["total"],
            val_avg["simple"],
            val_avg["pos"],
            val_avg["vel"],
            val_avg["foot"],
        )

        last_ckpt = os.path.join(args.save_dir, "last.pth")
        save_checkpoint(
            path=last_ckpt,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )

        if val_avg["total"] < best_val_loss:
            best_val_loss = val_avg["total"]
            best_ckpt = os.path.join(args.save_dir, "best.pth")
            save_checkpoint(
                path=best_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved new best checkpoint: %s", best_ckpt)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            epoch_ckpt = os.path.join(args.save_dir, f"epoch_{epoch + 1:03d}.pth")
            save_checkpoint(
                path=epoch_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved periodic checkpoint: %s", epoch_ckpt)

    logger.info("Training finished. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
