import argparse
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset_nfp import TextMotionPredictionDataset
from samp_v2 import (
    DistilBERTTextEncoder,
    LatentMotionPredictor,
    MotionPosteriorEncoder,
    SAMPFramework,
    SAMPriorVAE,
    X0DiffusionHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAMP v2 with dataset_nfp")
    parser.add_argument("--dataset_name", type=str, default="t2m_272")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_seq_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_samp_v2")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")

    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--motion_dim", type=int, default=272)
    parser.add_argument("--traj_dim", type=int, default=44)

    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"samp_v2_train_{timestamp}.log")

    logger = logging.getLogger("samp_v2_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(
    batch: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        [item[0] for item in batch],
        torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),  # post: [B,13,272]
        torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),  # x: [B,7,272]
        torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0),  # y: [B,7,272]
        torch.stack([torch.from_numpy(item[4]) for item in batch], dim=0),  # traj: [B,6,44]
    )


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextMotionPredictionDataset(
        dataset_name=args.dataset_name,
        split="train",
        min_seq_length=args.min_seq_length,
    )
    val_dataset = TextMotionPredictionDataset(
        dataset_name=args.dataset_name,
        split="val",
        min_seq_length=args.min_seq_length,
    )

    use_cuda = torch.cuda.is_available()
    common = {
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "collate_fn": collate_batch,
    }
    if args.num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> SAMPFramework:
    return SAMPFramework(
        text_encoder=DistilBERTTextEncoder(out_dim=args.z_dim),
        posterior_encoder=MotionPosteriorEncoder(
            motion_dim=args.motion_dim,
            text_dim=args.z_dim,
            z_dim=args.z_dim,
        ),
        prior_vae=SAMPriorVAE(
            traj_dim=args.traj_dim,
            z_dim=args.z_dim,
        ),
        latent_predictor=LatentMotionPredictor(
            z_dim=args.z_dim,
            motion_dim=args.motion_dim,
            latent_dim=args.latent_dim,
        ),
        diffusion_head=X0DiffusionHead(
            motion_dim=args.motion_dim,
            cond_dim=args.latent_dim,
            num_steps=8,
        ),
    )


def move_batch_to_device(
    batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    texts, post, x_hist, y_target, past_traj = batch
    return (
        texts,
        post.to(device, non_blocking=True),
        x_hist.to(device, non_blocking=True),
        y_target.to(device, non_blocking=True),
        past_traj.to(device, non_blocking=True),
    )


def run_epoch(
    model: SAMPFramework,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool,
    grad_clip: float,
    train: bool,
    log_interval: int,
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    diff_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_steps = 0

    for step, batch in enumerate(loader):
        texts, post, x_hist, y_target, past_traj = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                out = model.forward_train(
                    texts=texts,
                    motion_seq=post,
                    x_hist=x_hist,
                    y_target=y_target,
                    past_traj=past_traj,
                )
                losses = model.compute_loss(out, kl_weight=kl_weight)
                total_loss = losses["total_loss"]

            if train:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

        total_loss_sum += float(total_loss.detach().item())
        diff_loss_sum += float(losses["diff_loss"].item())
        kl_loss_sum += float(losses["kl_loss"].item())
        num_steps += 1

        if train and (step % log_interval == 0):
            logger.info(
                "Epoch %d/%d Step %d/%d | total=%.6f diff=%.6f kl=%.6f",
                epoch + 1,
                total_epochs,
                step,
                len(loader) - 1,
                float(total_loss.detach().item()),
                float(losses["diff_loss"].item()),
                float(losses["kl_loss"].item()),
            )

    if num_steps == 0:
        return {"total_loss": 0.0, "diff_loss": 0.0, "kl_loss": 0.0}

    return {
        "total_loss": total_loss_sum / num_steps,
        "diff_loss": diff_loss_sum / num_steps,
        "kl_loss": kl_loss_sum / num_steps,
    }


def save_checkpoint(
    path: str,
    model: SAMPFramework,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(args)
    logger.info("Train steps/epoch: %d | Val steps: %d", len(train_loader), len(val_loader))

    logger.info("Building model...")
    model = build_model(args).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Start training for %d epochs...", args.epochs)
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            kl_weight=args.kl_weight,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            train=True,
            log_interval=args.log_interval,
            logger=logger,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            kl_weight=args.kl_weight,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            train=False,
            log_interval=args.log_interval,
            logger=logger,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train(total=%.6f diff=%.6f kl=%.6f) | val(total=%.6f diff=%.6f kl=%.6f) | lr=%.6e",
            epoch + 1,
            args.epochs,
            train_metrics["total_loss"],
            train_metrics["diff_loss"],
            train_metrics["kl_loss"],
            val_metrics["total_loss"],
            val_metrics["diff_loss"],
            val_metrics["kl_loss"],
            optimizer.param_groups[0]["lr"],
        )

        latest_path = os.path.join(args.output_dir, "latest.pth")
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_path = os.path.join(args.output_dir, "best.pth")
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved new best checkpoint: %s", best_path)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_path = os.path.join(args.output_dir, f"epoch_{epoch + 1:03d}.pth")
            save_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved periodic checkpoint: %s", periodic_path)

    logger.info("Training finished. Best val total loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
