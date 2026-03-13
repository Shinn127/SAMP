import argparse
import codecs as cs
import json
import math
import os
import random
import sys
from os.path import join as pjoin
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from MDM_ori import MDMOriginalDenoiser, MotionDDPMOriginal


def ensure_evaluator_import_paths(bundle_path: str):
    # Direct bundle stores pickled module paths (e.g. mld.*), make them importable at runtime.
    bundle_dir = os.path.abspath(os.path.dirname(bundle_path) or ".")
    candidates = [bundle_dir, os.path.abspath(os.path.join(bundle_dir, ".."))]
    for p in candidates:
        if os.path.isdir(os.path.join(p, "mld")) and p not in sys.path:
            sys.path.insert(0, p)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDM generation quality with evaluator_272 bundle")
    parser.add_argument("--mdm_ckpt", type=str, required=True, help="Path to MDM checkpoint (best.pth / last.pth)")
    parser.add_argument(
        "--evaluator_bundle",
        type=str,
        default="./evaluator/evaluator_272_src.pth",
        help="Path to evaluator direct bundle (.pth)",
    )
    parser.add_argument("--dataset_name", type=str, default="t2m_272")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=512, help="-1 means use the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--motion_dim", type=int, default=272)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--out_json", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_mdm(args, device: torch.device) -> MotionDDPMOriginal:
    # Rebuild model structure from eval args, then load checkpoint weights.
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
    )
    model = MotionDDPMOriginal(denoiser=denoiser, timesteps=args.timesteps)
    try:
        ckpt = torch.load(args.mdm_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.mdm_ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_evaluator(bundle_path: str, device: torch.device):
    ensure_evaluator_import_paths(bundle_path)
    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(bundle_path, map_location="cpu")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. Please ensure the folder containing 'mld/' is importable. "
            f"Current bundle path: {bundle_path}"
        ) from e

    textencoder = bundle["textencoder"].to(device).eval()
    motionencoder = bundle["motionencoder"].to(device).eval()
    for p in textencoder.parameters():
        p.requires_grad = False
    for p in motionencoder.parameters():
        p.requires_grad = False
    return textencoder, motionencoder


def get_lengths_from_padded(motion_batch: torch.Tensor) -> torch.Tensor:
    nonzero = (motion_batch.abs().sum(dim=-1) > 0.0)
    lengths = nonzero.sum(dim=-1)
    lengths = torch.clamp(lengths, min=1)
    return lengths.long()


def pad_or_trim_motion(motion: torch.Tensor, target_len: int) -> torch.Tensor:
    # Evaluator motionencoder may have fixed max_len (e.g. 300). Force shape alignment here.
    cur_len = motion.shape[1]
    if cur_len == target_len:
        return motion
    if cur_len > target_len:
        return motion[:, :target_len, :]
    pad = torch.zeros(
        motion.shape[0],
        target_len - cur_len,
        motion.shape[2],
        device=motion.device,
        dtype=motion.dtype,
    )
    return torch.cat([motion, pad], dim=1)


def as_embedding(x) -> torch.Tensor:
    # Bundle modules may return Distribution(loc=...) or raw tensor/tuple, normalize to tensor.
    if hasattr(x, "loc"):
        return x.loc
    if isinstance(x, (tuple, list)):
        for item in x:
            if torch.is_tensor(item):
                return item
    if torch.is_tensor(x):
        return x
    raise TypeError(f"Unsupported encoder output type: {type(x)}")


def calculate_matching_score(et: np.ndarray, em: np.ndarray) -> float:
    assert et.shape[0] == em.shape[0]
    dist = np.linalg.norm(et - em, axis=1)
    return float(dist.mean())


def calculate_r_precision(et: np.ndarray, em: np.ndarray, top_k: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, float]:
    dist_mat = np.linalg.norm(et[:, None, :] - em[None, :, :], axis=2)
    sorted_idx = np.argsort(dist_mat, axis=1)
    gt = np.arange(et.shape[0])[:, None]
    out = {}
    for k in top_k:
        hit = (sorted_idx[:, :k] == gt).any(axis=1).mean()
        out[f"R@{k}"] = float(hit)
    return out


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0.0))
    return dists


def calculate_top_k(mat: np.ndarray, top_k: int) -> np.ndarray:
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = np.zeros(size, dtype=bool)
    top_k_list = []
    eff_k = min(top_k, bool_mat.shape[1])
    for i in range(eff_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    # Keep output shape stable when batch size < top_k.
    while len(top_k_list) < top_k:
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1: np.ndarray, embedding2: np.ndarray, top_k: int, sum_all: bool = False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


def calculate_diversity(activation: np.ndarray, diversity_times: int) -> float:
    assert len(activation.shape) == 2
    if activation.shape[0] <= 1:
        return 0.0
    diversity_times = min(diversity_times, activation.shape[0] - 1)
    first_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    second_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations: np.ndarray):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


class EvalOneCaptionDataset(torch.utils.data.Dataset):
    """Eval dataset with one-to-one mapping: one motion file -> first valid full caption."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        max_motion_length: int = 196,
        min_seq_length: int = 30,
        unit_length: int = 4,
    ):
        if dataset_name != "t2m_272":
            raise ValueError("Only t2m_272 is supported.")
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.data_root = "./humanml3d_272"
        self.motion_dir = pjoin(self.data_root, "motion_data")
        self.text_dir = pjoin(self.data_root, "texts")
        self.meta_dir = pjoin(self.data_root, "mean_std")
        self.mean = np.load(pjoin(self.meta_dir, "Mean.npy"))
        self.std = np.load(pjoin(self.meta_dir, "Std.npy"))
        split_file = pjoin(self.data_root, "split", f"{split}.txt")

        with cs.open(split_file, "r", encoding="utf-8") as f:
            id_list = [line.strip() for line in f.readlines()]

        self.samples = []
        for name in id_list:
            motion_path = pjoin(self.motion_dir, name + ".npy")
            text_path = pjoin(self.text_dir, name + ".txt")
            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue
            try:
                motion = np.load(motion_path)
            except Exception:
                continue
            if motion.shape[0] < min_seq_length:
                continue

            first_caption = None
            with cs.open(text_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split("#")
                    if len(parts) < 4:
                        continue
                    caption = parts[0]
                    try:
                        f_tag = float(parts[2]) if parts[2].strip() != "" else 0.0
                        to_tag = float(parts[3]) if parts[3].strip() != "" else 0.0
                    except ValueError:
                        continue
                    if np.isnan(f_tag):
                        f_tag = 0.0
                    if np.isnan(to_tag):
                        to_tag = 0.0
                    if f_tag == 0.0 and to_tag == 0.0:
                        # Keep only the first full-sequence caption for one-to-one eval.
                        first_caption = caption
                        break
            if first_caption is None:
                continue
            self.samples.append({"motion_path": motion_path, "caption": first_caption})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = np.load(sample["motion_path"])
        caption = sample["caption"]
        t = int(motion.shape[0])

        if t > self.max_motion_length:
            # Match HumanML style crop rule: mostly max_len, occasionally max_len-unit_length.
            if random.random() < 2.0 / 3.0:
                m_length = self.max_motion_length
            else:
                m_length = self.max_motion_length - self.unit_length
            # Random crop for long motions.
            start = random.randint(0, t - m_length)
            motion_seq = motion[start:start + m_length]
        else:
            m_length = t
            motion_seq = motion

        motion_seq = (motion_seq - self.mean) / self.std
        if m_length < self.max_motion_length:
            # Zero-pad to fixed-length tensor for batching/sampling shape consistency.
            pad = np.zeros((self.max_motion_length - m_length, motion_seq.shape[1]), dtype=motion_seq.dtype)
            motion_seq = np.concatenate([motion_seq, pad], axis=0)
        return caption, motion_seq.astype(np.float32), int(m_length)


def build_eval_loader(args) -> DataLoader:
    dataset = EvalOneCaptionDataset(
        dataset_name=args.dataset_name,
        split=args.split,
        max_motion_length=args.max_motion_length,
        min_seq_length=30,
        unit_length=args.unit_length,
    )
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = list(range(args.num_samples))
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.tensor([item[2] for item in batch], dtype=torch.long),
        ),
    )
    return loader


@torch.no_grad()
def run_eval(args):
    device = make_device(args.device)
    print(f"[Info] device: {device}")
    print(f"[Info] loading MDM ckpt: {args.mdm_ckpt}")
    model = load_mdm(args, device)
    print(f"[Info] loading evaluator bundle: {args.evaluator_bundle}")
    textencoder, motionencoder = load_evaluator(args.evaluator_bundle, device)
    evaluator_max_len = int(getattr(motionencoder, "max_len", -1))
    if evaluator_max_len > 0:
        print(f"[Info] evaluator motion max_len: {evaluator_max_len}")
    loader = build_eval_loader(args)
    print(f"[Info] eval batches: {len(loader)}")
    print(f"[Info] eval motion sequence length fixed to: {args.max_motion_length}")

    all_et = []
    all_em_gt = []
    all_em_pred = []
    # Keep the same accumulation style as the reference code:
    # top-k hit counts and trace-based matching score are summed then normalized by nb_sample.
    R_precision_real = torch.tensor([0.0, 0.0, 0.0], device=device)
    R_precision_pred = torch.tensor([0.0, 0.0, 0.0], device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    matching_score_pred = torch.tensor(0.0, device=device)
    nb_sample = torch.tensor(0.0, device=device)

    for i, (captions, motion_gt, m_length) in enumerate(tqdm(loader, desc="Evaluating", total=len(loader))):
        motion_gt = motion_gt.to(device=device, dtype=torch.float32, non_blocking=True)
        m_length = m_length.to(device=device)
        b = motion_gt.shape[0]
        motion_pred = model.sample(
            shape=torch.Size([b, args.max_motion_length, args.motion_dim]),
            device=device,
            texts=captions,
            guidance_scale=args.guidance_scale,
        )

        if evaluator_max_len > 0:
            motion_gt = pad_or_trim_motion(motion_gt, evaluator_max_len)
            motion_pred = pad_or_trim_motion(motion_pred, evaluator_max_len)

        len_gt = m_length
        # Generated motions are sampled in fixed length; use max length as predicted valid length.
        len_pred = torch.full_like(m_length, fill_value=args.max_motion_length)
        if evaluator_max_len > 0:
            len_gt = torch.clamp(len_gt, max=evaluator_max_len)
            len_pred = torch.clamp(len_pred, max=evaluator_max_len)

        et = as_embedding(textencoder(captions))
        em_gt = as_embedding(motionencoder(motion_gt, len_gt))
        em_pred = as_embedding(motionencoder(motion_pred, len_pred))

        all_et.append(et.detach().cpu())
        all_em_gt.append(em_gt.detach().cpu())
        all_em_pred.append(em_pred.detach().cpu())

        temp_R, temp_match = calculate_R_precision(
            et.detach().cpu().numpy(), em_gt.detach().cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_real += torch.tensor(temp_R, device=device, dtype=torch.float32)
        matching_score_real += torch.tensor(float(temp_match), device=device, dtype=torch.float32)
        temp_R, temp_match = calculate_R_precision(
            et.detach().cpu().numpy(), em_pred.detach().cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_pred += torch.tensor(temp_R, device=device, dtype=torch.float32)
        matching_score_pred += torch.tensor(float(temp_match), device=device, dtype=torch.float32)
        nb_sample += float(b)

        if (i + 1) % 10 == 0 or (i + 1) == len(loader):
            print(f"[Info] processed {i + 1}/{len(loader)} batches")

    em_gt_np = torch.cat(all_em_gt, dim=0).numpy()
    em_pred_np = torch.cat(all_em_pred, dim=0).numpy()
    gt_mu, gt_cov = calculate_activation_statistics(em_gt_np)
    mu, cov = calculate_activation_statistics(em_pred_np)

    metrics = {}
    # FID and Diversity follow the reference implementation style.
    metrics["FID"] = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    metrics["Diversity_GT"] = calculate_diversity(em_gt_np, args.diversity_times if nb_sample > 300 else 100)
    metrics["Diversity_Pred"] = calculate_diversity(em_pred_np, args.diversity_times if nb_sample > 300 else 100)
    r_real = (R_precision_real / nb_sample).detach().cpu().numpy()
    r_pred = (R_precision_pred / nb_sample).detach().cpu().numpy()
    metrics["GT_R@1"] = float(r_real[0])
    metrics["GT_R@2"] = float(r_real[1])
    metrics["GT_R@3"] = float(r_real[2])
    metrics["Pred_R@1"] = float(r_pred[0])
    metrics["Pred_R@2"] = float(r_pred[1])
    metrics["Pred_R@3"] = float(r_pred[2])
    metrics["Matching_GT"] = float((matching_score_real / nb_sample).item())
    metrics["Matching_Pred"] = float((matching_score_pred / nb_sample).item())

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    metrics = run_eval(args)
    print("\n===== Evaluation Results =====")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.6f}" if isinstance(metrics[k], float) else f"{k}: {metrics[k]}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[Info] metrics saved to: {args.out_json}")


if __name__ == "__main__":
    main()
