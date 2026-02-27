import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm
import os
import random

class TextMotionPredictionDataset(data.Dataset):
    def __init__(
        self,
        dataset_name,
        min_seq_length=121,
        save_original_npy_dir=None,  # 👈 新增参数：保存原始121帧的目录
        split='test'
    ):
        self.dataset_name = dataset_name
        self.min_seq_length = min_seq_length
        self.save_original_npy_dir = save_original_npy_dir
        self.split = split

        if save_original_npy_dir and not os.path.exists(save_original_npy_dir):
            os.makedirs(save_original_npy_dir)

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.fps = 30
            split_file = pjoin(self.data_root, 'split', f'{split}.txt')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # Load normalization stats
        self.mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))  # (272,)
        self.std = np.load(pjoin(self.meta_dir, 'Std.npy'))    # (272,)

        # Load split IDs
        with cs.open(split_file, 'r', encoding='utf-8') as f:
            id_list = [line.strip() for line in f.readlines()]

        self.samples = []  # (motion_path, caption, original_id)

        for name in tqdm(id_list, desc="Loading samples"):
            motion_path = pjoin(self.motion_dir, name + '.npy')
            text_path = pjoin(self.text_dir, name + '.txt')

            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue

            try:
                full_motion = np.load(motion_path)  # [T, 272]
                T = full_motion.shape[0]
            except Exception as e:
                continue

            if T < self.min_seq_length:
                continue

            with cs.open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('#')
                if len(parts) < 4:
                    continue
                caption = parts[0]
                try:
                    f_tag = float(parts[2]) if parts[2].strip() != '' else 0.0
                    to_tag = float(parts[3]) if parts[3].strip() != '' else 0.0
                except ValueError:
                    continue
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag == 0.0 and to_tag == 0.0:
                    self.samples.append({
                        'motion_path': motion_path,
                        'caption': caption,
                        'original_id': name  # 用于生成保存文件名
                    })

        print(f"✅ Loaded {len(self.samples)} samples (min length >= {min_seq_length}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = np.load(sample['motion_path'])  # [T, 272], T >= 121
        caption = sample['caption']
        original_id = sample['original_id']

        T = motion.shape[0]
        if T > 121:
            start = random.randint(0, T - 121)
            motion_121 = motion[start:start + 121]  # [121, 272]
        else:
            motion_121 = motion  # exactly 121

        # 构建 x 和 y
        center = 60
        x_indices = (
                [center - 60, center - 50, center - 40, center - 30, center - 20, center - 10] +
                [center] +
                [center + 10, center + 20, center + 30, center + 40, center + 50, center + 60]
        )
        x_raw = motion_121[x_indices].copy()  # 保留未归一化副本
        x = x_raw.copy()
        y = motion_121[[center + 1, center + 11, center + 21, center + 31]].copy()

        # 归一化 x, y（traj 将从归一化后的 x 中提取）
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # ——————————————————————————————————————————
        # ✅ 构建归一化的 traj: [6, 36]
        # ——————————————————————————————————————————

        joint_map = {
            'root': 0,
            'left_toe': 4,
            'right_toe': 8,
            'head': 13,
            'left_hand': 17,
            'right_hand': 21
        }

        K = 22
        JOINT_DIM = 12
        JOINTS_START = 8

        POS_SLICE = slice(0, 3)  # 位置
        VEL_SLICE = slice(3, 6)  # 速度

        # 提取过去6帧（归一化后）
        past_x = x[:6]  # [6, 272]

        traj_parts = []
        for name, joint_idx in joint_map.items():
            start = JOINTS_START + joint_idx * JOINT_DIM
            joint_slice = slice(start, start + JOINT_DIM)
            joint_data = past_x[:, joint_slice]  # [6, 12]
            pos_vel = joint_data[:, np.r_[POS_SLICE, VEL_SLICE]]  # [6, 6]
            traj_parts.append(pos_vel)

        traj = np.concatenate(traj_parts, axis=-1)  # [6, 6*6] = [6, 36]

        # ——————————————————————————————————————————
        # 保存原始 npy（可选）
        if self.save_original_npy_dir is not None:
            save_path = pjoin(self.save_original_npy_dir, f"{original_id}_{idx:06d}.npy")
            np.save(save_path, x_raw)  # 保存未归一化的 x
            with open(save_path.replace('.npy', '.txt'), 'w') as f:
                f.write(caption)

        # ——————————————————————————————————————————
        # ✅ 返回：caption, x, y, traj（全部 float32）
        return (
            caption,
            x.astype(np.float32),
            y.astype(np.float32),
            traj.astype(np.float32)
        )


def DATALoader(dataset_name, batch_size, num_workers=4, save_original_npy_dir=None, split='test'):
    dataset = TextMotionPredictionDataset(
        dataset_name=dataset_name,
        save_original_npy_dir=save_original_npy_dir,
        split=split
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],  # caption
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),  # x: [B, 13, 272]
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),  # y: [B, 4, 272]
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)   # traj: [B, 6, 36]
        )
    )
    return dataloader



def main():
    SAVE_NPY_DIR = "motion"
    OUTPUT_VIDEO_DIR = "./anim"

    os.makedirs(SAVE_NPY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    print("🔄 从数据集加载样本并保存原始 121 帧 272D 数据...")
    dataloader = DATALoader(
        dataset_name='t2m_272',
        batch_size=2,
        num_workers=0,
        save_original_npy_dir=SAVE_NPY_DIR
    )

    captions, x, y, traj = next(iter(dataloader))
    print(f"📝 Caption: {captions[0]}")

    # 获取刚保存的 .npy 文件（取最新或第一个）
    npy_files = sorted([f for f in os.listdir(SAVE_NPY_DIR) if f.endswith('.npy')])
    if not npy_files:
        raise RuntimeError("No .npy file saved!")

    latest_npy = npy_files[-1]  # 取最新生成的
    output_name = latest_npy.replace('.npy', '')

    # ✅ 检查并打印对应的 .txt 文件
    txt_path = os.path.join(SAVE_NPY_DIR, f"{output_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            saved_caption = f.read().strip()
        print(f"🔖 已保存文本到: {txt_path}")
        print(f"   内容: \"{saved_caption}\"")
    else:
        print("⚠️ 警告: 未找到对应的 .txt 文件！请检查 dataset_v1.py")

    # 调用可视化
    print("🎥 正在生成 3D 可视化视频...")
    cmd = (
        f"python visualize_272.py "
        f"--input_dir {SAVE_NPY_DIR} "
        f"--mode pos "
        f"--output_dir {OUTPUT_VIDEO_DIR}"
    )
    print(f"📦 执行: {cmd}")
    result = os.system(cmd)

    if result != 0:
        print("❌ 可视化失败！")
        return

    video_path = os.path.join(OUTPUT_VIDEO_DIR, f"pos_{output_name}.mp4")
    if os.path.exists(video_path):
        print(f"✅ 成功生成视频: {video_path}")
    else:
        print("❌ 视频未生成")


if __name__ == "__main__":
    main()
