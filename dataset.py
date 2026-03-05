import os
import random
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class TextMotionPredictionDataset(data.Dataset):
    """
    Text-Motion 预测数据集。

    每个样本来自一个 (motion, caption) 对：
    - 只使用文本标注里完整动作描述（f_tag=0.0, to_tag=0.0）。
    - 同一个动作文件中的多个完整 caption 会全部保留。

    返回字段：
    - caption: str
    - post: [13, 272]，连续13帧（归一化后）
    - x: [7, 272]，post 的前7帧（归一化后）
    - y: [7, 272]，post 的第2~8帧（归一化后）
    - traj: [6, 44]，由 x 前6帧提取的关键关节轨迹特征
    """

    def __init__(
        self,
        dataset_name,
        split='train',
        min_seq_length=30,
        save_original_npy_dir=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.min_seq_length = min_seq_length
        self.save_original_npy_dir = save_original_npy_dir

        if save_original_npy_dir and not os.path.exists(save_original_npy_dir):
            os.makedirs(save_original_npy_dir)

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            self.fps = 30
            split_file = pjoin(self.data_root, 'split', f'{split}.txt')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only 't2m_272' is available.")

        self.mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        with cs.open(split_file, 'r', encoding='utf-8') as f:
            id_list = [line.strip() for line in f.readlines()]

        self.samples = []

        for name in tqdm(id_list, desc=f"Loading {split} samples"):
            motion_path = pjoin(self.motion_dir, name + '.npy')
            text_path = pjoin(self.text_dir, name + '.txt')

            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue

            try:
                full_motion = np.load(motion_path)
                T = full_motion.shape[0]
            except Exception:
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

                if np.isnan(f_tag):
                    f_tag = 0.0
                if np.isnan(to_tag):
                    to_tag = 0.0

                # 仅保留完整动作描述；不 break，保留同一动作的多 caption
                if f_tag == 0.0 and to_tag == 0.0:
                    self.samples.append(
                        {
                            'motion_path': motion_path,
                            'caption': caption,
                            'original_id': name,
                        }
                    )

        print(
            f"✅ Loaded {len(self.samples)} samples from {len(id_list)} IDs "
            f"(min length >= {min_seq_length})."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = np.load(sample['motion_path'])
        caption = sample['caption']
        original_id = sample['original_id']

        T = motion.shape[0]

        # 1) 先随机截一段 min_seq_length 的子序列
        if T > self.min_seq_length:
            start = random.randint(0, T - self.min_seq_length)
            motion_seq = motion[start:start + self.min_seq_length]
        else:
            motion_seq = motion

        # 2) 再从子序列中随机取13帧窗口
        start_13 = random.randint(0, self.min_seq_length - 13)
        motion_13 = motion_seq[start_13:start_13 + 13]

        # 3) 构造监督信号
        post = motion_13.copy()      # [13, 272]
        x_raw = motion_13[:7].copy()  # [7, 272]
        y_raw = motion_13[1:8].copy()  # [7, 272]

        # 4) 标准化
        post = (post - self.mean) / self.std
        x = (x_raw - self.mean) / self.std
        y = (y_raw - self.mean) / self.std

        # 5) 构造 traj: 前6帧的关键关节 pos+vel + 全局8维
        GLOBAL_DIM = 8
        NUM_JOINTS = 22
        JOINT_POS_DIM = 3
        JOINT_VEL_DIM = 3

        POS_START = GLOBAL_DIM
        VEL_START = POS_START + NUM_JOINTS * JOINT_POS_DIM

        joint_map = {
            'root': 0,
            'left_toe': 4,
            'right_toe': 8,
            'head': 13,
            'left_hand': 17,
            'right_hand': 21,
        }

        past_x = x[:6]  # [6, 272]
        global_features = past_x[:, :GLOBAL_DIM]  # [6, 8]

        traj_parts = []
        for _, joint_idx in joint_map.items():
            pos_start = POS_START + joint_idx * JOINT_POS_DIM
            pos = past_x[:, pos_start:pos_start + JOINT_POS_DIM]  # [6, 3]

            vel_start = VEL_START + joint_idx * JOINT_VEL_DIM
            vel = past_x[:, vel_start:vel_start + JOINT_VEL_DIM]  # [6, 3]

            traj_parts.append(np.concatenate([pos, vel], axis=-1))  # [6, 6]

        traj_base = np.concatenate(traj_parts, axis=-1)  # [6, 36]
        traj = np.concatenate([global_features, traj_base], axis=-1)  # [6, 44]

        # 可选导出原始 x 及文本，便于可视化/排查
        if self.save_original_npy_dir is not None:
            save_path = pjoin(self.save_original_npy_dir, f"{original_id}_{idx:06d}.npy")
            np.save(save_path, x_raw)
            with open(save_path.replace('.npy', '.txt'), 'w', encoding='utf-8') as f:
                f.write(caption)

        return (
            caption,
            post.astype(np.float32),
            x.astype(np.float32),
            y.astype(np.float32),
            traj.astype(np.float32),
        )


class TextMotionGenerationDataset(data.Dataset):
    """
    Text-Motion 生成数据集。

    每个样本返回：
    - caption: str
    - motion: [196, 272]，按 unit_length 规则裁剪并补零后的序列（归一化后）
    """

    def __init__(
        self,
        dataset_name,
        split='train',
        min_seq_length=30,
        max_motion_length=196,
        unit_length=4,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.min_seq_length = min_seq_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self._max_motion_seq_length = 0

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            split_file = pjoin(self.data_root, 'split', f'{split}.txt')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only 't2m_272' is available.")

        self.mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        with cs.open(split_file, 'r', encoding='utf-8') as f:
            id_list = [line.strip() for line in f.readlines()]

        self.samples = []
        for name in tqdm(id_list, desc=f"Loading {split} generation samples"):
            motion_path = pjoin(self.motion_dir, name + '.npy')
            text_path = pjoin(self.text_dir, name + '.txt')
            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue

            try:
                full_motion = np.load(motion_path)
                T = full_motion.shape[0]
            except Exception:
                continue

            if T < self.min_seq_length:
                continue
            self._max_motion_seq_length = max(self._max_motion_seq_length, T)

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

                if np.isnan(f_tag):
                    f_tag = 0.0
                if np.isnan(to_tag):
                    to_tag = 0.0

                if f_tag == 0.0 and to_tag == 0.0:
                    self.samples.append(
                        {
                            'motion_path': motion_path,
                            'caption': caption,
                        }
                    )

        print(
            f"✅ Loaded {len(self.samples)} generation samples from {len(id_list)} IDs "
            f"(min length >= {min_seq_length})."
        )

    def get_max_motion_seq_length(self):
        """
        返回当前数据集（已过滤 min_seq_length 后）中 motion 的最大序列长度（帧数）。
        """
        return self._max_motion_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = np.load(sample['motion_path'])
        caption = sample['caption']

        T = motion.shape[0]

        if T > self.max_motion_length:
            if random.random() < 2.0 / 3.0:
                m_length = self.max_motion_length
            else:
                m_length = self.max_motion_length - self.unit_length
            start = random.randint(0, T - m_length)
            motion_seq = motion[start:start + m_length]
        else:
            m_length = T
            motion_seq = motion

        motion_seq = (motion_seq - self.mean) / self.std

        if m_length < self.max_motion_length:
            pad_len = self.max_motion_length - m_length
            pad = np.zeros((pad_len, motion_seq.shape[1]), dtype=motion_seq.dtype)
            motion_seq = np.concatenate([motion_seq, pad], axis=0)

        return caption, motion_seq.astype(np.float32)


def DATALoader(dataset_name, batch_size, num_workers=8, save_original_npy_dir=None, split='train'):
    dataset = TextMotionPredictionDataset(
        dataset_name=dataset_name,
        split=split,
        min_seq_length=30,
        save_original_npy_dir=save_original_npy_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0),
            torch.stack([torch.from_numpy(item[4]) for item in batch], dim=0),
        ),
    )
    return dataloader


def DATALoaderGeneration(
    dataset_name,
    batch_size,
    num_workers=8,
    split='train',
    min_seq_length=30,
    max_motion_length=196,
    unit_length=4,
):
    dataset = TextMotionGenerationDataset(
        dataset_name=dataset_name,
        split=split,
        min_seq_length=min_seq_length,
        max_motion_length=max_motion_length,
        unit_length=unit_length,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),
        ),
    )
    return dataloader


def test_dataset():
    print('🧪 Testing TextMotionPredictionDataset...')

    dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        split='val',
        min_seq_length=30,
        save_original_npy_dir=None,
    )

    print(f'✅ Dataset loaded: {len(dataset)} samples')

    if len(dataset) == 0:
        print('❌ Empty dataset. Check paths/split files.')
        return

    caption, post, x, y, traj = dataset[0]
    print(f'caption: {caption}')
    print(f'post.shape: {post.shape}')
    print(f'x.shape: {x.shape}')
    print(f'y.shape: {y.shape}')
    print(f'traj.shape: {traj.shape}')

    assert post.shape == (13, 272)
    assert x.shape == (7, 272)
    assert y.shape == (7, 272)
    assert traj.shape == (6, 44)
    assert isinstance(caption, str)

    try:
        dataloader = DATALoader(
            dataset_name='t2m_272',
            batch_size=2,
            num_workers=0,
            split='val',
        )
        batch = next(iter(dataloader))
        captions, post_batch, x_batch, y_batch, traj_batch = batch

        print('DataLoader batch check:')
        print(f'  batch size: {len(captions)}')
        print(f'  post_batch.shape: {post_batch.shape}')
        print(f'  x_batch.shape: {x_batch.shape}')
        print(f'  y_batch.shape: {y_batch.shape}')
        print(f'  traj_batch.shape: {traj_batch.shape}')

        assert post_batch.shape == (2, 13, 272)
        assert x_batch.shape == (2, 7, 272)
        assert y_batch.shape == (2, 7, 272)
        assert traj_batch.shape == (2, 6, 44)
    except StopIteration:
        print('⚠️ DataLoader is empty (< batch_size).')

    print('🎉 All checks passed.')


def test_generation_dataset():
    print('🧪 Testing TextMotionGenerationDataset...')

    max_motion_length = 196
    dataset = TextMotionGenerationDataset(
        dataset_name='t2m_272',
        split='val',
        min_seq_length=30,
        max_motion_length=max_motion_length,
        unit_length=4,
    )

    print(f'✅ Generation dataset loaded: {len(dataset)} samples')

    if len(dataset) == 0:
        print('❌ Empty generation dataset. Check paths/split files.')
        return

    # 1) 基础输出检查
    caption, motion_seq = dataset[0]
    print(f'caption: {caption}')
    print(f'motion_seq.shape: {motion_seq.shape}')
    assert isinstance(caption, str)
    assert motion_seq.shape == (max_motion_length, 272)
    assert motion_seq.dtype == np.float32

    # 2) 最大长度统计检查（与样本列表中真实最大 T 一致）
    computed_max_t = 0
    has_short_sample = False
    short_sample_idx = None
    short_sample_len = None
    for i, sample in enumerate(dataset.samples):
        t = np.load(sample['motion_path']).shape[0]
        computed_max_t = max(computed_max_t, t)
        if (not has_short_sample) and (dataset.min_seq_length <= t < max_motion_length):
            has_short_sample = True
            short_sample_idx = i
            short_sample_len = t

    print(f'max_motion_seq_length(dataset): {dataset.get_max_motion_seq_length()}')
    print(f'max_motion_seq_length(computed): {computed_max_t}')
    assert dataset.get_max_motion_seq_length() == computed_max_t

    # 3) 补零检查：若存在短序列，尾部 padding 必须全 0
    if has_short_sample:
        _, short_motion = dataset[short_sample_idx]
        tail = short_motion[short_sample_len:]
        assert tail.shape[0] == max_motion_length - short_sample_len
        assert np.allclose(tail, 0.0), 'Padding region should be all zeros.'
        print(f'padding check passed for sample idx={short_sample_idx}, raw_len={short_sample_len}')
    else:
        print('⚠️ No short sample in this split to verify zero-padding behavior.')

    try:
        dataloader = DATALoaderGeneration(
            dataset_name='t2m_272',
            batch_size=2,
            num_workers=0,
            split='val',
            min_seq_length=30,
            max_motion_length=max_motion_length,
            unit_length=4,
        )
        captions, motion_batch = next(iter(dataloader))

        print('Generation DataLoader batch check:')
        print(f'  batch size: {len(captions)}')
        print(f'  motion_batch.shape: {motion_batch.shape}')

        assert len(captions) == 2
        assert motion_batch.shape == (2, max_motion_length, 272)
    except StopIteration:
        print('⚠️ Generation DataLoader is empty (< batch_size).')

    print('🎉 Generation checks passed.')


if __name__ == '__main__':
    test_dataset()
    test_generation_dataset()
