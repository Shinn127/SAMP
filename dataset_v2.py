import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm
import os
import random

class TextMotionPredictionDataset(data.Dataset):
    """
    文本-动作预测数据集
    输入：自然语言描述（caption）
    输出：
        - x: 过去13帧归一化动作（用于条件）
        - y: 未来4帧归一化动作（预测目标，与x有重叠）
        - traj: 6个关键关节点在过去6帧的位置+速度（辅助轨迹特征）
    """

    def __init__(
        self,
        dataset_name,
        split='train',
        min_seq_length=30,                  # 原始动作序列的最小长度（用于裁剪）
        save_original_npy_dir=None          # 可选：保存原始未归一化x_raw（13帧）的目录
    ):
        self.dataset_name = dataset_name
        self.min_seq_length = min_seq_length
        self.save_original_npy_dir = save_original_npy_dir
        self.split = split

        # 如果指定了保存目录，且目录不存在，则创建
        if save_original_npy_dir and not os.path.exists(save_original_npy_dir):
            os.makedirs(save_original_npy_dir)

        # 支持的唯一数据集：HumanML3D 的 272 维动作表示（t2m_272）
        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')   # 动作 .npy 文件目录
            self.text_dir = pjoin(self.data_root, 'texts')           # 文本 .txt 文件目录
            self.fps = 30                                            # 帧率（30 FPS）
            split_file = pjoin(self.data_root, 'split', f'{split}.txt')  # 使用测试集划分
            self.meta_dir = pjoin(self.data_root, 'mean_std')        # 归一化参数目录
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only 't2m_272' is available.")

        # 加载动作数据的全局均值和标准差（用于归一化）
        self.mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))  # shape: (272,)
        self.std = np.load(pjoin(self.meta_dir, 'Std.npy'))    # shape: (272,)

        # 读取测试集 ID 列表（每个 ID 对应一个动作-文本对）
        with cs.open(split_file, 'r', encoding='utf-8') as f:
            id_list = [line.strip() for line in f.readlines()]

        # 存储有效样本：每个元素为字典 {'motion_path', 'caption', 'original_id'}
        self.samples = []

        # 遍历所有 ID，加载有效样本
        for name in tqdm(id_list, desc="Loading samples"):
            motion_path = pjoin(self.motion_dir, name + '.npy')
            text_path = pjoin(self.text_dir, name + '.txt')

            # 跳过缺失文件的样本
            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue

            # 尝试加载动作数据（T × 272）
            try:
                full_motion = np.load(motion_path)  # shape: [T, 272]
                T = full_motion.shape[0]
            except Exception as e:
                continue  # 加载失败则跳过

            # 跳过长度不足 min_seq_length 的动作序列
            if T < self.min_seq_length:
                continue

            # 读取文本文件（每行格式：caption#xxx#start_time#end_time）
            with cs.open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 遍历所有文本描述行
            for line in lines:
                parts = line.strip().split('#')
                if len(parts) < 4:
                    continue  # 格式不合法则跳过

                caption = parts[0]
                f_tag = float(parts[2]) if parts[2] else 0.0  # 起始时间（秒）
                to_tag = float(parts[3]) if parts[3] else 0.0  # 结束时间（秒）

                # 处理 NaN（罕见情况）
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                # 只保留完整动作描述（即未指定时间范围的 caption）
                if f_tag == 0.0 and to_tag == 0.0:
                    self.samples.append({
                        'motion_path': motion_path,
                        'caption': caption,
                        'original_id': name  # 用于生成保存文件名
                    })
                    break  # 每个动作只取第一个完整描述

        print(f"✅ Loaded {len(self.samples)} samples (min length >= {min_seq_length}).")

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本：
            caption: str
            x: [13, 272] → 归一化过去13帧（连续）
            y: [4, 272]  → 归一化未来4帧（与x重叠后3帧）
            traj: [6, 36] → 6个关键点 × (位置3 + 速度3) × 6帧
        """
        sample = self.samples[idx]
        motion = np.load(sample['motion_path'])     # [T, 272]
        caption = sample['caption']
        original_id = sample['original_id']

        T = motion.shape[0]

        # 步骤1: 从原始动作中随机截取一段长度为 min_seq_length 的子序列
        if T > self.min_seq_length:
            start = random.randint(0, T - self.min_seq_length)
            motion_seq = motion[start:start + self.min_seq_length]  # [L, 272], L = min_seq_length
        else:
            motion_seq = motion  # 若不足，则用全部

        # 步骤2: 从该子序列中再随机取连续14帧（用于构造x和y）
        start_14 = random.randint(0, self.min_seq_length - 14)
        motion_14 = motion_seq[start_14:start_14 + 14]  # [14, 272]

        # 步骤3: 构造输入x（13帧）和目标y（4帧）——注意：两者重叠（x[10:13] == y[0:3]）
        x_raw = motion_14[:13].copy()   # [13, 272] ← 连续13帧作为条件
        y_raw = motion_14[-4:].copy()   # [4, 272]  ← 最后4帧作为预测目标

        # 步骤4: 对x和y进行归一化
        x = (x_raw - self.mean) / self.std  # [13, 272]
        y = (y_raw - self.mean) / self.std  # [4, 272]

        # 步骤5: 定义动作向量的语义结构（HumanML3D 272维）
        GLOBAL_DIM = 8          # 全局根节点信息（如根位置、根旋转等）
        NUM_JOINTS = 22         # 关节数量
        JOINT_POS_DIM = 3       # 每个关节的位置维度 (x, y, z)
        JOINT_VEL_DIM = 3       # 每个关节的速度维度 (vx, vy, vz)

        POS_START = GLOBAL_DIM                          # 关节位置起始索引: 8
        VEL_START = POS_START + NUM_JOINTS * JOINT_POS_DIM  # 关节速度起始索引: 8 + 66 = 74

        # 关注的6个关键关节点（索引基于22个关节点）
        joint_map = {
            'root': 0,          # 根节点
            'left_toe': 4,      # 左脚趾
            'right_toe': 8,     # 右脚趾
            'head': 13,         # 头部
            'left_hand': 17,    # 左手
            'right_hand': 21    # 右手
        }

        # 从x的前6帧构建轨迹特征（traj）
        past_x = x[:7]  # [6, 272] ← 取最早6帧作为“历史轨迹”

        traj_parts = []
        for name, joint_idx in joint_map.items():
            # 计算该关节的位置起始索引
            pos_start = POS_START + joint_idx * JOINT_POS_DIM
            pos = past_x[:, pos_start:pos_start + JOINT_POS_DIM]  # [6, 3]

            # 计算该关节的速度起始索引
            vel_start = VEL_START + joint_idx * JOINT_VEL_DIM
            vel = past_x[:, vel_start:vel_start + JOINT_VEL_DIM]  # [6, 3]

            # 拼接位置与速度 → [6, 6]
            pos_vel = np.concatenate([pos, vel], axis=-1)
            traj_parts.append(pos_vel)

        # 拼接所有关键点 → [6, 36]（6帧 × 6点 × 6维 = 6×36）
        traj = np.concatenate(traj_parts, axis=-1)  # [6, 36]

        # 步骤6（可选）: 保存原始未归一化的x_raw（13帧）及对应文本
        if self.save_original_npy_dir is not None:
            save_path = pjoin(self.save_original_npy_dir, f"{original_id}_{idx:06d}.npy")
            np.save(save_path, x_raw)  # 保存未归一化的13帧
            # 同时保存文本描述
            with open(save_path.replace('.npy', '.txt'), 'w', encoding='utf-8') as f:
                f.write(caption)

        # 步骤7: 转换为 float32（PyTorch 默认精度）
        return (
            caption,
            x[:7].astype(np.float32),      # [13, 272]
            y.astype(np.float32),      # [4, 272]
            traj.astype(np.float32)    # [6, 36]
        )


def DATALoader(dataset_name, batch_size, num_workers=4, save_original_npy_dir=None):
    """
    构建数据加载器
    """
    dataset = TextMotionPredictionDataset(
        dataset_name=dataset_name,
        min_seq_length=30,
        save_original_npy_dir=save_original_npy_dir
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,          # 训练/评估时打乱顺序
        num_workers=num_workers,
        drop_last=True,        # 丢弃最后不足一个 batch 的样本
        collate_fn=lambda batch: (
            [item[0] for item in batch],  # List[str]: captions
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),  # [B, 13, 272]
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),  # [B, 4, 272]
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0)   # [B, 6, 36]
        )
    )
    return dataloader