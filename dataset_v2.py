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
        - post: 过去现在未来13帧归一化动作（后验动作）
        - x: 过去6帧+现在1帧归一化动作（输入）
        - y: 未来5帧+现在1帧+未来1帧归一化动作（输出，与x有重叠）
        - traj: 6个关键关节点在过去6帧的位置+速度（辅助轨迹特征）

    修改说明：
        - 不再只为每个动作 ID 保留第一个完整 caption；
        - 所有满足 `#0.0#0.0` 的完整描述都会被加入样本列表；
        - 每个 (motion, caption) 对视为一个独立样本，提升数据多样性。
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
            split_file = pjoin(self.data_root, 'split', f'{split}.txt')  # 划分文件路径（如 train.txt）
            self.meta_dir = pjoin(self.data_root, 'mean_std')        # 归一化参数（均值和标准差）目录
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only 't2m_272' is available.")

        # 加载全局归一化参数（均值和标准差），用于将动作特征归一化到标准正态分布
        self.mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))  # shape: (272,)
        self.std = np.load(pjoin(self.meta_dir, 'Std.npy'))    # shape: (272,)

        # 读取指定划分（如 train / val / test）中的动作 ID 列表
        with cs.open(split_file, 'r', encoding='utf-8') as f:
            id_list = [line.strip() for line in f.readlines()]

        # 初始化样本列表：每个元素为字典 {'motion_path', 'caption', 'original_id'}
        self.samples = []

        # 遍历所有动作 ID，加载有效的（motion, caption）对
        for name in tqdm(id_list, desc=f"Loading {split} samples"):
            motion_path = pjoin(self.motion_dir, name + '.npy')
            text_path = pjoin(self.text_dir, name + '.txt')

            # 若动作或文本文件缺失，跳过该 ID
            if not (os.path.exists(motion_path) and os.path.exists(text_path)):
                continue

            # 尝试加载动作数据（T × 272）
            try:
                full_motion = np.load(motion_path)  # shape: [T, 272]
                T = full_motion.shape[0]
            except Exception as e:
                continue  # 加载失败则跳过

            # 跳过长度不足 min_seq_length 的动作序列（无法裁剪出14帧）
            if T < self.min_seq_length:
                continue

            # 读取文本文件：每行格式为 "caption#parsed#start_time#end_time"
            with cs.open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 遍历文本文件中的每一行
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                parts = line.split('#')
                if len(parts) < 4:
                    continue  # 格式不合法，跳过

                caption = parts[0]  # 原始自然语言描述
                # 尝试解析起止时间（第3、4部分）
                try:
                    f_tag = float(parts[2]) if parts[2].strip() != '' else 0.0
                    to_tag = float(parts[3]) if parts[3].strip() != '' else 0.0
                except ValueError:
                    continue  # 非法时间格式，跳过

                # 处理可能的 NaN（虽然罕见，但安全起见）
                if np.isnan(f_tag):
                    f_tag = 0.0
                if np.isnan(to_tag):
                    to_tag = 0.0

                # 仅保留“完整动作”描述：即未指定时间片段（start=0.0, end=0.0）
                if f_tag == 0.0 and to_tag == 0.0:
                    # 将此 (motion, caption) 对作为一个独立样本加入列表
                    self.samples.append({
                        'motion_path': motion_path,
                        'caption': caption,
                        'original_id': name
                    })
                    # 注意：不再 break！继续处理该文件中的其他完整描述
                    # 这样，一个动作文件可能对应多个样本（不同 caption）

        print(f"✅ Loaded {len(self.samples)} samples from {len(id_list)} IDs (min length >= {min_seq_length}).")

    def __len__(self):
        """返回数据集总样本数（即 (motion, caption) 对的数量）"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取第 idx 个样本。
        返回：
            caption: str —— 当前样本对应的自然语言描述
            post: [13, 272] 过去现在未来13帧归一化动作（后验动作）
            x: [7, 272] 过去6帧+现在1帧归一化动作（输入）
            y: [7, 272] 未来5帧+现在1帧+未来1帧归一化动作（输出，与x有重叠）
            traj: [6, 36] —— 6个关键关节点 × (位置3 + 速度3) × 6帧的历史轨迹特征
        """
        # 从样本列表中获取当前样本信息
        sample = self.samples[idx]
        motion = np.load(sample['motion_path'])     # 加载完整动作序列 [T, 272]
        caption = sample['caption']                 # 对应的文本描述
        original_id = sample['original_id']         # 原始动作ID（用于保存）

        T = motion.shape[0]

        # 步骤1: 从原始动作中随机截取一段长度为 min_seq_length 的子序列
        if T > self.min_seq_length:
            start = random.randint(0, T - self.min_seq_length)
            motion_seq = motion[start:start + self.min_seq_length]  # [min_seq_length, 272]
        else:
            motion_seq = motion  # 若长度刚好，直接使用全部

        # 步骤2: 从该子序列中再随机取连续13帧（用于构造x和y）
        start_13 = random.randint(0, self.min_seq_length - 13)
        motion_13 = motion_seq[start_13:start_13 + 13]  # [13, 272]

        # 步骤3: 构造后验post输入x和目标y
        post = motion_13.copy()      # [13, 272]
        x_raw = motion_13[:7].copy()   # [7, 272]
        y_raw = motion_13[1:8].copy()   # [7, 272]

        # 步骤4: 使用全局均值和标准差对动作进行归一化
        post = (post - self.mean) / self.std  # [13, 272]
        x = (x_raw - self.mean) / self.std  # [7, 272]
        y = (y_raw - self.mean) / self.std  # [7, 272]

        # 步骤5: 构建轨迹特征（traj）
        # HumanML3D 272维动作向量的结构定义：
        GLOBAL_DIM = 8          # 全局根节点信息（如根位置、根旋转等）
        NUM_JOINTS = 22         # 总关节数
        JOINT_POS_DIM = 3       # 每个关节的位置维度 (x, y, z)
        JOINT_VEL_DIM = 3       # 每个关节的速度维度 (vx, vy, vz)

        POS_START = GLOBAL_DIM                          # 关节位置起始索引 = 8
        VEL_START = POS_START + NUM_JOINTS * JOINT_POS_DIM  # 速度起始索引 = 8 + 66 = 74

        # 定义6个关键关节点（基于HumanML3D的22关节索引）
        joint_map = {
            'root': 0,          # 根节点
            'left_toe': 4,      # 左脚趾
            'right_toe': 8,     # 右脚趾
            'head': 13,         # 头部
            'left_hand': 17,    # 左手
            'right_hand': 21    # 右手
        }

        # 使用x的前6帧（即 earliest 6 frames within the 13-frame window）构建轨迹
        past_x = x[:6]  # [6, 272]

        # 为每个关键点提取位置和速度，拼接成 [6, 6] 特征
        traj_parts = []
        for joint_name, joint_idx in joint_map.items():
            # 计算该关节位置在272维中的起始索引
            pos_start = POS_START + joint_idx * JOINT_POS_DIM
            pos = past_x[:, pos_start:pos_start + JOINT_POS_DIM]  # [6, 3]

            # 计算该关节速度在272维中的起始索引
            vel_start = VEL_START + joint_idx * JOINT_VEL_DIM
            vel = past_x[:, vel_start:vel_start + JOINT_VEL_DIM]  # [6, 3]

            # 拼接位置与速度 → [6, 6]
            pos_vel = np.concatenate([pos, vel], axis=-1)
            traj_parts.append(pos_vel)

        # 拼接6个关键点 → [6, 36]（6帧 × 6点 × 6维 = 6×36）
        traj = np.concatenate(traj_parts, axis=-1)  # [6, 36]

        # 步骤6（可选）: 保存原始未归一化的x_raw（13帧）及对应文本
        if self.save_original_npy_dir is not None:
            save_path = pjoin(self.save_original_npy_dir, f"{original_id}_{idx:06d}.npy")
            np.save(save_path, x_raw)
            # 同时保存文本描述（用于可视化或调试）
            with open(save_path.replace('.npy', '.txt'), 'w', encoding='utf-8') as f:
                f.write(caption)

        # 步骤7: 转换为 float32（PyTorch 默认精度）
        return (
            caption,
            post.astype(np.float32),    # [13, 272]
            x.astype(np.float32),      # [7, 272]
            y.astype(np.float32),      # [7, 272]
            traj.astype(np.float32)    # [6, 36]
        )


def DATALoader(dataset_name, batch_size, num_workers=8, save_original_npy_dir=None, split='train'):
    """
    构建数据加载器
    参数：
        dataset_name: 数据集名称（如 't2m_272'）
        batch_size: 批大小
        num_workers: 数据加载子进程数
        save_original_npy_dir: 是否保存原始动作片段（用于可视化）
        split: 数据划分（'train' / 'val' / 'test'）
    返回：
        dataloader: PyTorch DataLoader 对象
    """
    dataset = TextMotionPredictionDataset(
        dataset_name=dataset_name,
        split=split,
        min_seq_length=30,
        save_original_npy_dir=save_original_npy_dir
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,          # 每个 epoch 打乱样本顺序（含不同 caption）
        num_workers=num_workers,
        drop_last=True,        # 丢弃最后一个不完整 batch
        collate_fn=lambda batch: (
            [item[0] for item in batch],  # List[str]: 所有 caption
            torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0),  # [B, 13, 272]
            torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0),  # [B, 7, 272]
            torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0),  # [B, 7, 272]
            torch.stack([torch.from_numpy(item[4]) for item in batch], dim=0)   # [B, 6, 36]
        )
    )
    return dataloader


# test_dataset.py

def test_dataset():
    print("🧪 正在测试 TextMotionPredictionDataset...")

    # 创建数据集（使用 'val' 或 'test' 避免过长训练集）
    dataset = TextMotionPredictionDataset(
        dataset_name='t2m_272',
        split='val',               # 建议用 val 或 test，样本少、加载快
        min_seq_length=30,
        save_original_npy_dir=None  # 不保存原始数据
    )

    print(f"✅ 数据集加载完成，共 {len(dataset)} 个样本")

    # 测试单个样本
    if len(dataset) == 0:
        print("❌ 警告：数据集为空！请检查路径或 split 文件")
        return

    caption, post, x, y, traj = dataset[0]
    print(f"\n📝 样本 0 的 caption: {caption}")
    print(f"📊 post.shape: {post.shape} (应为 [13, 272])")
    print(f"📊 x.shape: {x.shape} (应为 [7, 272])")
    print(f"📊 y.shape: {y.shape} (应为 [7, 272])")
    print(f"📊 traj.shape: {traj.shape} (应为 [6, 36])")

    # 验证形状
    assert post.shape == (13, 272), f"post 形状错误: {post.shape}"
    assert x.shape == (7, 272), f"x 形状错误: {x.shape}"
    assert y.shape == (7, 272), f"y 形状错误: {y.shape}"
    assert traj.shape == (6, 36), f"traj 形状错误: {traj.shape}"
    assert isinstance(caption, str), "caption 不是字符串"

    print("\n✅ 单样本测试通过！")

    # 测试 DataLoader（小 batch）
    try:
        dataloader = DATALoader(
            dataset_name='t2m_272',
            batch_size=2,
            num_workers=0,      # 测试时设为0避免多进程问题
            split='val'
        )

        batch = next(iter(dataloader))
        captions, post_batch, x_batch, y_batch, traj_batch = batch

        print(f"\n📦 DataLoader batch 测试:")
        print(f"   batch size: {len(captions)}")
        print(f"   post_batch.shape: {post_batch.shape} (应为 [2, 13, 272])")
        print(f"   x_batch.shape: {x_batch.shape} (应为 [2, 7, 272])")
        print(f"   y_batch.shape: {y_batch.shape} (应为 [2, 7, 272])")
        print(f"   traj_batch.shape: {traj_batch.shape} (应为 [2, 6, 36])")

        assert post_batch.shape == (2, 13, 272)
        assert x_batch.shape == (2, 7, 272)
        assert y_batch.shape == (2, 7, 272)
        assert traj_batch.shape == (2, 6, 36)
        print("\n✅ DataLoader 测试通过！")
    except StopIteration:
        print("⚠️  DataLoader 为空（样本数 < batch_size），跳过 batch 测试")

    print("\n🎉 所有测试通过！数据集修改成功。")

if __name__ == "__main__":
    test_dataset()