# SAMP: Text-to-Motion Prediction Playground

一个用于文本驱动人体动作预测的实验仓库，当前包含两条主要技术路线：

- `MDM`：基于 Transformer 的扩散模型（DDPM）动作预测
- `SAMP`：语义对齐的先验/后验潜变量建模 + 扩散生成

项目以 HumanML3D 的 272 维动作表示作为训练与推理输入输出格式。

## 1. 项目目标

给定：

- 文本描述（例如 `a person walks forward and waves`）
- 历史动作片段

预测：

- 后续未来动作序列（272-dim motion representation）

## 2. 目录结构

```text
SAMP/
├── train_mdm.py              # MDM 训练入口
├── train_samp.py             # SAMP 训练入口
├── MDM.py                    # MDM 相关模型定义
├── SAMP.py                   # SAMP 框架实现
├── dataset_nfp.py            # 当前训练常用数据集封装
├── dataset_nsp.py            # 另一版数据采样/封装
├── download_dataset.py       # 从 Hugging Face 下载数据
├── visualization/            # 272 表示还原与可视化脚本
├── utils/                    # 旋转/SMPL-X/几何等工具
├── humanml3d_272/            # 数据集目录（本地）
└── visualize_result/         # 可视化输出示例
```

## 3. 环境依赖

仓库目前没有 `requirements.txt`，建议使用 Python 3.10+ 并手动安装常用依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy scipy tqdm einops
pip install huggingface_hub
pip install git+https://github.com/openai/CLIP.git
```

如果使用 `visualization/smplx2joints.py`，还需要额外依赖（如 `torchgeometry` 以及 SMPL-X 相关资源）。

## 4. 数据准备

### 4.1 自动下载（推荐）

```bash
python download_dataset.py
```

默认会下载到：

- `./humanml3d_272`

### 4.2 数据目录约定

代码默认读取以下结构：

```text
humanml3d_272/
├── motion_data/
├── texts/
├── split/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── mean_std/
    ├── Mean.npy
    └── Std.npy
```

## 5. 训练

### 5.1 训练 MDM

```bash
python train_mdm.py
```

输出：

- 日志：`./logs/`
- checkpoint：`./checkpoints_ddpm_t2m/`

### 5.2 训练 SAMP

```bash
python train_samp.py
```

输出：

- 日志：`./logs/`
- checkpoint：`./checkpoints_samp/`

## 6. 可视化

将 272 维动作 `.npy` 还原并导出视频：

```bash
python visualization/recover_visualize.py \
  --input_dir <npy目录> \
  --mode pos \
  --output_dir visualize_result
```

`mode` 可选：

- `pos`：基于位置恢复并可视化
- `rot`：基于旋转恢复（依赖更多 SMPL 相关组件）

## 7. 当前代码状态与注意事项

该仓库处于实验迭代阶段，以下点在使用前请优先核对：

- `train_*` 与 `dataset_nfp.py` 的字段语义和 shape 需要严格对齐后再做正式训练。
- 仓库包含多个“新旧并行”文件（如 `new.py`、`new_SAMP.py`），默认建议先从 `train_mdm.py` / `train_samp.py` + `MDM.py` / `SAMP.py` 这一组主线阅读。
- 目前缺少统一配置系统（如 `yaml`/`argparse` 全参数化），超参数主要写死在训练脚本中。

## 8. 建议的上手顺序

1. 准备数据并确认 `humanml3d_272` 结构正确。
2. 先跑通一个小 batch 的数据读取与前向（不训练全量）。
3. 再启动 `train_mdm.py` 或 `train_samp.py` 做短轮次试跑。
4. 使用 `visualization/recover_visualize.py` 检查生成动作质量。

## 9. 致谢

- HumanML3D 数据与 272 维动作表示相关工作
- DDPM / 文本条件动作生成相关开源社区实现
