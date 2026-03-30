# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

ikun-net 是一个基于 ImageNet 的图像处理与分析项目，使用颜色分离 + Transformer 架构进行图像分类。

**核心思想：** 将图像分解为颜色掩码序列，通过 Transformer 编码后聚合进行分类。

## 算法架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ikun-net 算法架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入图像 (H×W×3)                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │     1. ColorSeparator (颜色分离)         │                               │
│  │     - preprocess_image: 640×640 标准化   │                               │
│  │     - analyze_color_groups: RGB区间分组  │                               │
│  │     - create_binary_masks: 生成二值掩码  │                               │
│  └─────────────────────────────────────────┘                               │
│       │                                                                     │
│       ▼                                                                     │
│  N 个颜色掩码 (每个 640×640) + RGB 中心颜色                                  │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │     2. ColorTokenEncoder (Token编码)     │                               │
│  │     ┌─────────────┐  ┌───────────────┐  │                               │
│  │     │ CNNEncoder  │  │RGBPositional  │  │                               │
│  │     │ (形状特征)   │+│Encoding(颜色) │  │                               │
│  │     └─────────────┘  └───────────────┘  │                               │
│  └─────────────────────────────────────────┘                               │
│       │                                                                     │
│       ▼                                                                     │
│  Token序列 (N×256) + Valid Mask                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │     3. QueryPooling (Token聚合)          │                               │
│  │     - Learnable Query + Cross-Attention  │                               │
│  │     - 变长序列 → 单一向量 (256)           │                               │
│  └─────────────────────────────────────────┘                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────┐                               │
│  │     4. Classifier (分类头)               │                               │
│  │     LayerNorm → MLP → Linear → Logits    │                               │
│  └─────────────────────────────────────────┘                               │
│       │                                                                     │
│       ▼                                                                     │
│  类别预测 (num_classes=1000)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
ikunnet/
├── __init__.py           # 包导出：ColorSeparator, ColorGroup
├── color_separator.py    # 核心颜色分离算法
├── cli.py               # CLI 处理器（separate-colors 命令）
└── models/
    ├── encoder.py                    # CNN 编码器 (CNNEncoder, ResNetEncoder)
    ├── color_token_encoder.py        # Token 编码器 (形状 + 颜色位置编码)
    ├── rgb_positional_encoding.py    # RGB 位置编码 (正弦/MLP)
    ├── query_pooling.py              # Token 聚合模块 (Cross-Attention)
    ├── vicreg_loss.py                # VICReg 损失函数
    ├── projection_head.py            # 投影头
    └── color_transformer_classifier.py  # 端到端分类模型

ikunnet/training/
├── trainer.py            # 训练器 (VICReg 表征学习)
├── dataset.py            # 预处理掩码数据集
├── online_dataset.py     # 在线掩码提取数据集
└── augmentation.py       # 掩码数据增强

data/
├── __init__.py
├── download_dataset.py   # ImageNet 下载器（通过 Kaggle API）
└── prepare_masks.py      # 掩码预处理脚本

main.py                   # CLI 入口点
tests/                    # 测试脚本
```

### 核心模块

#### 1. ColorSeparator ([ikunnet/color_separator.py](ikunnet/color_separator.py))

颜色分离算法：RGB 三维区间组合

```python
# RGB空间划分
编码公式: group_key = r_idx * 10000 + g_idx * 100 + b_idx

# 数据流程
preprocess_image:    原始图像 → 640×640 标准化图像 + padding信息
analyze_color_groups: 标准化图像 → ColorGroup列表 (区间范围、中心颜色、像素统计)
create_binary_masks:  图像 + ColorGroups → 二值掩码字典 {group_id: mask}
```

#### 2. ColorTokenEncoder ([ikunnet/models/color_token_encoder.py](ikunnet/models/color_token_encoder.py))

双路编码器：`token = shape_features + color_position_features`

| 路径 | 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| 形状编码 | CNNEncoder | 掩码 (1×224×224) | 256维 | 提取掩码形状特征 |
| 颜色编码 | RGBPositionalEncoding | RGB值 (3,) | 256维 | 位置编码表示颜色信息 |

#### 3. QueryPooling ([ikunnet/models/query_pooling.py](ikunnet/models/query_pooling.py))

可学习 Query + Cross-Attention 聚合变长 Token 序列，灵感来自 Perceiver IO / Set Transformer。

#### 4. 双损失训练 ([ikunnet/models/color_transformer_classifier.py](ikunnet/models/color_transformer_classifier.py))

```python
Total Loss = CrossEntropy(logits, labels) + VICRegLoss(tokens)

# VICReg 损失 (无需负样本的表征学习)
- 方差项: 鼓励每个维度有足够方差 (std → 1.0)
- 协方差项: 去相关不同维度 (off-diagonal → 0)
```

## 开发命令

### 运行工具
```bash
# 处理指定图像
uv run python main.py separate-colors image.jpg

# 使用自定义参数
uv run python main.py separate-colors image.jpg --interval-size 10 --min-pixels 500 --output custom_output

# 从 ImageNet 随机选择图像
uv run python main.py separate-colors --dataset data/imagenet1k/imagenet-mini/

# 仅分析（不保存掩码）
uv run python main.py separate-colors image.jpg --analyze-only
```

### 环境配置
```bash
# 安装依赖
uv sync
```

### 测试
```bash
# 运行测试脚本（创建合成图像并运行颜色分离）
uv run python tests/test_color_separator.py
```

### 数据集下载
```bash
# 下载 ImageNet（需要 Kaggle API 凭证）
uv run python data/download_dataset.py --dataset mini --save-path data/imagenet1k/
```

## 关键实现细节

- **图像预处理：** 图像缩放至 640x640，同时保持宽高比（短边用零填充）
- **颜色区间算法：** 每个 RGB 通道被划分为 `interval_size`（默认 20）大小的区间。像素按其 (r_idx, g_idx, b_idx) 组合分组。
- **分组编码：** 区间索引编码为 `r_idx * 10000 + g_idx * 100 + b_idx` 以便通过 numpy 高效分组
- **输出：** 为每个颜色组生成二值掩码图像，并保存 metadata.json 和 summary.txt

## 依赖项

项目使用 UV 作为包管理器，使用清华大学 PyPI 镜像。主要依赖：
- PyTorch 生态系统（torch、torchvision）
- OpenCV（opencv-python）用于图像 I/O
- NumPy 用于数组操作
- Rich 用于 CLI 输出格式化
- Kaggle API 用于数据集下载
