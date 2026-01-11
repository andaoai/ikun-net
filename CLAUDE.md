# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

ikun-net 是一个基于 ImageNet 的图像处理与分析项目。

**第一步 - 颜色分离：** `ColorSeparator` 使用 RGB 三维区间组合将图像分离为颜色层并生成二值掩码。这是后续图像分析和模型训练的数据预处理基础。

## 架构

```
ikunnet/
├── __init__.py           # 包导出：ColorSeparator, ColorGroup
├── color_separator.py    # 核心颜色分离算法
└── cli.py               # CLI 处理器（separate-colors 命令）

data/
├── __init__.py
└── download_dataset.py  # ImageNet 下载器（通过 Kaggle API）

main.py                  # CLI 入口点
tests/                   # 测试脚本
```

### 核心组件

1. **ColorSeparator 类** ([ikunnet/color_separator.py](ikunnet/color_separator.py))
   - `analyze_color_groups()` - 按 RGB 区间组合对像素分组
   - `create_binary_masks()` - 为每个颜色组生成二值掩码
   - `separate_colors()` - 完整流程（加载 → 预处理 → 分析 → 掩码）
   - `save_masks()` - 保存掩码、metadata.json 和 summary.txt

2. **CLI** ([main.py](main.py), [ikunnet/cli.py](ikunnet/cli.py))
   - 主命令：`separate-colors`
   - 使用 Rich 控制台输出格式化表格
   - 如果未指定输入图像，则从 ImageNet 数据集随机选择

3. **ColorGroup 数据类** - 表示颜色组合，包含区间范围、中心颜色、像素数和百分比

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
