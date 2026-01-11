"""
Color separator module for analyzing images and generating binary color masks.

This module implements RGB 3D interval combination for color layering.
All core operations use PyTorch for CPU/GPU acceleration support.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table


@dataclass
class ColorGroup:
    """颜色组合数据类"""
    group_id: str                  # 组合ID: "r_interval_g_interval_b_interval"
    r_interval: tuple[int, int]    # (min, max)
    g_interval: tuple[int, int]
    b_interval: tuple[int, int]
    center_color: tuple[int, int, int]  # 代表性中心颜色 (R, G, B)
    pixel_count: int
    percentage: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert numpy types to Python native types for JSON serialization
        d['r_interval'] = tuple(int(x) for x in self.r_interval)
        d['g_interval'] = tuple(int(x) for x in self.g_interval)
        d['b_interval'] = tuple(int(x) for x in self.b_interval)
        d['center_color'] = tuple(int(x) for x in self.center_color)
        d['pixel_count'] = int(self.pixel_count)
        d['percentage'] = float(self.percentage)
        return d


class ColorSeparator:
    """颜色分层器：使用RGB三维区间组合进行颜色分析"""

    def __init__(self, interval_size: int = 20, min_pixel_count: int = 100, device: str = 'cpu'):
        """
        初始化颜色分层器

        Args:
            interval_size: RGB每个通道的区间大小 (默认20)
            min_pixel_count: 最小像素数阈值，低于此值不生成mask (默认100)
            device: 计算设备，支持 'cpu', 'cuda', 'auto' (默认 'cpu')
        """
        if interval_size <= 0 or interval_size > 256:
            raise ValueError(f"interval_size must be between 1 and 256, got {interval_size}")
        if min_pixel_count < 0:
            raise ValueError(f"min_pixel_count must be >= 0, got {min_pixel_count}")

        self.interval_size = interval_size
        self.min_pixel_count = min_pixel_count

        # 设置设备
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda' and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA not available, falling back to CPU")
            device = 'cpu'

        self.device = torch.device(device)
        self.console = Console()

    def _get_interval_index(self, value: int) -> int:
        """获取颜色值所在的区间索引"""
        return min(value // self.interval_size, 255 // self.interval_size)

    def _get_interval_range(self, index: int) -> tuple[int, int]:
        """获取区间索引对应的实际范围"""
        start = index * self.interval_size
        end = min(start + self.interval_size - 1, 255)
        return (start, end)

    def _get_center_color(self, r_idx: int, g_idx: int, b_idx: int) -> tuple[int, int, int]:
        """获取区间组合的中心颜色值"""
        r_range = self._get_interval_range(r_idx)
        g_range = self._get_interval_range(g_idx)
        b_range = self._get_interval_range(b_idx)

        r_center = (r_range[0] + r_range[1]) // 2
        g_center = (g_range[0] + g_range[1]) // 2
        b_center = (b_range[0] + b_range[1]) // 2

        return (r_center, g_center, b_center)

    def preprocess_image(self, image: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, dict[str, Any]]:
        """
        图像预处理：保持宽高比缩放到目标尺寸，短边用0填充
        使用 PyTorch 实现，支持 CPU/GPU 加速

        Args:
            image: 输入图像 (H, W, C)
            target_size: 目标尺寸 (默认640)

        Returns:
            (预处理后的图像, 预处理信息字典)
        """
        original_height, original_width = image.shape[:2]

        # numpy -> tensor
        image_tensor = torch.from_numpy(image).float()  # (H, W, 3)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        image_tensor = image_tensor.to(self.device)

        # 计算缩放比例
        scale = target_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # PyTorch 缩放
        resized = F.interpolate(
            image_tensor,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )

        # 创建画布并填充
        canvas = torch.zeros(1, 3, target_size, target_size, device=self.device)
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        canvas[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # tensor -> numpy
        canvas = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # 预处理信息
        preprocess_info = {
            "original_size": (original_width, original_height),
            "target_size": target_size,
            "scale_factor": round(scale, 4),
            "resized_size": (new_width, new_height),
            "padding": {
                "top": y_offset,
                "bottom": target_size - new_height - y_offset,
                "left": x_offset,
                "right": target_size - new_width - x_offset
            }
        }

        return canvas, preprocess_info

    def analyze_color_groups(self, image: np.ndarray) -> list[ColorGroup]:
        """
        分析图像中的颜色组合
        使用 PyTorch 实现，支持 CPU/GPU 加速

        算法流程：
        1. 将每个像素的RGB值映射到对应的区间索引
        2. 按区间索引组合进行分组统计
        3. 计算每个组的中心颜色和像素占比
        4. 过滤掉像素数低于阈值的组

        Args:
            image: 输入图像 (H, W, 3)

        Returns:
            ColorGroup列表，按像素数降序排序
        """
        height, width = image.shape[:2]
        total_pixels = height * width

        # numpy -> tensor
        image_tensor = torch.from_numpy(image).long().to(self.device)  # (H, W, 3)

        # Reshape to (N, 3)
        pixels = image_tensor.reshape(-1, 3)  # (N, 3)

        # 计算区间索引
        interval_size = self.interval_size
        max_idx = 255 // interval_size

        r_indices = torch.div(pixels[:, 0], interval_size, rounding_mode='floor')
        g_indices = torch.div(pixels[:, 1], interval_size, rounding_mode='floor')
        b_indices = torch.div(pixels[:, 2], interval_size, rounding_mode='floor')

        # Clamp
        r_indices = torch.minimum(r_indices, torch.tensor(max_idx, device=self.device))
        g_indices = torch.minimum(g_indices, torch.tensor(max_idx, device=self.device))
        b_indices = torch.minimum(b_indices, torch.tensor(max_idx, device=self.device))

        # 编码
        group_keys = r_indices * 10000 + g_indices * 100 + b_indices

        # 统计
        unique_groups, counts = torch.unique(group_keys, return_counts=True)

        # tensor -> numpy
        unique_groups = unique_groups.cpu().numpy()
        counts = counts.cpu().numpy()

        # 构建 ColorGroup
        color_groups = []
        for group_key, count in zip(unique_groups, counts):
            if count < self.min_pixel_count:
                continue

            r_idx = group_key // 10000
            g_idx = (group_key % 10000) // 100
            b_idx = group_key % 100

            r_interval = self._get_interval_range(r_idx)
            g_interval = self._get_interval_range(g_idx)
            b_interval = self._get_interval_range(b_idx)
            center_color = self._get_center_color(r_idx, g_idx, b_idx)

            group_id = f"{r_idx}_{g_idx}_{b_idx}"

            color_groups.append(ColorGroup(
                group_id=group_id,
                r_interval=r_interval,
                g_interval=g_interval,
                b_interval=b_interval,
                center_color=center_color,
                pixel_count=int(count),
                percentage=count / total_pixels
            ))

        color_groups.sort(key=lambda g: g.pixel_count, reverse=True)
        return color_groups

    def create_binary_masks(
        self,
        image: np.ndarray,
        color_groups: list[ColorGroup],
        preprocess_info: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """
        为每个颜色组合创建二值化掩码
        使用 PyTorch 实现，支持 CPU/GPU 加速

        Args:
            image: 输入图像
            color_groups: 颜色组合列表
            preprocess_info: 预处理信息（用于排除填充区域）

        Returns:
            字典: {group_id: binary_mask}
        """
        masks = {}

        # numpy -> tensor (一次性)
        image_tensor = torch.from_numpy(image).long().to(self.device)  # (H, W, 3)

        for group in color_groups:
            r_min, r_max = group.r_interval
            g_min, g_max = group.g_interval
            b_min, b_max = group.b_interval

            # 向量化比较
            mask = (
                (image_tensor[:, :, 0] >= r_min) & (image_tensor[:, :, 0] <= r_max) &
                (image_tensor[:, :, 1] >= g_min) & (image_tensor[:, :, 1] <= g_max) &
                (image_tensor[:, :, 2] >= b_min) & (image_tensor[:, :, 2] <= b_max)
            )

            # tensor -> numpy (返回 0/1，用于后续算法处理)
            mask_uint8 = mask.byte().cpu().numpy().astype(np.uint8)
            masks[group.group_id] = mask_uint8

        # 处理填充区域
        if preprocess_info is not None:
            padding = preprocess_info["padding"]
            target_size = preprocess_info["target_size"]

            pad_top = padding["top"]
            pad_bottom = padding["bottom"]
            pad_left = padding["left"]
            pad_right = padding["right"]

            valid_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            valid_mask[pad_top:target_size - pad_bottom, pad_left:target_size - pad_right] = 255

            for group_id in masks:
                masks[group_id] = masks[group_id] & valid_mask

        return masks

    def separate_colors(self, image_path: str) -> dict[str, Any]:
        """
        完整的颜色分离流程

        Args:
            image_path: 输入图像路径

        Returns:
            结果字典，包含:
                - 'color_groups': list[ColorGroup]
                - 'masks': dict[str, np.ndarray]
                - 'metadata': dict
        """
        start_time = time.time()

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB (OpenCV uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理：标准化为640x640
        image, preprocess_info = self.preprocess_image(image, target_size=640)

        height, width = image.shape[:2]

        # Analyze color groups
        color_groups = self.analyze_color_groups(image)

        # Create binary masks
        masks = self.create_binary_masks(image, color_groups, preprocess_info)

        processing_time = time.time() - start_time

        # Build metadata
        metadata = {
            "image_info": {
                "path": str(image_path),
                "original_size": preprocess_info["original_size"],
                "processed_size": (width, height),
                "total_pixels": height * width
            },
            "preprocessing": preprocess_info,
            "parameters": {
                "interval_size": self.interval_size,
                "min_pixel_count": self.min_pixel_count,
                "device": str(self.device)
            },
            "color_groups": [g.to_dict() for g in color_groups],
            "statistics": {
                "total_groups_found": len(color_groups),
                "masks_created": len(masks),
                "processing_time_seconds": round(processing_time, 3)
            }
        }

        return {
            "color_groups": color_groups,
            "masks": masks,
            "metadata": metadata,
            "original_image": image  # 保存原始图像用于输出
        }

    def save_masks(
        self,
        results: dict[str, Any],
        output_dir: str | Path
    ) -> list[Path]:
        """
        保存二值化掩码到磁盘

        Args:
            results: separate_colors() 返回的结果
            output_dir: 输出目录路径

        Returns:
            保存的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Save original image
        original_image = results.get("original_image")
        if original_image is not None:
            # Convert RGB back to BGR for OpenCV saving
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            original_path = output_path / "original.png"
            cv2.imwrite(str(original_path), original_bgr)
            saved_files.append(original_path)

        # Save each mask
        for group in results["color_groups"]:
            mask = results["masks"][group.group_id]

            # Filename: mask_r{idx}_g{idx}_b{idx}.png
            # group_id format: "r_idx_g_idx_b_idx" e.g., "0_0_12"
            parts = group.group_id.split('_')
            mask_filename = f"mask_r{parts[0]}_g{parts[1]}_b{parts[2]}.png"
            mask_path = output_path / mask_filename

            # Save mask (将 0/1 转换为 0/255 用于可视化)
            mask_255 = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_255)
            saved_files.append(mask_path)

            # Update metadata with mask path
            group_dict = next(
                (g for g in results["metadata"]["color_groups"] if g["group_id"] == group.group_id),
                None
            )
            if group_dict:
                group_dict["mask_path"] = mask_filename

        # Save metadata as JSON
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(results["metadata"], f, indent=2)
        saved_files.append(metadata_path)

        # Save human-readable summary
        summary_path = output_path / "summary.txt"
        self._save_summary(results, summary_path)
        saved_files.append(summary_path)

        return saved_files

    def _save_summary(self, results: dict[str, Any], output_path: Path):
        """保存人类可读的摘要文件"""
        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Color Separation Summary\n")
            f.write("=" * 60 + "\n\n")

            # Image info
            img_info = results["metadata"]["image_info"]
            preprocess = results["metadata"]["preprocessing"]
            f.write(f"Image: {img_info['path']}\n")
            f.write(f"Original Size: {img_info['original_size'][0]}x{img_info['original_size'][1]}\n")
            f.write(f"Processed Size: {img_info['processed_size'][0]}x{img_info['processed_size'][1]} ({img_info['total_pixels']} pixels)\n")
            f.write(f"Scale Factor: {preprocess['scale_factor']}\n")
            f.write(f"Padding: top={preprocess['padding']['top']}, bottom={preprocess['padding']['bottom']}, "
                   f"left={preprocess['padding']['left']}, right={preprocess['padding']['right']}\n\n")

            # Parameters
            params = results["metadata"]["parameters"]
            f.write(f"Parameters:\n")
            f.write(f"  Interval Size: {params['interval_size']}\n")
            f.write(f"  Min Pixel Count: {params['min_pixel_count']}\n\n")

            # Statistics
            stats = results["metadata"]["statistics"]
            f.write(f"Statistics:\n")
            f.write(f"  Total Groups Found: {stats['total_groups_found']}\n")
            f.write(f"  Masks Created: {stats['masks_created']}\n")
            f.write(f"  Processing Time: {stats['processing_time_seconds']}s\n\n")

            # Color groups table
            f.write("Color Groups (sorted by pixel count):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Group ID':<15} {'RGB Interval':<30} {'Pixels':<10} {'%':<8}\n")
            f.write("-" * 60 + "\n")

            for group in results["color_groups"]:
                r_range = f"R{group.r_interval[0]}-{group.r_interval[1]}"
                g_range = f"G{group.g_interval[0]}-{group.g_interval[1]}"
                b_range = f"B{group.b_interval[0]}-{group.b_interval[1]}"
                interval_str = f"{r_range}, {g_range}, {b_range}"

                f.write(f"{group.group_id:<15} {interval_str:<30} {group.pixel_count:<10} {group.percentage*100:>6.2f}%\n")

    def print_results(self, results: dict[str, Any]):
        """使用 Rich 打印结果表格"""
        table = Table(title="Detected Color Groups")
        table.add_column("Group ID", style="cyan")
        table.add_column("RGB Interval", style="magenta")
        table.add_column("Center Color", style="yellow")
        table.add_column("Pixels", justify="right")
        table.add_column("Percentage", justify="right")

        for group in results["color_groups"]:
            r_range = f"R{group.r_interval[0]}-{group.r_interval[1]}"
            g_range = f"G{group.g_interval[0]}-{group.g_interval[1]}"
            b_range = f"B{group.b_interval[0]}-{group.b_interval[1]}"
            interval_str = f"{r_range} {g_range} {b_range}"

            center_str = f"({group.center_color[0]}, {group.center_color[1]}, {group.center_color[2]})"

            table.add_row(
                group.group_id,
                interval_str,
                center_str,
                str(group.pixel_count),
                f"{group.percentage * 100:.2f}%"
            )

        self.console.print(table)

        stats = results["metadata"]["statistics"]
        self.console.print(f"\nTotal groups: {stats['total_groups_found']}, "
                          f"Masks created: {stats['masks_created']}, "
                          f"Time: {stats['processing_time_seconds']}s")
