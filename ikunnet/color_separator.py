"""
Color separator module for analyzing images and generating binary color masks.

This module implements RGB 3D interval combination for color layering.
"""

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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

    def __init__(self, interval_size: int = 20, min_pixel_count: int = 100):
        """
        初始化颜色分层器

        Args:
            interval_size: RGB每个通道的区间大小 (默认20)
            min_pixel_count: 最小像素数阈值，低于此值不生成mask (默认100)
        """
        if interval_size <= 0 or interval_size > 256:
            raise ValueError(f"interval_size must be between 1 and 256, got {interval_size}")
        if min_pixel_count < 0:
            raise ValueError(f"min_pixel_count must be >= 0, got {min_pixel_count}")

        self.interval_size = interval_size
        self.min_pixel_count = min_pixel_count
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

    def analyze_color_groups(self, image: np.ndarray) -> list[ColorGroup]:
        """
        分析图像中的颜色组合

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

        # Reshape to (N, 3) for efficient processing
        pixels = image.reshape(-1, 3)

        # Calculate interval indices for each channel
        r_indices = pixels[:, 0] // self.interval_size
        g_indices = pixels[:, 1] // self.interval_size
        b_indices = pixels[:, 2] // self.interval_size

        # Clamp max index to handle edge case (value 255)
        max_idx = 255 // self.interval_size
        r_indices = np.minimum(r_indices, max_idx).astype(np.int32)
        g_indices = np.minimum(g_indices, max_idx).astype(np.int32)
        b_indices = np.minimum(b_indices, max_idx).astype(np.int32)

        # Create unique group keys
        # Use encoding: r_idx * 10000 + g_idx * 100 + b_idx
        group_keys = r_indices * 10000 + g_indices * 100 + b_indices

        # Count unique groups
        unique_groups, counts = np.unique(group_keys, return_counts=True)

        # Build ColorGroup objects
        color_groups = []

        for group_key, count in zip(unique_groups, counts):
            # Filter by minimum pixel count
            if count < self.min_pixel_count:
                continue

            # Decode group key
            r_idx = group_key // 10000
            g_idx = (group_key % 10000) // 100
            b_idx = group_key % 100

            # Get intervals and center color
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

        # Sort by pixel count (descending)
        color_groups.sort(key=lambda g: g.pixel_count, reverse=True)

        return color_groups

    def create_binary_masks(
        self,
        image: np.ndarray,
        color_groups: list[ColorGroup]
    ) -> dict[str, np.ndarray]:
        """
        为每个颜色组合创建二值化掩码

        Args:
            image: 输入图像
            color_groups: 颜色组合列表

        Returns:
            字典: {group_id: binary_mask}
        """
        masks = {}

        for group in color_groups:
            r_min, r_max = group.r_interval
            g_min, g_max = group.g_interval
            b_min, b_max = group.b_interval

            # Create boolean mask for pixels in this color range
            mask = (
                (image[:, :, 0] >= r_min) & (image[:, :, 0] <= r_max) &
                (image[:, :, 1] >= g_min) & (image[:, :, 1] <= g_max) &
                (image[:, :, 2] >= b_min) & (image[:, :, 2] <= b_max)
            )

            # Convert boolean to 0/255
            masks[group.group_id] = mask.astype(np.uint8) * 255

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

        height, width = image.shape[:2]

        # Analyze color groups
        color_groups = self.analyze_color_groups(image)

        # Create binary masks
        masks = self.create_binary_masks(image, color_groups)

        processing_time = time.time() - start_time

        # Build metadata
        metadata = {
            "image_info": {
                "path": str(image_path),
                "width": width,
                "height": height,
                "total_pixels": height * width
            },
            "parameters": {
                "interval_size": self.interval_size,
                "min_pixel_count": self.min_pixel_count
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

            # Save mask (OpenCV expects grayscale)
            cv2.imwrite(str(mask_path), mask)
            saved_files.append(mask_path)

            # Update metadata with mask path
            group_dict = next(
                (g for g in results["metadata"]["color_groups"] if g["group_id"] == group.group_id),
                None
            )
            if group_dict:
                group_dict["mask_path"] = mask_filename

        # Save metadata as JSON
        import json
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
            f.write(f"Image: {img_info['path']}\n")
            f.write(f"Size: {img_info['width']}x{img_info['height']} ({img_info['total_pixels']} pixels)\n\n")

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
