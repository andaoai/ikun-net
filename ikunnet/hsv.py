"""HSV Color Separator implementation."""

import time
from pathlib import Path
from typing import Union, Optional, List, Tuple

import cv2
import numpy as np

from .types import ColorGroup, HSVMaskResult, ColorCombination, HSVCombinedResult
from .config import HSVClassifierConfig


class HSVSeparator:
    """HSV 颜色空间分离器

    对 H、S、V 三个通道分别进行区间划分和分类。

    HSV 范围（OpenCV）:
        H (色调): 0-180 (对应 0-360 度)
        S (饱和度): 0-255
        V (明度): 0-255
    """

    def __init__(self, config: Optional[HSVClassifierConfig] = None):
        self.config = config or HSVClassifierConfig()

        # 预计算区间边界
        self._h_bounds = self._compute_bounds(
            180, self.config.h_intervals, self.config.h_custom_bounds
        )
        self._s_bounds = self._compute_bounds(
            256, self.config.s_intervals, self.config.s_custom_bounds
        )
        self._v_bounds = self._compute_bounds(
            256, self.config.v_intervals, self.config.v_custom_bounds
        )

    def _compute_bounds(
        self,
        max_val: int,
        n_intervals: int,
        custom_bounds: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """计算区间边界"""
        if custom_bounds:
            return custom_bounds

        interval_size = max_val / n_intervals
        bounds = []
        for i in range(n_intervals):
            low = int(i * interval_size)
            high = int((i + 1) * interval_size) - 1 if i < n_intervals - 1 else max_val - 1
            bounds.append((low, high))
        return bounds

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """加载图片，返回 RGB 格式 numpy 数组

        Args:
            image_path: 图片路径

        Returns:
            RGB 格式图片数组 (H, W, 3)
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        # BGR -> RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def rgb_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """将 RGB 图片转换为 HSV

        Args:
            image: RGB 格式图片 (H, W, 3)

        Returns:
            HSV 格式图片 (H, W, 3)
        """
        # OpenCV 需要 BGR 输入
        bgr = image[..., ::-1]  # RGB -> BGR
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv

    def classify_channel(
        self,
        channel: np.ndarray,
        bounds: List[Tuple[int, int]],
        channel_name: str
    ) -> Tuple[np.ndarray, List[ColorGroup]]:
        """对单个通道进行分类

        Args:
            channel: 单通道图像 (H, W)
            bounds: 区间边界列表 [(low, high), ...]
            channel_name: 通道名称 ('h', 's', 'v')

        Returns:
            mask: 分类结果，每个像素值为其所属区间索引
            groups: 颜色分组信息列表
        """
        h, w = channel.shape
        total_pixels = h * w

        # 初始化 mask，使用 -1 表示未分类
        mask = np.full_like(channel, -1, dtype=np.int32)
        groups = []

        for idx, (low, high) in enumerate(bounds):
            # 找到属于该区间的像素
            in_interval = (channel >= low) & (channel <= high)
            pixel_count = int(in_interval.sum())

            if pixel_count >= self.config.min_pixel_count:
                mask[in_interval] = idx
                center = (low + high) // 2
                group = ColorGroup(
                    group_id=f"{channel_name}{idx}",
                    interval=(low, high),
                    center_value=center,
                    pixel_count=pixel_count,
                    percentage=pixel_count / total_pixels
                )
                groups.append(group)

        return mask, groups

    def separate(self, image: np.ndarray, image_path: Optional[str] = None) -> HSVMaskResult:
        """执行 HSV 颜色分离

        Args:
            image: RGB 格式图片 (H, W, 3)
            image_path: 图片路径（可选，用于记录）

        Returns:
            HSVMaskResult: 包含三个通道的 mask 和分组信息
        """
        start_time = time.time()

        # 转换为 HSV
        hsv = self.rgb_to_hsv(image)

        # 分别处理 H、S、V 通道
        h_mask, h_groups = self.classify_channel(
            hsv[:, :, 0], self._h_bounds, 'h'
        )
        s_mask, s_groups = self.classify_channel(
            hsv[:, :, 1], self._s_bounds, 's'
        )
        v_mask, v_groups = self.classify_channel(
            hsv[:, :, 2], self._v_bounds, 'v'
        )

        processing_time = time.time() - start_time

        return HSVMaskResult(
            original_image=image,
            h_mask=h_mask,
            s_mask=s_mask,
            v_mask=v_mask,
            h_groups=h_groups,
            s_groups=s_groups,
            v_groups=v_groups,
            image_path=image_path,
            processing_time=processing_time
        )

    def separate_from_file(self, image_path: Union[str, Path]) -> HSVMaskResult:
        """从文件加载图片并执行 HSV 颜色分离

        Args:
            image_path: 图片路径

        Returns:
            HSVMaskResult: 包含三个通道的 mask 和分组信息
        """
        image_path = str(image_path)
        image = self.load_image(image_path)
        return self.separate(image, image_path=image_path)

    def separate_combined(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None
    ) -> HSVCombinedResult:
        """执行 H-S-V 三通道组合编码颜色分离

        每个像素被分配到一个唯一的组合 (h_idx, s_idx, v_idx)，
        生成组合 mask 和 ColorCombination 列表。

        Args:
            image: RGB 格式图片 (H, W, 3)
            image_path: 图片路径（可选，用于记录）

        Returns:
            HSVCombinedResult: 包含组合 mask 和分组信息
        """
        start_time = time.time()

        # 转换为 HSV
        hsv = self.rgb_to_hsv(image)

        # 分别对 H、S、V 进行分类
        h_mask, _ = self.classify_channel(hsv[:, :, 0], self._h_bounds, 'h')
        s_mask, _ = self.classify_channel(hsv[:, :, 1], self._s_bounds, 's')
        v_mask, _ = self.classify_channel(hsv[:, :, 2], self._v_bounds, 'v')

        h, w = h_mask.shape
        total_pixels = h * w

        # 计算组合索引: combined_idx = h_idx * (s_intervals * v_intervals) + s_idx * v_intervals + v_idx
        s_v_product = self.config.s_intervals * self.config.v_intervals
        combined_mask = np.full((h, w), -1, dtype=np.int32)

        groups = []
        group_idx = 0

        # 遍历所有可能的组合
        for h_idx in range(self.config.h_intervals):
            for s_idx in range(self.config.s_intervals):
                for v_idx in range(self.config.v_intervals):
                    # 找到同时满足三个条件的像素
                    mask = (h_mask == h_idx) & (s_mask == s_idx) & (v_mask == v_idx)
                    pixel_count = int(mask.sum())

                    if pixel_count >= self.config.min_pixel_count:
                        # 分配组合索引
                        combined_mask[mask] = group_idx

                        # 计算中心值
                        h_low, h_high = self._h_bounds[h_idx]
                        s_low, s_high = self._s_bounds[s_idx]
                        v_low, v_high = self._v_bounds[v_idx]

                        group = ColorCombination(
                            group_id=f"h{h_idx}-s{s_idx}-v{v_idx}",
                            h_interval=(h_low, h_high),
                            s_interval=(s_low, s_high),
                            v_interval=(v_low, v_high),
                            h_center=(h_low + h_high) // 2,
                            s_center=(s_low + s_high) // 2,
                            v_center=(v_low + v_high) // 2,
                            pixel_count=pixel_count,
                            percentage=pixel_count / total_pixels
                        )
                        groups.append(group)
                        group_idx += 1

        processing_time = time.time() - start_time

        return HSVCombinedResult(
            original_image=image,
            h_mask=h_mask,
            s_mask=s_mask,
            v_mask=v_mask,
            combined_mask=combined_mask,
            groups=groups,
            image_path=image_path,
            processing_time=processing_time,
            h_bounds=self._h_bounds,
            s_bounds=self._s_bounds,
            v_bounds=self._v_bounds
        )

    def separate_combined_from_file(self, image_path: Union[str, Path]) -> HSVCombinedResult:
        """从文件加载图片并执行 H-S-V 组合编码颜色分离

        Args:
            image_path: 图片路径

        Returns:
            HSVCombinedResult: 包含组合 mask 和分组信息
        """
        image_path = str(image_path)
        image = self.load_image(image_path)
        return self.separate_combined(image, image_path=image_path)
