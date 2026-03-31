"""Configuration for HSV color classifier."""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class HSVClassifierConfig:
    """HSV 分类器配置

    HSV 范围（OpenCV）:
        H (色调): 0-180 (对应 0-360 度)
        S (饱和度): 0-255
        V (明度): 0-255
    """
    # H 通道配置 (0-180)
    h_intervals: int = 24           # H 分为 24 个区间，每个约 7.5 (对应色环约 15°)
    h_custom_bounds: Optional[List[Tuple[int, int]]] = None

    # S 通道配置 (0-255)
    s_intervals: int = 16           # S 分为 16 个区间
    s_custom_bounds: Optional[List[Tuple[int, int]]] = None

    # V 通道配置 (0-255)
    v_intervals: int = 16           # V 分为 16 个区间
    v_custom_bounds: Optional[List[Tuple[int, int]]] = None

    # 通用配置
    min_pixel_count: int = 100      # 最小像素数阈值
    device: str = 'cpu'             # 'cpu' 或 'cuda'

    def to_dict(self) -> dict:
        return {
            'h_intervals': self.h_intervals,
            's_intervals': self.s_intervals,
            'v_intervals': self.v_intervals,
            'min_pixel_count': self.min_pixel_count,
            'device': self.device
        }
