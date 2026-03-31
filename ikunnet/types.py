"""Type definitions for ikunnet module."""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


@dataclass
class ColorGroup:
    """颜色分组信息"""
    group_id: str
    interval: Tuple[int, int]       # 区间 [low, high]
    center_value: int               # 区间中心值
    pixel_count: int                # 像素数量
    percentage: float               # 占比百分比
    mask_path: Optional[str] = None # mask 文件路径

    def to_dict(self) -> Dict[str, Any]:
        return {
            'group_id': self.group_id,
            'interval': list(self.interval),
            'center_value': self.center_value,
            'pixel_count': self.pixel_count,
            'percentage': round(self.percentage, 4),
            'mask_path': self.mask_path
        }


@dataclass
class ColorCombination:
    """H-S-V 三通道组合颜色分组信息"""
    group_id: str                           # 格式: "h0-s0-v0"
    h_interval: Tuple[int, int]             # H 区间 [low, high]
    s_interval: Tuple[int, int]             # S 区间 [low, high]
    v_interval: Tuple[int, int]             # V 区间 [low, high]
    h_center: int                           # H 区间中心值
    s_center: int                           # S 区间中心值
    v_center: int                           # V 区间中心值
    pixel_count: int                        # 像素数量
    percentage: float                       # 占比百分比
    mask_path: Optional[str] = None         # mask 文件路径

    def to_dict(self) -> Dict[str, Any]:
        return {
            'group_id': self.group_id,
            'h_interval': list(self.h_interval),
            's_interval': list(self.s_interval),
            'v_interval': list(self.v_interval),
            'h_center': self.h_center,
            's_center': self.s_center,
            'v_center': self.v_center,
            'pixel_count': self.pixel_count,
            'percentage': round(self.percentage, 4),
            'mask_path': self.mask_path
        }


@dataclass
class HSVMaskResult:
    """HSV 分离结果"""
    original_image: np.ndarray
    h_mask: np.ndarray              # H 通道分类 mask
    s_mask: np.ndarray              # S 通道分类 mask
    v_mask: np.ndarray              # V 通道分类 mask
    h_groups: List[ColorGroup] = field(default_factory=list)
    s_groups: List[ColorGroup] = field(default_factory=list)
    v_groups: List[ColorGroup] = field(default_factory=list)
    image_path: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'processing_time': round(self.processing_time, 3),
            'h_channel': {
                'groups': [g.to_dict() for g in self.h_groups]
            },
            's_channel': {
                'groups': [g.to_dict() for g in self.s_groups]
            },
            'v_channel': {
                'groups': [g.to_dict() for g in self.v_groups]
            }
        }


@dataclass
class HSVCombinedResult:
    """H-S-V 组合编码分离结果"""
    original_image: np.ndarray
    h_mask: np.ndarray                              # H 通道分类 mask (索引)
    s_mask: np.ndarray                              # S 通道分类 mask (索引)
    v_mask: np.ndarray                              # V 通道分类 mask (索引)
    combined_mask: np.ndarray                       # 组合 mask (扁平化索引)
    groups: List[ColorCombination] = field(default_factory=list)
    image_path: Optional[str] = None
    processing_time: float = 0.0
    h_bounds: List[Tuple[int, int]] = field(default_factory=list)
    s_bounds: List[Tuple[int, int]] = field(default_factory=list)
    v_bounds: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'processing_time': round(self.processing_time, 3),
            'total_groups': len(self.groups),
            'groups': [g.to_dict() for g in self.groups]
        }
