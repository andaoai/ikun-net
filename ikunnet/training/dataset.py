"""Dataset for mask embedding training."""

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MaskDataset(Dataset):
    """
    Dataset for loading binary masks for contrastive learning.

    Expects masks stored as individual PNG files in a directory.
    """

    def __init__(
        self,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
        two_view_transform: Optional[Callable] = None
    ):
        """
        Args:
            mask_dir: Directory containing mask PNG files
            transform: Optional transform for single view (validation/test)
            two_view_transform: Optional transform for two views (training)
        """
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.two_view_transform = two_view_transform

        # Find all mask files
        self.mask_files = sorted(self.mask_dir.glob("*.png"))
        if not self.mask_files:
            raise ValueError(f"No PNG files found in {mask_dir}")

        # Load metadata if exists
        metadata_file = self.mask_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def __len__(self) -> int:
        return len(self.mask_files)

    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single mask.

        Args:
            idx: Index

        Returns:
            If two_view_transform: (view1, view2) - two augmented views
            Else: mask - single (possibly transformed) mask
        """
        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path)

        # Convert to numpy then tensor
        mask_np = np.array(mask, dtype=np.uint8)

        # Convert to tensor (1, H, W) and scale to [0, 1]
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0

        # Ensure 3D (1, H, W)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Apply transforms
        if self.two_view_transform is not None:
            # Training mode: return two views
            view1, view2 = self.two_view_transform(mask_tensor)
            return view1, view2
        elif self.transform is not None:
            # Validation/test mode: return single transformed view
            return self.transform(mask_tensor)
        else:
            # No transform: return as-is
            return mask_tensor

    def get_mask_info(self, idx: int) -> dict:
        """
        Get metadata for a specific mask.

        Args:
            idx: Index

        Returns:
            Metadata dict with keys like 'group_id', 'color_info', etc.
        """
        mask_filename = self.mask_files[idx].name

        if self.metadata:
            # Find matching metadata by filename
            for item in self.metadata.get('masks', []):
                if item.get('filename') == mask_filename:
                    return item

        # Extract group_id from filename
        # Format: mask_r{idx}_g{idx}_b{idx}.png
        group_id = mask_filename.replace('mask_', '').replace('.png', '')

        return {
            'filename': mask_filename,
            'group_id': group_id,
            'path': str(self.mask_files[idx])
        }


class MaskDatasetWithPath(MaskDataset):
    """
    Extended MaskDataset that also returns the file path.

    Useful for tracking which mask corresponds to which embedding.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single mask with its path.

        Args:
            idx: Index

        Returns:
            (mask, path): Mask tensor and file path
        """
        mask_path = self.mask_files[idx]

        # Load mask
        mask = Image.open(mask_path)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if self.transform:
            mask_tensor = self.transform(mask_tensor)

        return mask_tensor, str(mask_path)
