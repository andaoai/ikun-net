"""Data transforms for binary masks (no augmentation)."""

import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import resize


class SimpleTwoViewTransform:
    """
    Create two identical views without augmentation.

    Returns two copies of the same mask for contrastive learning.
    Only resizes to a fixed size.
    """

    def __init__(self, image_size: int = 224):
        """
        Args:
            image_size: Target size after resize
        """
        self.image_size = image_size

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create two identical views (no augmentation).

        Args:
            x: Input tensor (H, W) or (C, H, W)

        Returns:
            view1, view2: Two identical views (C, H, W) each
        """
        # Ensure 3D input (1, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, H, W)

        # Resize to fixed size
        x_resized = resize(x, [self.image_size, self.image_size], antialias=True)

        # Return two identical copies
        return x_resized.clone(), x_resized.clone()


class MaskTransform:
    """
    Basic transform without augmentation (for validation/test).
    """

    def __init__(self, image_size: int = 224):
        """
        Args:
            image_size: Target size
        """
        self.transforms = v2.Compose([
            v2.Resize(image_size, antialias=True),
            v2.CenterCrop(image_size),
            v2.ToTensor(),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transform.

        Args:
            x: Input tensor (H, W) or (C, H, W)

        Returns:
            Transformed tensor (C, H, W)
        """
        return self.transforms(x)
