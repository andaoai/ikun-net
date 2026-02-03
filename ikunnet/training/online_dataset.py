"""Online mask extraction dataset for contrastive learning."""

from pathlib import Path
from torch.utils.data import Dataset

from ikunnet import ColorSeparator


class ImageMaskDataset(Dataset):
    """
    在线从 ImageNet 图片提取 masks 的 Dataset。

    每个 item 返回一组图片路径，由 collate_fn 处理提取。
    """

    def __init__(
        self,
        image_dir: str,
        images_per_batch: int = 4,
        interval_size: int = 40,
        min_pixel_count: int = 100,
        device: str = 'cuda'
    ):
        """
        Args:
            image_dir: ImageNet dataset directory
            images_per_batch: Number of images to process per batch
            interval_size: Color interval size for ColorSeparator
            min_pixel_count: Minimum pixel count for color groups
            device: Device for ColorSeparator ('cpu' or 'cuda')
        """
        self.image_files = self._find_images(image_dir)
        self.images_per_batch = images_per_batch
        self.separator_kwargs = {
            'interval_size': interval_size,
            'min_pixel_count': min_pixel_count,
            'device': device
        }

    def _find_images(self, image_dir: str):
        """查找所有图片文件"""
        image_dir = Path(image_dir)
        image_files = []
        for ext in ['*.JPEG', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(image_dir.rglob(ext))
        return image_files

    def __len__(self):
        return len(self.image_files) // self.images_per_batch

    def __getitem__(self, idx):
        """
        返回一个 batch 的图片路径

        Returns:
            List of image paths (length = images_per_batch)
        """
        start_idx = idx * self.images_per_batch
        image_paths = self.image_files[start_idx:start_idx + self.images_per_batch]
        return [str(p) for p in image_paths]

    def get_separator(self):
        """返回 ColorSeparator 实例供 collate_fn 使用"""
        return ColorSeparator(**self.separator_kwargs)
