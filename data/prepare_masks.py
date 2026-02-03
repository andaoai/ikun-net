"""Prepare mask dataset from ColorSeparator output."""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rich.console import Console
from rich.progress import track

from ikunnet import ColorSeparator


def prepare_masks_from_imagenet(
    imagenet_path: str,
    output_dir: str,
    interval_size: int = 40,
    min_pixel_count: int = 100,
    device: str = 'cpu',
    max_images: Optional[int] = None,
    mask_count_limit: Optional[int] = None
):
    """
    Extract masks from ImageNet dataset using ColorSeparator.

    Args:
        imagenet_path: Path to ImageNet dataset
        output_dir: Output directory for masks
        interval_size: Color interval size
        min_pixel_count: Minimum pixel count for a color group
        device: 'cpu' or 'cuda'
        max_images: Maximum number of images to process
        mask_count_limit: Maximum number of masks to extract per image
    """
    console = Console()

    output_path = Path(output_dir)
    mask_dir = output_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Create color separator
    separator = ColorSeparator(
        interval_size=interval_size,
        min_pixel_count=min_pixel_count,
        device=device
    )

    # Find all images in ImageNet
    imagenet_path = Path(imagenet_path)
    image_files = []
    for ext in ['*.JPEG', '*.jpg', '*.jpeg', '*.png']:
        image_files.extend(imagenet_path.rglob(ext))

    if max_images:
        image_files = image_files[:max_images]

    console.print(f"[cyan]Found {len(image_files)} images[/cyan]")
    console.print(f"[cyan]Output directory: {mask_dir}[/cyan]\n")

    # Process each image
    metadata = {
        'parameters': {
            'interval_size': interval_size,
            'min_pixel_count': min_pixel_count,
            'device': device
        },
        'masks': []
    }

    total_masks = 0

    for img_path in track(image_files, description="Extracting masks"):
        try:
            # Process image
            results = separator.separate_colors(img_path)

            # Extract masks
            masks = results['masks']
            color_groups = results['color_groups']

            # Save each mask
            for group in color_groups:
                if mask_count_limit and total_masks >= mask_count_limit:
                    break

                group_id = group.group_id
                mask = masks[group_id]

                # Save mask as PNG (values are 0/1, save as 0-255 for visualization)
                mask_255 = (mask * 255).astype(np.uint8)
                mask_filename = f"{group_id}_{total_masks:06d}.png"
                mask_path = mask_dir / mask_filename

                cv2.imwrite(str(mask_path), mask_255)

                # Record metadata (use to_dict() method)
                group_dict = group.to_dict()
                group_dict['filename'] = mask_filename
                group_dict['source_image'] = str(img_path)
                metadata['masks'].append(group_dict)

                total_masks += 1

            if mask_count_limit and total_masks >= mask_count_limit:
                break

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to process {img_path}: {e}[/yellow]")

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[green]Extraction complete![/green]")
    console.print(f"Total masks extracted: {total_masks}")
    console.print(f"Masks saved to: {mask_dir}")
    console.print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare mask dataset from ImageNet")
    parser.add_argument('--imagenet-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='data/masks_dataset', help='Output directory')
    parser.add_argument('--interval-size', type=int, default=40, help='Color interval size')
    parser.add_argument('--min-pixels', type=int, default=100, help='Minimum pixel count')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process')
    parser.add_argument('--mask-limit', type=int, default=None, help='Max masks to extract total')

    args = parser.parse_args()

    prepare_masks_from_imagenet(
        imagenet_path=args.imagenet_path,
        output_dir=args.output_dir,
        interval_size=args.interval_size,
        min_pixel_count=args.min_pixels,
        device=args.device,
        max_images=args.max_images,
        mask_count_limit=args.mask_limit
    )


if __name__ == '__main__':
    main()
