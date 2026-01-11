"""
CLI handlers for ikun-net color separator.
"""

import argparse
import random
from pathlib import Path
from rich.console import Console

from ikunnet.color_separator import ColorSeparator


console = Console()


def get_random_image_from_imagenet(dataset_path: str = "data/imagenet1k/imagenet-mini/") -> Path:
    """
    从ImageNet数据集随机选择一张图片

    Args:
        dataset_path: 数据集路径

    Returns:
        随机图片的完整路径

    Raises:
        FileNotFoundError: 如果数据集不存在或没有找到图片
    """
    dataset = Path(dataset_path)

    if not dataset.exists():
        raise FileNotFoundError(
            f"ImageNet dataset not found at: {dataset_path}\n"
            f"Please ensure the dataset exists or specify a custom path with --dataset"
        )

    # 支持的图片格式
    image_extensions = {'.JPEG', '.jpg', '.jpeg', '.png', '.bmp'}

    # 查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset.rglob(f"*{ext}"))
        image_files.extend(dataset.rglob(f"*{ext.upper()}"))

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in dataset: {dataset_path}\n"
            f"Supported formats: {', '.join(image_extensions)}"
        )

    # 随机选择一张图片
    return random.choice(image_files)


def handle_separate_colors(args: argparse.Namespace):
    """
    Handle the separate-colors command.

    Args:
        args: Parsed command-line arguments
    """
    # 处理图片路径：如果未指定，则随机选择
    if args.input is None:
        try:
            input_path = get_random_image_from_imagenet(args.dataset)
            console.print(f"[cyan]Randomly selected image from ImageNet dataset[/cyan]")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1
    else:
        input_path = Path(args.input)

    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        return 1

    if not input_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
        console.print(f"[yellow]Warning: Input may not be an image file: {input_path}[/yellow]")

    # Create color separator
    separator = ColorSeparator(
        interval_size=args.interval_size,
        min_pixel_count=args.min_pixels
    )

    console.print(f"[cyan]Processing image: {input_path}[/cyan]")
    console.print(f"  Interval size: {args.interval_size}")
    console.print(f"  Min pixel count: {args.min_pixels}")
    console.print()

    try:
        # Process image
        results = separator.separate_colors(input_path)

        # Print results
        separator.print_results(results)

        # Save masks if not analyze-only
        if not args.analyze_only:
            output_dir = Path(args.output) / input_path.stem
            saved_files = separator.save_masks(results, output_dir)

            console.print(f"\n[green]Masks saved to: {output_dir}[/green]")
            console.print(f"  Files created: {len(saved_files)}")
        else:
            console.print("\n[yellow]Analyze-only mode: masks not saved[/yellow]")

        return 0

    except Exception as e:
        console.print(f"[red]Error processing image: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return 1


def add_separate_colors_parser(subparsers):
    """
    Add the separate-colors subcommand to the argument parser.

    Args:
        subparsers: The subparsers object from argparse
    """
    parser = subparsers.add_parser(
        'separate-colors',
        help='Separate image into color layers using RGB 3D interval combination'
    )

    parser.add_argument(
        'input',
        nargs='?',
        default=None,
        help='Input image path (optional: if not specified, randomly selects from ImageNet dataset)'
    )

    parser.add_argument(
        '--dataset',
        default='data/imagenet1k/imagenet-mini/',
        help='Path to ImageNet dataset (default: data/imagenet1k/imagenet-mini/)'
    )

    parser.add_argument(
        '--output', '-o',
        default='tmp/color_masks',
        help='Output directory (default: tmp/color_masks)'
    )

    parser.add_argument(
        '--interval-size',
        type=int,
        default=20,
        help='Color interval size for RGB channels (default: 20)'
    )

    parser.add_argument(
        '--min-pixels',
        type=int,
        default=100,
        help='Minimum pixel count to create a mask (default: 100)'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze colors, do not save masks'
    )

    parser.set_defaults(func=handle_separate_colors)
