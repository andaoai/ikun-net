"""HSV Color Classification CLI Tool.

从 ImageNet 数据集随机抽取图片，进行 HSV 颜色分类。
使用 H-S-V 三通道组合编码，每个 mask 代表一种完整颜色。
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from ikunnet import HSVSeparator, HSVClassifierConfig


def reconstruct_from_combined(result) -> np.ndarray:
    """从组合 mask 重建图像

    使用每个组合的中心值重建 HSV 图像。

    Args:
        result: HSVCombinedResult

    Returns:
        重建的 RGB 图像
    """
    h, w = result.original_image.shape[:2]
    reconstructed_hsv = np.zeros((h, w, 3), dtype=np.uint8)

    for group in result.groups:
        # 读取 mask
        if group.mask_path:
            mask = cv2.imread(str(Path(result.image_path).parent if result.image_path else '.') / group.mask_path, 0)
            if mask is not None:
                mask = mask > 127
            else:
                continue
        else:
            continue

        # 使用中心值填充
        reconstructed_hsv[mask] = [group.h_center, group.s_center, group.v_center]

    # 转换为 RGB
    reconstructed_rgb = cv2.cvtColor(reconstructed_hsv, cv2.COLOR_HSV2RGB)
    return reconstructed_rgb


def reconstruct_from_combined_mask(result) -> np.ndarray:
    """从内存中的 combined_mask 重建图像（不依赖文件）

    Args:
        result: HSVCombinedResult

    Returns:
        重建的 RGB 图像
    """
    h, w = result.combined_mask.shape
    reconstructed_hsv = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, group in enumerate(result.groups):
        mask = result.combined_mask == idx
        reconstructed_hsv[mask] = [group.h_center, group.s_center, group.v_center]

    # 转换为 RGB
    reconstructed_rgb = cv2.cvtColor(reconstructed_hsv, cv2.COLOR_HSV2RGB)
    return reconstructed_rgb


def calculate_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """计算重建质量指标"""
    # MSE
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)

    # PSNR
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

    # 颜色数统计
    original_colors = len(np.unique(original.reshape(-1, 3), axis=0))
    reconstructed_colors = len(np.unique(reconstructed.reshape(-1, 3), axis=0))

    return {
        'mse': round(mse, 2),
        'psnr': round(psnr, 2),
        'original_colors': original_colors,
        'reconstructed_colors': reconstructed_colors
    }


def save_combined_results(
    result,
    output_dir: Path,
    separator: HSVSeparator
):
    """保存组合编码分类结果

    Args:
        result: HSVCombinedResult
        output_dir: 输出目录
        separator: HSVSeparator 实例
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始图片
    original_path = output_dir / "original.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(result.original_image, cv2.COLOR_RGB2BGR))

    # 创建 masks 子目录
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    # 保存每个组合的 mask
    for idx, group in enumerate(result.groups):
        mask = (result.combined_mask == idx).astype(np.uint8) * 255
        mask_path = masks_dir / f"{group.group_id}.png"
        cv2.imwrite(str(mask_path), mask)
        group.mask_path = f"masks/{group.group_id}.png"

    # 从组合 mask 重建图像
    reconstructed = reconstruct_from_combined_mask(result)

    # 计算质量指标
    quality = calculate_quality_metrics(result.original_image, reconstructed)

    # 保存重建图
    cv2.imwrite(
        str(output_dir / "reconstructed.png"),
        cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
    )

    # 保存对比图（原图 | 重建图）
    comparison = np.hstack([result.original_image, reconstructed])
    cv2.imwrite(
        str(output_dir / "comparison.png"),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )

    # 保存 metadata
    metadata = {
        'image_info': {
            'path': result.image_path,
            'original_size': list(result.original_image.shape[:2]),
            'total_pixels': result.original_image.shape[0] * result.original_image.shape[1]
        },
        'config': separator.config.to_dict(),
        'bounds': {
            'h_bounds': [list(b) for b in result.h_bounds],
            's_bounds': [list(b) for b in result.s_bounds],
            'v_bounds': [list(b) for b in result.v_bounds]
        },
        'reconstruction_quality': quality,
        **result.to_dict()
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata, quality


def print_combined_summary(result, quality: dict, console: Console):
    """打印组合编码分类摘要"""
    console.print(f"\n[bold green]处理完成！[/bold green]")
    console.print(f"图片: {result.image_path}")
    console.print(f"处理时间: {result.processing_time:.3f}s")

    # 重建质量
    console.print(f"\n[bold yellow]重建质量指标[/bold yellow]")
    console.print(f"  MSE: {quality['mse']:.2f}")
    console.print(f"  PSNR: {quality['psnr']:.2f} dB")
    console.print(f"  原图颜色数: {quality['original_colors']:,}")
    console.print(f"  重建颜色数: {quality['reconstructed_colors']:,}")

    # 统计信息
    console.print(f"\n[bold cyan]组合颜色统计[/bold cyan]")
    console.print(f"  有效颜色组合数: {len(result.groups)}")
    console.print(f"  理论最大组合数: {len(result.h_bounds) * len(result.s_bounds) * len(result.v_bounds)}")

    # Top 20 颜色组合
    if result.groups:
        sorted_groups = sorted(result.groups, key=lambda x: x.pixel_count, reverse=True)
        top_n = min(20, len(sorted_groups))

        table = Table(title=f"Top {top_n} 颜色组合")
        table.add_column("Group ID", style="cyan")
        table.add_column("H 区间", style="red")
        table.add_column("S 区间", style="green")
        table.add_column("V 区间", style="blue")
        table.add_column("像素数", justify="right")
        table.add_column("占比", justify="right")

        for g in sorted_groups[:top_n]:
            table.add_row(
                g.group_id,
                f"{g.h_interval[0]}-{g.h_interval[1]}",
                f"{g.s_interval[0]}-{g.s_interval[1]}",
                f"{g.v_interval[0]}-{g.v_interval[1]}",
                f"{g.pixel_count:,}",
                f"{g.percentage*100:.2f}%"
            )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="HSV 颜色分类工具（三通道组合编码）")
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--imagenet-path', type=str, default='data/imagenet1k',
                        help='ImageNet 数据集路径')
    parser.add_argument('--output-dir', type=str, default='output/hsv_masks',
                        help='输出目录')
    parser.add_argument('--random', action='store_true',
                        help='从 ImageNet 随机抽取一张图片')
    parser.add_argument('--h-intervals', type=int, default=24,
                        help='H 通道区间数 (默认 24)')
    parser.add_argument('--s-intervals', type=int, default=12,
                        help='S 通道区间数 (默认 12)')
    parser.add_argument('--v-intervals', type=int, default=12,
                        help='V 通道区间数 (默认 12)')
    parser.add_argument('--min-pixels', type=int, default=1,
                        help='最小像素数阈值 (默认 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='计算设备')

    args = parser.parse_args()
    console = Console()

    # 配置分类器
    config = HSVClassifierConfig(
        h_intervals=args.h_intervals,
        s_intervals=args.s_intervals,
        v_intervals=args.v_intervals,
        min_pixel_count=args.min_pixels,
        device=args.device
    )

    console.print(f"[cyan]H 区间数: {args.h_intervals}[/cyan]")
    console.print(f"[cyan]S 区间数: {args.s_intervals}[/cyan]")
    console.print(f"[cyan]V 区间数: {args.v_intervals}[/cyan]")
    console.print(f"[cyan]理论最大组合数: {args.h_intervals * args.s_intervals * args.v_intervals}[/cyan]")

    separator = HSVSeparator(config)

    # 获取图片路径
    if args.image:
        image_path = Path(args.image)
    elif args.random:
        imagenet_path = Path(args.imagenet_path)
        if not imagenet_path.exists():
            console.print(f"[red]ImageNet 路径不存在: {imagenet_path}[/red]")
            return

        all_images = list(imagenet_path.rglob('*.JPEG'))
        if not all_images:
            console.print(f"[red]未找到图片文件[/red]")
            return

        image_path = random.choice(all_images)
        console.print(f"[cyan]随机选择: {image_path}[/cyan]")
    else:
        console.print("[yellow]未指定图片，使用 --random 随机抽取或 --image 指定图片[/yellow]")
        console.print("[yellow]尝试从默认路径随机抽取...[/yellow]")
        imagenet_path = Path(args.imagenet_path)
        if imagenet_path.exists():
            all_images = list(imagenet_path.rglob('*.JPEG'))
            if all_images:
                image_path = random.choice(all_images)
                console.print(f"[cyan]随机选择: {image_path}[/cyan]")
            else:
                console.print(f"[red]未找到图片文件[/red]")
                return
        else:
            console.print(f"[red]请指定 --image 或 --random[/red]")
            return

    # 处理图片
    console.print(f"\n[bold]正在处理...[/bold]")
    result = separator.separate_combined_from_file(image_path)

    # 保存结果
    output_path = Path(args.output_dir) / image_path.stem
    metadata, quality = save_combined_results(result, output_path, separator)

    # 打印摘要
    print_combined_summary(result, quality, console)
    console.print(f"\n[green]结果已保存到: {output_path}[/green]")
    console.print(f"[green]对比图: {output_path / 'comparison.png'}[/green]")
    console.print(f"[green]组合 mask 目录: {output_path / 'masks'}[/green]")


if __name__ == '__main__':
    main()
