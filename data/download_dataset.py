"""
ImageNet-1K 数据集下载脚本

使用 Kaggle API 下载 ImageNet-1K 数据集。
数据将保存在 data/imagenet1k/ 目录下。
"""

import argparse
import json
import os
from pathlib import Path

from rich.console import Console

console = Console()

# 可用的 ImageNet 数据集
IMAGENET_DATASETS = {
    "stable": "vitaliykinakh/stable-imagenet1k",  # Stable ImageNet-1K (推荐)
    "full": "lijiyu/imagenet",                     # 完整 ImageNet
    "mini": "ifigotin/imagenetmini-1000",         # 迷你版 (测试用)
}


def setup_kaggle_api(kaggle_key_path: str = None, username: str = None, key: str = None) -> bool:
    """
    配置 Kaggle API

    Args:
        kaggle_key_path: kaggle.json 文件路径
        username: Kaggle 用户名
        key: Kaggle API key

    Returns:
        配置是否成功
    """
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # 如果提供了用户名和 key，直接创建配置
    if username and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        console.print("[green]✓[/green] Kaggle API key 已配置")
        return True

    # 如果提供了自定义路径，从该路径读取
    if kaggle_key_path:
        src = Path(kaggle_key_path)
        if src.exists():
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, kaggle_json)
            os.chmod(kaggle_json, 0o600)
            console.print("[green]✓[/green] Kaggle API key 已配置")
            return True
        else:
            console.print(f"[red]✗[/red] 找不到文件: {kaggle_key_path}")
            return False

    # 检查是否已存在 kaggle.json
    if kaggle_json.exists():
        console.print("[green]✓[/green] Kaggle API key 已存在")
        return True

    # 尝试从环境变量读取
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if kaggle_username and kaggle_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)
        os.chmod(kaggle_json, 0o600)
        console.print("[green]✓[/green] Kaggle API key 已从环境变量配置")
        return True

    console.print("[yellow]![/yellow] 未找到 Kaggle API key")
    console.print("\n请通过以下方式之一配置:")
    console.print("  1. 设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY")
    console.print("  2. 将 kaggle.json 放在 ~/.kaggle/ 目录")
    console.print("  3. 运行时指定 kaggle.json 路径")
    console.print("  4. 运行时直接传入 --username 和 --key")
    return False


def download_imagenet1k(
    dataset: str = "stable",
    save_path: str = "data/imagenet1k",
    kaggle_key_path: str = None,
    username: str = None,
    key: str = None,
) -> None:
    """
    通过 Kaggle API 下载 ImageNet-1K 数据集

    Args:
        dataset: 数据集名称 (stable, full, mini)
        save_path: 数据保存路径
        kaggle_key_path: kaggle.json 文件路径
        username: Kaggle 用户名
        key: Kaggle API key
    """
    console.print("[bold cyan]ImageNet-1K 数据集下载工具 (Kaggle)[/bold cyan]\n")

    # 检查数据集名称
    if dataset not in IMAGENET_DATASETS:
        console.print(f"[red]✗[/red] 未知数据集: {dataset}")
        console.print(f"可用数据集: {', '.join(IMAGENET_DATASETS.keys())}")
        return

    dataset_ref = IMAGENET_DATASETS[dataset]
    console.print(f"[bold]数据集:[/bold] {dataset_ref}\n")

    # 配置 Kaggle API
    if not setup_kaggle_api(kaggle_key_path, username, key):
        return

    console.print(f"[bold blue]开始下载 ImageNet-1K ({dataset})...[/bold blue]")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # 验证 Kaggle API 认证
        api = KaggleApi()
        api.authenticate()

        console.print("[green]✓[/green] Kaggle API 认证成功")

        # 创建保存目录
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 下载数据集 (使用 dataset API 而不是 competition API)
        console.print(f"[yellow]下载数据集到: {save_path}[/yellow]")
        api.dataset_download_files(
            dataset=dataset_ref,
            path=str(save_path),
            unzip=False,  # 手动解压以显示进度
        )

        # 解压下载的文件
        console.print("[yellow]解压文件...[/yellow]")
        import zipfile
        for zip_file in save_path.glob("*.zip"):
            console.print(f"  解压: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            # 删除 zip 文件以节省空间
            zip_file.unlink()

        console.print(f"\n[green]✓[/green] 下载完成!")
        console.print(f"\n数据已保存到: {save_path}")

        # 列出下载的文件
        console.print("\n[bold]下载的文件:[/bold]")
        for item in sorted(save_path.rglob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                console.print(f"  - {item.relative_to(save_path)} ({size_mb:.1f} MB)")

    except ImportError:
        console.print("[bold red]✗[/bold red] 未安装 kaggle 包")
        console.print("请运行: uv add kaggle")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Kaggle API 下载失败: {e}")
        console.print("\n请确保:")
        console.print("  1. 已安装 kaggle: uv add kaggle")
        console.print("  2. 已配置正确的 API key")


def main():
    parser = argparse.ArgumentParser(description="下载 ImageNet-1K 数据集")
    parser.add_argument("--dataset", default="stable", choices=list(IMAGENET_DATASETS.keys()),
                        help="数据集名称 (stable: 推荐, full: 完整版, mini: 迷你版)")
    parser.add_argument("--save-path", default="data/imagenet1k", help="数据保存路径")
    parser.add_argument("--kaggle-key", help="kaggle.json 文件路径")
    parser.add_argument("--username", help="Kaggle 用户名")
    parser.add_argument("--key", help="Kaggle API key")

    args = parser.parse_args()

    download_imagenet1k(
        dataset=args.dataset,
        save_path=args.save_path,
        kaggle_key_path=args.kaggle_key,
        username=args.username,
        key=args.key,
    )


if __name__ == "__main__":
    main()
