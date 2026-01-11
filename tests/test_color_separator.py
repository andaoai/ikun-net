"""
Test script for color separator functionality.
"""

import numpy as np
import cv2
from pathlib import Path


def create_test_image(output_path: str = "test_image.png", size: int = 400):
    """
    创建测试图片，包含不同颜色的色块。

    Args:
        output_path: 输出图片路径
        size: 图片大小（正方形边长）
    """
    # 创建空白图片
    image = np.zeros((size, size, 3), dtype=np.uint8)

    half = size // 2
    quarter = size // 4

    # 左上角：红色 (255, 0, 0)
    image[0:half, 0:half] = [255, 0, 0]

    # 右上角：绿色 (0, 255, 0)
    image[0:half, half:size] = [0, 255, 0]

    # 左下角：蓝色 (0, 0, 255)
    image[half:size, 0:half] = [0, 0, 255]

    # 右下角：白色 (255, 255, 255)
    image[half:size, half:size] = [255, 255, 255]

    # 保存图片 (OpenCV使用BGR格式，所以需要转换)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    print(f"Test image created: {output_path}")
    print(f"  Size: {size}x{size}")
    print(f"  Colors: Red (top-left), Green (top-right), Blue (bottom-left), White (bottom-right)")

    return output_path


if __name__ == "__main__":
    # 创建测试图片
    test_img_path = create_test_image()

    # 运行颜色分离
    print("\n" + "=" * 60)
    print("Running color separator...")
    print("=" * 60 + "\n")

    import subprocess
    result = subprocess.run(
        ["python", "main.py", "separate-colors", test_img_path, "--interval-size", "20"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    print(f"\nReturn code: {result.returncode}")

    # Check if original.png was created
    import os
    output_dir = "tmp/color_masks/test_image"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\nGenerated files: {len(files)}")
        if "original.png" in files:
            print("✓ Original image saved")
        if "metadata.json" in files:
            print("✓ Metadata JSON saved")
        if "summary.txt" in files:
            print("✓ Summary text saved")
        mask_files = [f for f in files if f.startswith("mask_")]
        print(f"✓ {len(mask_files)} mask files generated")
