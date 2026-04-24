"""遮挡帧左下角摄像头区域：在每帧左下角覆盖白色矩形。"""

import glob
from pathlib import Path

import cv2


def overlay_camera_block(filtered_dir: str, width_ratio: float = 0.243, height_ratio: float = 0.243):
    """对 filtered 目录下的每帧图片，在左下角覆盖白色矩形以遮挡摄像头。

    Args:
        filtered_dir: filtered 帧目录路径
        width_ratio: 矩形宽度占图片宽度的比例，默认 0.243
        height_ratio: 矩形高度占图片高度的比例，默认 0.243
    """
    filtered_dir = Path(filtered_dir)
    if not filtered_dir.exists():
        print(f"  filtered 目录不存在，跳过: {filtered_dir}")
        return 0

    image_paths = sorted(glob.glob(str(filtered_dir / "*.png")))
    if not image_paths:
        print(f"  filtered 目录下没有 PNG 文件，跳过")
        return 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        rect_w = int(w * width_ratio)
        rect_h = int(h * height_ratio)

        # 左下角白色矩形
        cv2.rectangle(img, (0, h - rect_h), (rect_w, h), (255, 255, 255), -1)
        cv2.imwrite(img_path, img)

    return len(image_paths)
