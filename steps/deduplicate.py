"""基于 CLIP 相似度去除重复帧"""
import csv
import os
import shutil


def deduplicate_frames(csv_path, frames_dir, output_dir, threshold=0.9):
    """读取相似度 CSV，去除相似度 > threshold 的重复帧，复制非重复帧到 output_dir。

    Returns: list of kept image filenames (in order)
    """
    duplicates = set()
    all_images_ordered = []
    seen_images = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img1 = row["image1"].strip()
            img2 = row["image2"].strip()
            try:
                similarity = float(row["similarity"])
            except ValueError:
                continue

            if img1 not in seen_images:
                all_images_ordered.append(img1)
                seen_images.add(img1)
            if img2 not in seen_images:
                all_images_ordered.append(img2)
                seen_images.add(img2)

            if similarity > threshold:
                duplicates.add(img2)

    final_images = [img for img in all_images_ordered if img not in duplicates]

    os.makedirs(output_dir, exist_ok=True)
    for img in final_images:
        src = os.path.join(frames_dir, img)
        dst = os.path.join(output_dir, img)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)

    return final_images
