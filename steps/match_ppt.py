"""使用 CLIP 相似度将视频帧替换为最相似的 PPT 图片。"""
import os
import shutil
from pathlib import Path


def _load_clip(device=None):
    """加载 CLIP 模型和预处理函数"""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import clip as clip_oa
        model, preprocess = clip_oa.load("ViT-B/32", device=device)
    except ModuleNotFoundError:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device)
    return model, preprocess, device


def _encode_images(image_paths, model, preprocess, device):
    """批量编码图片为 CLIP 特征"""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np

    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
            features.append(feat.cpu())
        except Exception as e:
            print(f"  警告: 无法编码图片 {path}: {e}")
            features.append(None)
    return features


def _deduplicate_by_clip(image_paths, model, preprocess, device, threshold=0.97):
    """对一组图片进行 CLIP 去重，保留每组中第一张"""
    import torch
    import torch.nn.functional as F

    if not image_paths:
        return []

    features = _encode_images(image_paths, model, preprocess, device)

    unique_paths = []
    unique_features = []

    for path, feat in zip(image_paths, features):
        if feat is None:
            continue

        is_duplicate = False
        for ufeat in unique_features:
            if ufeat is not None:
                sim = F.cosine_similarity(feat, ufeat, dim=-1).item()
                if sim > threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_paths.append(path)
            unique_features.append(feat)

    return unique_paths


def match_frames_to_ppt(frames_dir, ppt_dir, output_dir, dup_threshold=0.99,
                        min_similarity=0.0, device=None):
    """将帧图片替换为最相似的 PPT 图片，并进行去重。

    Args:
        frames_dir: 输入帧目录 (如 output/filtered 或 output/segments/XX/frames)
        ppt_dir: PPT 图片目录 (GAMES001-Lecture01_pages)
        output_dir: 输出目录
        dup_threshold: 段内去重阈值 (默认 0.99)
        min_similarity: 最小相似度阈值 (默认 0.0)，当相似度低于此值时保留原始帧
        device: CLIP 运行设备
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, preprocess, device = _load_clip(device)

    # 获取帧文件列表
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_files:
        print(f"  警告: 目录中未找到图片 {frames_dir}")
        return

    # 获取PPT文件列表
    ppt_files = sorted([
        f for f in os.listdir(ppt_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not ppt_files:
        print(f"  错误: PPT目录中未找到图片 {ppt_dir}")
        return

    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    ppt_paths = [os.path.join(ppt_dir, f) for f in ppt_files]

    print(f"  编码 {len(frame_paths)} 张帧图片...")
    frame_features = _encode_images(frame_paths, model, preprocess, device)

    print(f"  编码 {len(ppt_paths)} 张PPT图片...")
    ppt_features = _encode_images(ppt_paths, model, preprocess, device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 对每张帧图片，找到最相似的PPT图片
    matched_count = 0
    for i, (frame_file, frame_feat) in enumerate(zip(frame_files, frame_features)):
        if frame_feat is None:
            continue

        best_sim = -1
        best_ppt_idx = 0

        for j, ppt_feat in enumerate(ppt_features):
            if ppt_feat is None:
                continue
            sim = F.cosine_similarity(frame_feat, ppt_feat, dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_ppt_idx = j

        # 根据相似度决定替换为PPT还是保留原始帧
        if best_sim >= min_similarity:
            # 相似度足够高，替换为PPT图片
            src_ppt = ppt_paths[best_ppt_idx]
            dst_path = os.path.join(output_dir, frame_file)
            shutil.copy2(src_ppt, dst_path)
        else:
            # 相似度太低，保留原始帧
            src_frame = os.path.join(frames_dir, frame_file)
            dst_path = os.path.join(output_dir, frame_file)
            shutil.copy2(src_frame, dst_path)
        matched_count += 1

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(frame_files)} 张帧图片")

    print(f"  完成: {matched_count} 张帧图片已替换为最相似的PPT图片")

    # 去重
    output_files = sorted([
        f for f in os.listdir(output_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    output_paths = [os.path.join(output_dir, f) for f in output_files]

    print(f"  对输出图片进行去重 (阈值={dup_threshold})...")
    unique_paths = _deduplicate_by_clip(output_paths, model, preprocess, device, dup_threshold)

    # 删除重复图片
    duplicate_files = set(output_files) - set(os.path.basename(p) for p in unique_paths)
    for f in duplicate_files:
        os.remove(os.path.join(output_dir, f))

    print(f"  去重完成: 保留 {len(unique_paths)} 张图片，移除 {len(duplicate_files)} 张重复图片")


def process_filtered_frames(output_dir, ppt_dir, dup_threshold=0.99,
                             min_similarity=0.0, device=None):
    """处理 output/filtered 目录中的帧图片

    Args:
        output_dir: output 目录 (包含 filtered 和 segments)
        ppt_dir: PPT 图片目录
        dup_threshold: 段内去重阈值
        min_similarity: 最小相似度阈值，低于此值保留原始帧
        device: CLIP 运行设备
    """
    filtered_dir = os.path.join(output_dir, "filtered")
    if not os.path.exists(filtered_dir):
        print(f"错误: filtered 目录不存在: {filtered_dir}")
        return

    # 输出到 filtered_ppt 目录
    filtered_ppt_dir = os.path.join(output_dir, "filtered_ppt")
    print(f"处理 filtered 帧: {filtered_dir}")
    match_frames_to_ppt(filtered_dir, ppt_dir, filtered_ppt_dir, dup_threshold, min_similarity, device)


def process_segments(output_dir, ppt_dir, dup_threshold=0.99,
                      min_similarity=0.0, device=None):
    """处理 output/segments 目录下所有 segment 的帧图片

    Args:
        output_dir: output 目录 (包含 segments)
        ppt_dir: PPT 图片目录
        dup_threshold: 段内去重阈值
        min_similarity: 最小相似度阈值，低于此值保留原始帧
        device: CLIP 运行设备
    """
    segments_dir = os.path.join(output_dir, "segments")
    if not os.path.exists(segments_dir):
        print(f"错误: segments 目录不存在: {segments_dir}")
        return

    # 获取所有segment目录
    segment_dirs = sorted([
        d for d in os.listdir(segments_dir)
        if os.path.isdir(os.path.join(segments_dir, d)) and d.isdigit()
    ])

    for seg_id in segment_dirs:
        frames_dir = os.path.join(segments_dir, seg_id, "frames")
        if not os.path.exists(frames_dir):
            print(f"  跳过空的 segment: {seg_id}")
            continue

        # 输出到 segments_ppt 目录
        seg_ppt_dir = os.path.join(segments_dir, seg_id, "frames_ppt")
        print(f"处理 segment {seg_id} 帧: {frames_dir}")
        match_frames_to_ppt(frames_dir, ppt_dir, seg_ppt_dir, dup_threshold, min_similarity, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将视频帧替换为最相似的PPT图片")
    parser.add_argument("--output", default="./output041302", help="output 目录")
    parser.add_argument("--ppt", default="./GAMES001-Lecture01_pages", help="PPT 图片目录")
    parser.add_argument("--dup-threshold", type=float, default=0.99, help="段内去重阈值")
    parser.add_argument("--min-similarity", type=float, default=0.0,
                        help="最小相似度阈值，低于此值保留原始帧 (默认 0.0)")
    parser.add_argument("--device", default=None, help="CLIP 运行设备 (cuda/cpu)")

    args = parser.parse_args()

    # 处理 filtered
    process_filtered_frames(args.output, args.ppt, args.dup_threshold,
                            args.min_similarity, args.device)

    # 处理 segments
    process_segments(args.output, args.ppt, args.dup_threshold,
                      args.min_similarity, args.device)

    print("\n完成! 所有帧图片已替换为最相似的PPT图片。")
