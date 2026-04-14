"""流式处理：提取帧 + CLIP 相似度计算 + 段落切分，合并为单次视频遍历。"""
import csv
import os
import re
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


def extract_frames_with_similarity(video_path, output_dir, step_seconds=0.5,
                                    image_format="png", device=None):
    """单次遍历视频：提取帧 + 计算 CLIP 相邻帧相似度。

    与原来 extract_frames + compute_dir_adjacent_similarities 两步相比，
    此函数只遍历视频一次，CLIP 特征在内存中计算，不写 PNG 再读回。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录 (写入 frames/ 和 adjacent_similarity.csv)
        step_seconds: 帧提取间隔（秒）
        image_format: 图片格式
        device: CLIP 运行设备

    Returns:
        (frame_paths, csv_path, n_pairs)
    """
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np

    model, preprocess, device = _load_clip(device)

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "adjacent_similarity.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = None
    if fps and fps > 0 and frame_count and frame_count > 0:
        duration_ms = int((frame_count / fps) * 1000)

    step_ms = int(step_seconds * 1000)

    # 流式处理：每次读取一帧，CLIP 编码后保留 feature 用于下一对比较
    frame_paths = []
    prev_feature = None
    prev_name = None
    similarities = []  # [(name1, name2, sim), ...]

    t = 0
    if duration_ms is not None:
        last_t = (duration_ms // step_ms) * step_ms
        time_points = range(0, last_t + 1, step_ms)
    else:
        # 回退：逐帧尝试直到读取失败
        time_points = []
        t = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, t)
            ret, _ = cap.read()
            if not ret:
                break
            time_points.append(t)
            t += step_ms
        # 重置回开头
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    for t in time_points:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ret, frame = cap.read()
        if not ret:
            continue

        name = f"frame_{t:09d}.{image_format}"
        path = str(frames_dir / name)

        # CLIP 编码 — 直接从内存中的帧处理，不写磁盘
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(tensor)
        feature = F.normalize(feature, dim=-1)

        # 写帧到磁盘
        if image_format.lower() in ("jpg", "jpeg"):
            cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(path, frame)
        frame_paths.append(path)

        # 计算与前一帧的相似度
        if prev_feature is not None:
            sim = F.cosine_similarity(prev_feature, feature).item()
            similarities.append((prev_name, name, sim))

        prev_feature = feature
        prev_name = name

    cap.release()

    # 写入 list 文件
    list_file = frames_dir / "images.txt"
    list_file.write_text("\n".join(frame_paths), encoding="utf-8")

    # 写 CSV
    os.makedirs(os.path.dirname(str(csv_path)) or ".", exist_ok=True)
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image1", "image2", "similarity"])
        for name1, name2, sim in similarities:
            w.writerow([name1, name2, f"{sim:.6f}"])

    return frame_paths, csv_path, len(similarities)


def deduplicate_and_segment(csv_path, frames_dir, output_dir, segments_dir,
                             dup_threshold=0.9, threshold_low=0.62,
                             min_gap=15, min_duration_s=60):
    """合并：去重 + PPT 页面切分。读取 CSV，输出 filtered 帧和 segments 列表。

    Args:
        csv_path: adjacent_similarity.csv
        frames_dir: 原始帧目录
        output_dir: filtered 帧输出目录
        segments_dir: segments 输出目录
        dup_threshold: 去重阈值
        threshold_low: 切分阈值
        min_gap: 最小切分间隔（行数）
        min_duration_s: 最小片段时长（秒）

    Returns:
        segments: list of lists, each inner list is frame file paths for one PPT page
    """
    # 第一遍：读取 CSV，识别重复帧和切分点
    duplicates = set()
    all_images_ordered = []
    seen_images = set()
    low_sim_positions = []  # 相似度低于 threshold_low 的行号
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for file_line, row in enumerate(reader, start=2):
            total_rows += 1
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

            if similarity > dup_threshold:
                duplicates.add(img2)
            if similarity < threshold_low:
                low_sim_positions.append(file_line)

    last_line = total_rows + 1

    # 去重：复制非重复帧
    final_images = [img for img in all_images_ordered if img not in duplicates]
    os.makedirs(output_dir, exist_ok=True)
    import shutil
    for img in final_images:
        src = os.path.join(frames_dir, img)
        dst = os.path.join(output_dir, img)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)

    # 切分点过滤（min_gap）
    filtered_positions = []
    last = None
    for p in low_sim_positions:
        if last is None or p - last > min_gap:
            filtered_positions.append(p)
            last = p

    # 过滤短片段
    if min_duration_s and min_duration_s > 0:
        filtered_positions = _filter_short_segments(
            filtered_positions, 2, last_line, csv_path, min_duration_s
        )

    # 构建 segments
    raw_segments = _build_segments(filtered_positions, 2, last_line)
    segments = []
    for seg in raw_segments:
        paths = _process_segment(csv_path, seg, frames_dir, dup_threshold)
        if paths:
            segments.append(paths)

    # 将 segment 帧复制到 segments_dir
    os.makedirs(segments_dir, exist_ok=True)
    for idx, seg_paths in enumerate(segments):
        seg_dir = os.path.join(segments_dir, f"{idx:02d}", "frames")
        os.makedirs(seg_dir, exist_ok=True)
        for p in seg_paths:
            dst = os.path.join(seg_dir, os.path.basename(p))
            if os.path.exists(p):
                shutil.copy2(p, dst)

    return segments


# --- 以下函数从 split_segments.py 复用 ---

def _parse_ms_from_line(csv_path, line_num):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for file_line, row in enumerate(reader, start=2):
            if file_line == line_num:
                img = row.get("image1", "").strip()
                match = re.search(r"frame_(\d+)", img)
                if match:
                    return int(match.group(1))
                return 0
    return 0


def _filter_short_segments(filtered_positions, start_line, end_line, csv_path, min_duration_s):
    min_ms = int(min_duration_s * 1000)
    all_points = [start_line] + filtered_positions + [end_line]
    changed = True
    while changed:
        changed = False
        new_points = [all_points[0]]
        for i in range(1, len(all_points)):
            prev_ms = _parse_ms_from_line(csv_path, new_points[-1])
            cur_ms = _parse_ms_from_line(csv_path, all_points[i])
            if cur_ms - prev_ms < min_ms:
                changed = True
                continue
            new_points.append(all_points[i])
        all_points = new_points
    return all_points[1:-1]


def _build_segments(filtered_positions, start_line, end_line):
    if not filtered_positions:
        return [(start_line, end_line)]
    segments = []
    prev = start_line
    for p in filtered_positions:
        segments.append((prev, p - 1))
        prev = p
    segments.append((prev, end_line))
    return segments


def _process_segment(csv_path, segment, frames_dir, dup_threshold):
    start_line, end_line = segment
    seen = set()
    ordered = []
    duplicates = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for file_line, row in enumerate(reader, start=2):
            if file_line < start_line:
                continue
            if file_line > end_line:
                break
            img1 = row.get("image1", "").strip()
            img2 = row.get("image2", "").strip()
            try:
                s = float(row["similarity"])
            except (KeyError, ValueError):
                continue
            if img1 and img1 not in seen:
                ordered.append(img1)
                seen.add(img1)
            if img2 and img2 not in seen:
                ordered.append(img2)
                seen.add(img2)
            if s > dup_threshold and img2:
                duplicates.add(img2)
    final_images = [i for i in ordered if i not in duplicates]
    full_paths = [os.path.join(frames_dir, i) for i in final_images]
    return full_paths
