"""按低相似度切分帧序列为 PPT 页面片段"""
import csv
import os
import re


def _stream_positions(csv_path, threshold):
    positions = []
    total_rows = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for file_line, row in enumerate(reader, start=2):
            total_rows += 1
            try:
                s = float(row["similarity"])
            except (KeyError, ValueError):
                continue
            if s < threshold:
                positions.append(file_line)
    last_line = total_rows + 1
    return positions, last_line


def _filter_positions(positions, min_gap):
    filtered = []
    last = None
    for p in positions:
        if last is None or p - last > min_gap:
            filtered.append(p)
            last = p
    return filtered


def _parse_ms_from_line(csv_path, line_num):
    """从 CSV 指定行中解析帧文件名对应的时间戳（毫秒）"""
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
    """移除会导致前后片段短于 min_duration_s 的切分点。"""
    min_ms = int(min_duration_s * 1000)
    # 候选切分点列表：start_line, p1, p2, ..., end_line
    all_points = [start_line] + filtered_positions + [end_line]
    # 过滤：移除导致片段过短的切分点
    changed = True
    while changed:
        changed = False
        new_points = [all_points[0]]
        for i in range(1, len(all_points)):
            prev_ms = _parse_ms_from_line(csv_path, new_points[-1])
            cur_ms = _parse_ms_from_line(csv_path, all_points[i])
            if cur_ms - prev_ms < min_ms:
                # 这段太短，跳过这个切分点（合并前后两段）
                changed = True
                continue
            new_points.append(all_points[i])
        all_points = new_points
    # 返回去除首尾（start_line, end_line）后的切分点
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


def split_by_similarity(csv_path, frames_dir, output_dir,
                         threshold_low=0.62, min_gap=15, dup_threshold=0.97,
                         min_duration_s=60):
    """按低相似度切分帧序列，每个 segment 对应一页 PPT。

    切分后过滤掉时长不足 min_duration_s 秒的片段（合并到相邻片段）。
    Returns: list of segments, each segment is a list of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    positions, last_line = _stream_positions(csv_path, threshold_low)
    filtered = _filter_positions(positions, min_gap)
    if min_duration_s and min_duration_s > 0:
        filtered = _filter_short_segments(filtered, 2, last_line, csv_path, min_duration_s)
    raw_segments = _build_segments(filtered, 2, last_line)

    segments = []
    for idx, seg in enumerate(raw_segments):
        paths = _process_segment(csv_path, seg, frames_dir, dup_threshold)
        if paths:
            segments.append(paths)
    return segments
