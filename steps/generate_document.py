"""后处理：将 output 目录中的结果整合为一份多模态 Markdown 文档。

用法:
    python -c "from steps.generate_document import generate_document; generate_document('./output')"
"""
import os
import re
from pathlib import Path


def _promote_headings(text: str, levels: int = 1) -> str:
    """将 markdown 文本中的标题级别提升指定数量的 #。"""
    def _replace(m):
        return '#' * (len(m.group(1)) + levels) + m.group(2)
    return re.sub(r'^(#{1,6})(\s+.+)$', _replace, text, flags=re.MULTILINE)


def _frame_time_ms(frame_path: str) -> int:
    """从帧文件名中提取毫秒时间戳，如 frame_000009000.png → 9000。"""
    stem = Path(frame_path).stem  # frame_000009000
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def _format_time(ms: int) -> str:
    """将毫秒转为 HH:MM:SS 或 MM:SS 格式。"""
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def generate_document(output_dir: str, segments_dir: str = None, segments: list = None) -> str:
    """读取 output 目录中的所有结果，生成一份多模态 Markdown 文档。

    Args:
        output_dir: 输出目录路径，包含 summary.md 和 segments/ 子目录。
        segments_dir: segments 目录路径，如果为 None 则自动从 output_dir 推导。
        segments: 来自 deduplicate_and_segment() 的 list[list[str]]，用于提取起始时间。
                  如果为 None 则跳过起始时间标注。

    Returns:
        生成的文档写入 output_dir/course_document.md，返回文档文件路径。
    """
    output_path = Path(output_dir)
    if segments_dir is None:
        segments_dir = output_path / "segments"
    segments_path = Path(segments_dir)
    lines: list[str] = []

    # ---- 文档标题 ----
    lines.append("# 课程文档\n")

    # ---- 课程总结（summary.md） ----
    summary_path = output_path / "summary.md"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8").strip()
        lines.append(_promote_headings(summary_text))
        lines.append("")
    else:
        lines.append("*（未找到 summary.md）*\n")

    # ---- 分隔线 ----
    lines.append("---\n")

    # ---- 逐个 segment ----
    if segments_path.exists():
        seg_dirs = sorted(
            [d for d in segments_path.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        for seg_dir in seg_dirs:
            seg_name = seg_dir.name
            try:
                page_num = int(seg_name) + 1
            except ValueError:
                page_num = seg_name

            # -- 起始时间 --
            start_time_str = ""
            if segments is not None:
                try:
                    seg_idx = int(seg_name)
                    if 0 <= seg_idx < len(segments) and segments[seg_idx]:
                        first_frame = segments[seg_idx][0]
                        start_ms = _frame_time_ms(first_frame)
                        start_time_str = f"（起始时间: {_format_time(start_ms)}）"
                except (ValueError, IndexError):
                    pass

            lines.append(f"## 第 {page_num} 页 {start_time_str}\n")

            # -- segment_summary.md --
            seg_summary_path = seg_dir / "segment_summary.md"
            if seg_summary_path.exists():
                seg_text = seg_summary_path.read_text(encoding="utf-8").strip()
                lines.append(_promote_headings(seg_text))
                lines.append("")

            # -- 帧图片（优先使用 frames_ppt） --
            frames_ppt_dir = seg_dir / "frames_ppt"
            frames_orig_dir = seg_dir / "frames"
            img_dir = frames_ppt_dir if frames_ppt_dir.exists() else frames_orig_dir
            if img_dir.exists():
                frame_files = sorted(
                    f for f in img_dir.iterdir()
                    if f.suffix.lower() == ".png"
                )
                if frame_files:
                    lines.append("### 关键帧\n")
                    for frame_file in frame_files:
                        rel = os.path.relpath(frame_file, output_path).replace("\\", "/")
                        lines.append(f"![{frame_file.stem}]({rel})")
                        lines.append("")

            # -- 视频片段 --
            video_path = seg_dir / "video.mp4"
            if video_path.exists():
                rel_video = os.path.relpath(video_path, output_path).replace("\\", "/")
                lines.append("### 视频片段\n")
                lines.append(f'<video src="{rel_video}" controls width="640"></video>')
                lines.append("")
                lines.append(f"[video.mp4]({rel_video})")
                lines.append("")

            # -- 分隔线 --
            lines.append("---\n")

    # ---- 写入文件 ----
    doc = "\n".join(lines)
    out_path = output_path / "course_document.md"
    out_path.write_text(doc, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成多模态课程文档")
    parser.add_argument("--output", default="./output", help="输出目录路径")
    args = parser.parse_args()
    generate_document(args.output)
