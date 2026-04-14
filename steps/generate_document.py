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


def generate_document(output_dir: str) -> str:
    """读取 output 目录中的所有结果，生成一份多模态 Markdown 文档。

    Args:
        output_dir: 输出目录路径，包含 summary.md 和 segments/ 子目录。

    Returns:
        生成的文档写入 output_dir/course_document.md，同时返回文档内容。
    """
    output_path = Path(output_dir)
    segments_dir = output_path / "segments"
    lines: list[str] = []

    # ---- 文档标题 ----
    lines.append("# 课程文档\n")

    # ---- 课程总结（summary.md） ----
    summary_path = output_path / "summary.md"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8").strip()
        # summary.md 自带 # 课程总结 / # 知识点列表，提升一级使其成为子章节
        lines.append(_promote_headings(summary_text))
        lines.append("")
    else:
        lines.append("*（未找到 summary.md）*\n")

    # ---- 分隔线 ----
    lines.append("---\n")

    # ---- 逐个 segment ----
    if segments_dir.exists():
        seg_dirs = sorted(
            [d for d in segments_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        for seg_dir in seg_dirs:
            seg_name = seg_dir.name
            try:
                page_num = int(seg_name) + 1
            except ValueError:
                page_num = seg_name

            lines.append(f"## 第 {page_num} 页\n")

            # -- segment_summary.md --
            seg_summary_path = seg_dir / "segment_summary.md"
            if seg_summary_path.exists():
                seg_text = seg_summary_path.read_text(encoding="utf-8").strip()
                # 提升标题级别（原文 # → ##，## → ###）
                lines.append(_promote_headings(seg_text))
                lines.append("")

            # -- 帧图片 --
            frames_dir = seg_dir / "frames"
            if frames_dir.exists():
                frame_files = sorted(
                    f for f in frames_dir.iterdir()
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
    print(f"多模态文档已生成: {out_path}")
    return doc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成多模态课程文档")
    parser.add_argument("--output", default="./output", help="输出目录路径")
    args = parser.parse_args()
    generate_document(args.output)
