"""后处理：将 output 目录中的结果整合为一份多模态 Markdown 文档。

用法:
    python -c "from steps.generate_document import generate_document; generate_document('./output')"
"""
import os
import re
from pathlib import Path


def _promote_headings(text: str, levels: int = 1) -> str:
    """将 markdown 文本中的标题级别提升指定数量的 #。"""
    import re
    def _replace(m):
        return '#' * (len(m.group(1)) + levels) + m.group(2)
    return re.sub(r'^(#{1,6})(\s+.+)$', _replace, text, flags=re.MULTILINE)


def generate_document(output_dir: str, segments_dir: str = None, segments: list = None) -> str:
    """读取 output 目录中的所有结果，生成一份多模态 Markdown 文档。

    Args:
        output_dir: 输出目录路径，包含 summary.md 和 segments/ 子目录。
        segments_dir: segments 目录路径，如果为 None 则自动从 output_dir 推导。
        segments: 未使用，保留参数以兼容 main.py 调用。

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

            lines.append(f"## 第 {page_num} 页\n")

            # 优先使用带图片的版本，否则使用原始版本
            summary_with_img = seg_dir / "segment_summary_with_images.md"
            summary_orig = seg_dir / "segment_summary.md"
            summary_file = summary_with_img if summary_with_img.exists() else summary_orig

            if summary_file.exists():
                seg_text = summary_file.read_text(encoding="utf-8").strip()
                # 如果是带图片的版本，修正图片路径前缀
                if summary_file == summary_with_img:
                    seg_text = re.sub(
                        r'!\[([^\]]*)\]\(frames_ppt/',
                        f'![\\1](segments/{seg_name}/frames_ppt/',
                        seg_text,
                    )
                lines.append(_promote_headings(seg_text))
                lines.append("")

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
