"""
PPT PDF 拆页为图片工具
将 PDF 格式的 PPT 文件按页拆解为 PNG 图片并保存。

用法:
    python ppt_to_images.py <file1.pdf> [file2.pdf ...] [--dpi 200]

输出:
    每个 PDF 文件同目录下创建 {文件名}_pages/ 文件夹，
    保存 page_001.png, page_002.png, ...
"""

import argparse
from pathlib import Path

import fitz  # pymupdf


def pdf_to_images(pdf_path: str, dpi: int = 200) -> None:
    """将 PDF 文件按页渲染为 PNG 图片。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[错误] 文件不存在: {pdf_path}")
        return

    output_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
    output_dir.mkdir(exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"正在处理: {pdf_path.name} ({total} 页)")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        out_file = output_dir / f"page_{i + 1:03d}.png"
        pix.save(str(out_file))
        print(f"  [{i + 1}/{total}] 已保存: {out_file.name}")

    doc.close()
    print(f"完成! 图片保存在: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="将 PDF 拆页为图片")
    parser.add_argument("file", nargs="?", default="./GAMES001-Lecture01.pdf", help="PDF 文件路径")
    parser.add_argument("--dpi", type=int, default=200, help="输出图片的 DPI (默认 200)")
    args = parser.parse_args()

    pdf_to_images(args.file, args.dpi)


if __name__ == "__main__":
    main()
