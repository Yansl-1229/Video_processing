"""使用 qwen3.6-flash 多模态模型，逐张描述 frames_ppt 中的图片，
再结合 segment_summary.md 生成图文并茂的 Markdown 文档。

用法:
    python -c "from steps.generate_multimodal_segment import generate_multimodal_segments; generate_multimodal_segments('./output')"
"""
import os
import time
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

API_DELAY = 1  # 秒，避免 API 限流


def _describe_single_image(image_path: str, api_key: str) -> str:
    """调用 qwen3.6-flash 为单张图片生成不超过 50 字的描述。"""
    abs_path = os.path.abspath(image_path).replace("\\", "/")
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{abs_path}"},
                {"text": "请用不超过50个字描述这张图片的内容，重点描述图片中的文字、图表、公式或关键信息。"},
            ],
        }
    ]
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen3.6-flash",
        messages=messages,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API 调用失败: {response.message}")
    return response.output.choices[0].message.content[0]["text"].strip()


def _build_image_descriptions_doc(seg_dir: Path, image_descriptions: dict[str, str]) -> str:
    """将图片描述和路径组装为 markdown 格式的图片描述文档。

    格式: ![描述](图片路径)
    """
    lines = []
    for filename, desc in image_descriptions.items():
        rel_path = f"frames_ppt/{filename}"
        lines.append(f"![{desc}]({rel_path})")
    return "\n".join(lines)


def _generate_final_doc(summary_text: str, image_desc_doc: str, api_key: str, segment_idx: int) -> str:
    """调用模型，结合 segment_summary.md 和图片描述文档，生成图文并茂的 markdown。"""
    prompt = f"""你是一个课程内容整理助手。下面有两份材料：

1. **课程知识点摘要文档**：对课程片段的文字总结
2. **图片描述文档**：该课程片段中截帧图片的描述及路径，格式为 ![描述](路径)

请将图片合理地插入到文档的对应位置，生成一份图文并茂的 Markdown 文档。

## 要求

- 根据图片描述的内容与文档段落的语义关联，将每张图片插入到最合适的段落位置
- 图片应放在与其内容相关的文字之后
- 保持原文档的标题结构和文字内容不变
- 只在需要的地方插入图片，不要强行插入无关图片
- 输出完整的 Markdown 文档

## 课程知识点摘要文档

{summary_text}

## 图片描述文档

{image_desc_doc}"""

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen3.6-flash",
        messages=messages,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API 调用失败 (segment {segment_idx}): {response.message}")
    return response.output.choices[0].message.content[0]["text"].strip()


def generate_multimodal_segments(segments_dir: str, api_key: str = None):
    """遍历所有 segment 目录，逐张描述图片后生成图文并茂的文档。

    Args:
        segments_dir: segments/ 目录路径
        api_key: DashScope API key（可选）
    """
    segments_dir = Path(segments_dir)
    api_key_to_use = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key_to_use:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量")

    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue

        summary_path = seg_dir / "segment_summary.md"
        frames_ppt_dir = seg_dir / "frames_ppt"

        if not summary_path.exists():
            print(f"  跳过 {seg_dir.name}: 无 segment_summary.md")
            continue
        if not frames_ppt_dir.exists():
            print(f"  跳过 {seg_dir.name}: 无 frames_ppt 目录")
            continue

        summary_text = summary_path.read_text(encoding="utf-8").strip()
        if not summary_text:
            print(f"  跳过 {seg_dir.name}: segment_summary.md 为空")
            continue

        # 收集图片
        image_files = sorted(
            f for f in os.listdir(frames_ppt_dir) if f.lower().endswith(".png")
        )
        if not image_files:
            print(f"  跳过 {seg_dir.name}: frames_ppt 中无图片")
            continue

        idx = int(seg_dir.name)
        page_num = idx + 1
        print(f"  处理第 {page_num} 页 ({len(image_files)} 张图片)...", flush=True)

        # === 第一步：逐张图片生成描述 ===
        image_descriptions: dict[str, str] = {}
        for i, filename in enumerate(image_files):
            image_path = str(frames_ppt_dir / filename)
            try:
                t0 = time.time()
                desc = _describe_single_image(image_path, api_key_to_use)
                elapsed = time.time() - t0
                image_descriptions[filename] = desc
                print(f"    [{i+1}/{len(image_files)}] {filename}: {desc} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"    [{i+1}/{len(image_files)}] {filename}: 描述失败 - {e}")

            # 避免限流
            if i < len(image_files) - 1:
                time.sleep(API_DELAY)

        if not image_descriptions:
            print(f"    所有图片描述失败，跳过 {seg_dir.name}")
            continue

        # === 第二步：组装图片描述文档 ===
        image_desc_doc = _build_image_descriptions_doc(seg_dir, image_descriptions)
        print(f"    图片描述文档已生成 ({len(image_descriptions)} 张图片)")

        # === 第三步：调用模型生成图文并茂的最终文档 ===
        try:
            t0 = time.time()
            final_doc = _generate_final_doc(summary_text, image_desc_doc, api_key_to_use, idx)
            elapsed = time.time() - t0
            output_path = seg_dir / "segment_summary_with_images.md"
            output_path.write_text(final_doc, encoding="utf-8")
            print(f"    已保存: {output_path} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    最终文档生成失败: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成图文并茂的 segment 摘要文档")
    parser.add_argument("--output", default="./output", help="输出目录路径")
    args = parser.parse_args()

    segments_dir = os.path.join(args.output, "segments")
    generate_multimodal_segments(segments_dir)
