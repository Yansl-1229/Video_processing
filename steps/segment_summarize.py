"""使用 qwen-plus 对每个视频片段单独提取知识点和讲解摘要"""
import os
import re
from pathlib import Path

import dashscope


def _get_segment_start_time(seg_dir):
    """从 segment 目录中第一帧的文件名提取起始时间点（秒）。

    帧文件名格式: frame_XXXXXXXXX.png，XXXXXXXXX 为毫秒数。
    """
    frames_dir = Path(seg_dir) / "frames"
    if not frames_dir.exists():
        return None
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if re.match(r"frame_\d+", f)
    )
    if not frame_files:
        return None
    match = re.search(r"frame_(\d+)", frame_files[0])
    if match:
        ms = int(match.group(1))
        return ms / 1000.0
    return None


def _format_time(seconds):
    """将秒数格式化为 HH:MM:SS 或 MM:SS。"""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _summarize_single(text, page_num, start_time_s=None, api_key=None):
    """对单段转录文本调用 qwen-plus，返回 Markdown 总结。"""
    system = "你是一位专业的课程内容分析助手，擅长从讲解内容中提取关键知识点并进行简洁摘要。"
    time_info = ""
    if start_time_s is not None:
        formatted = _format_time(start_time_s)
        time_info = f"- **原视频时间点**: {formatted}\n"

    prompt = (
        "以下是一段课程讲解的转录文本，请从中提取：\n"
        "1. **知识点**：该段涉及的核心概念和知识点，以编号列表形式呈现\n"
        "2. **知识点内容描述**：对每个知识点用50-150字描述其具体内容和含义\n\n"
        "请用以下 Markdown 格式输出：\n\n"
        f"# 第 {page_num} 页 - 知识点与讲解摘要\n\n"
        f"{time_info}"
        "## 知识点\n"
        "1. ...\n"
        "2. ...\n\n"
        "## 知识点内容描述\n"
        "1. ...\n"
        "2. ...\n\n"
        f"转录文本：\n{text}"
    )
    kwargs = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if api_key:
        kwargs["api_key"] = api_key

    response = dashscope.Generation.call(**kwargs)
    if response.status_code == 200:
        output = response.output
        # qwen-plus 返回 output.text，qwen3-max 返回 output.choices[0].message.content
        if output.choices:
            return output.choices[0].message.content
        elif output.text:
            return output.text
        else:
            raise RuntimeError("API 返回了空内容")
    else:
        raise RuntimeError(f"qwen-plus 调用失败: {response.message}")


def summarize_segments(segments_dir, api_key=None):
    """遍历所有 segment 目录，逐个调用 qwen-plus 提取知识点。

    Args:
        segments_dir: segments/ 目录路径
        api_key: DashScope API key（可选）
    """
    segments_dir = Path(segments_dir)
    for seg_dir in sorted(segments_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        transcript_path = seg_dir / "transcript.txt"
        if not transcript_path.exists():
            continue
        text = transcript_path.read_text(encoding="utf-8").strip()
        if not text or text.startswith("[转录失败"):
            continue

        idx = int(seg_dir.name)
        page_num = idx + 1
        start_time = _get_segment_start_time(seg_dir)
        time_str = _format_time(start_time) if start_time is not None else "未知"
        print(f"  提取第 {page_num} 页知识点... (原视频 {time_str})", end=" ", flush=True)

        import time
        t0 = time.time()
        try:
            summary = _summarize_single(text, page_num, start_time_s=start_time, api_key=api_key)
            summary_path = seg_dir / "segment_summary.md"
            summary_path.write_text(summary, encoding="utf-8")
            print(f"({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
