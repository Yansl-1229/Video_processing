"""使用 qwen-plus 对每个视频片段单独提取知识点和讲解摘要"""
import os
import re
from pathlib import Path
from dashscope import Generation,MultiModalConversation
import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


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

    prompt = f"""
# Role
你是一位专业的课程内容分析师和教育摘要专家。你擅长从冗长的课程转录文本中提取关键信息，并将其整理为结构清晰、易于理解的知识点摘要。

# Task
请阅读提供的【转录文本】，分析其中的教学内容，并严格按照指定的 Markdown 格式输出该页面的知识点与讲解摘要。

# Inputs
- **页码**: {page_num}
- **时间信息**: {time_info}
- **转录文本**: {text}


# Constraints & Guidelines
1. **内容提取**：忽略口语化的填充词（如“呃”、“那个”）、重复语句和无关闲聊，只保留核心教学内容和逻辑。
2. **结构清晰**：将内容归纳为若干个核心知识点，每个知识点需包含标题和详细解释。
3. **语言风格**：使用专业、简洁的教学语言，对原文进行适当的润色和总结，而非简单的复制粘贴。
4. **格式严格**：必须严格遵守下方的【Output Format】，不要添加任何额外的开场白或结束语。

# Output Format
请完全按照以下模板输出（不要修改标题层级）：

# 第 {page_num} 节 - 知识点与讲解摘要

{time_info}

一、核心讲解知识点
（一）[知识点名称]
[此处填写该知识点的详细讲解内容，包括定义、原理、案例等]

（二）[知识点名称]
[此处填写该知识点的详细讲解内容]

......（根据实际内容数量列出所有知识点）

二、整体内容概要总结
[用简练的语言概括本页内容的中心思想或教学目标，字数控制在300-500字左右]
"""

    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ]
    
    kwargs = {
        "model": "qwen-plus",
        "messages": messages,
        "result_format": "message",
    }
    api_key_to_use = api_key or os.getenv("DASHSCOPE_API_KEY")
    if api_key_to_use:
        kwargs["api_key"] = api_key_to_use
        
    response = Generation.call(**kwargs)

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
