"""使用 qwen3-max 对所有讲解内容进行总结和知识点提取"""
import os


def summarize_transcripts(transcripts, api_key=None):
    """对所有 PPT 页面的转录文本进行整体总结和知识点提取。

    Args:
        transcripts: list of (segment_index, transcript_text)
        api_key: DashScope API key

    Returns: markdown 格式的总结
    """
    from dashscope import Generation

    # 拼接所有转录文本
    combined = []
    for idx, text in transcripts:
        combined.append(f"## 第 {idx + 1} 页\n{text.strip()}")
    full_text = "\n\n".join(combined)

    prompt = f"""你是一位专业的课程内容分析助手。以下是一段 PPT 讲解视频的逐页转录内容：

{full_text}

请完成以下任务：
1. 对整个课程内容进行总结（200-500字）
2. 提取课程涉及的关键知识点，以列表形式呈现（每个知识点简洁明了）

请用 Markdown 格式输出，结构如下：
# 课程总结
（总结内容）

# 知识点列表
1. ...
2. ..."""

    messages = [
        {"role": "system", "content": "你是一位专业的课程内容分析助手，擅长从讲解内容中提取关键信息并进行总结。"},
        {"role": "user", "content": prompt}
    ]
    kwargs = {
        "model": "qwen3-max",
        "messages": messages,
    }
    if api_key:
        kwargs["api_key"] = api_key
    response = Generation.call(**kwargs)

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise RuntimeError(f"总结生成失败: {response.message}")
