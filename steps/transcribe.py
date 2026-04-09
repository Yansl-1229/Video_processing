"""使用 qwen3-omni-flash 将音频转录为文字"""
import base64
import math
import os
import shutil
import subprocess

# DashScope API 的 base64 data-URI 大小限制：10MB
MAX_BASE64_BYTES = 10_000_000
# WAV 16kHz 16bit 单声道：每秒字节数 = 16000 * 2 = 32000
WAV_BYTES_PER_SEC = 32_000


def _encode_audio_base64(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_audio_duration(audio_path):
    """获取音频时长（秒）"""
    from moviepy import AudioFileClip
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    audio.close()
    return duration


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def _split_audio_file(audio_path):
    """如果音频文件过大，使用 ffmpeg 切分为多段。

    每段时长自适应，确保 base64 编码后不超过 API 限制。
    """
    file_size = os.path.getsize(audio_path)
    b64_size = file_size * 4 / 3

    if b64_size <= MAX_BASE64_BYTES:
        return [audio_path]

    # base64 开销 4/3，每秒 WAV 有 WAV_BYTES_PER_SEC 字节
    # 每段最大秒数 = MAX_BASE64_BYTES / (4/3) / WAV_BYTES_PER_SEC
    max_chunk_s = int(MAX_BASE64_BYTES * 3 / 4 / WAV_BYTES_PER_SEC)  # ≈ 234s，取保守值

    duration = _get_audio_duration(audio_path)
    if duration <= max_chunk_s:
        return [audio_path]

    ffmpeg_bin = _get_ffmpeg()
    n_chunks = math.ceil(duration / max_chunk_s)
    chunks = []
    for i in range(n_chunks):
        start = i * max_chunk_s
        end = min((i + 1) * max_chunk_s, duration)
        duration_s = end - start
        chunk_path = audio_path.replace(".wav", f"_chunk{i:02d}.wav")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(audio_path),
            "-ss", f"{start:.3f}",
            "-t", f"{duration_s:.3f}",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        chunks.append(chunk_path)
    return chunks


def transcribe_audio(audio_path, api_key=None):
    """使用 qwen3-omni-flash 转录音频为文字。

    对于长音频会自动分段处理。
    Returns: 转录的文字
    """
    from dashscope import MultiModalConversation

    chunks = _split_audio_file(audio_path)
    all_text = []

    for chunk_path in chunks:
        audio_b64 = _encode_audio_base64(chunk_path)
        messages = [{
            "role": "user",
            "content": [
                {"audio": f"data:audio/wav;base64,{audio_b64}"},
                {"text": "请将这段音频中的语音内容完整转录为文字。只输出转录结果，不要添加任何额外说明。"}
            ]
        }]
        kwargs = {"model": "qwen3-omni-flash", "messages": messages}
        if api_key:
            kwargs["api_key"] = api_key
        response = MultiModalConversation.call(**kwargs)

        if response.status_code == 200:
            content = response.output["choices"][0]["message"]["content"]
            if isinstance(content, list):
                text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            else:
                text = str(content)
            all_text.append(text)
        else:
            all_text.append(f"[转录失败: {response.message}]")

        # 清理临时 chunk 文件
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except OSError:
                pass

    return "\n".join(all_text)
