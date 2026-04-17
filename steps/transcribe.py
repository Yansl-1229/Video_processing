"""使用 qwen3.5-omni-flash 将音频转录为文字"""
import base64
import math
import os
import shutil
import subprocess

# WAV 16kHz 16bit 单声道：每秒字节数 = 16000 * 2 = 32000
WAV_BYTES_PER_SEC = 32_000
# 每段音频最大秒数，避免单段过长
MAX_CHUNK_SEC = 200


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
    """如果音频文件过长，使用 ffmpeg 切分为多段。"""
    duration = _get_audio_duration(audio_path)
    if duration <= MAX_CHUNK_SEC:
        return [audio_path]

    ffmpeg_bin = _get_ffmpeg()
    n_chunks = math.ceil(duration / MAX_CHUNK_SEC)
    chunks = []
    for i in range(n_chunks):
        start = i * MAX_CHUNK_SEC
        end = min((i + 1) * MAX_CHUNK_SEC, duration)
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
    """使用 qwen3.5-omni-flash 转录音频为文字。

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
        kwargs = {"model": "qwen3.5-omni-flash", "messages": messages}
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
