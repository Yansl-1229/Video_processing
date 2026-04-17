"""从视频中提取音频，并按时间段切分为每段对应一个 PPT 页面"""
import os
import subprocess
from pathlib import Path


def extract_full_audio(video_path, output_path):
    """使用 moviepy 从视频中提取完整音频为 WAV 文件"""
    from moviepy import VideoFileClip
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(video_path))
    if clip.audio is None:
        clip.close()
        raise RuntimeError(f"视频没有音轨: {video_path}")
    clip.audio.write_audiofile(str(output_path))
    audio_duration = clip.audio.duration
    clip.close()
    return audio_duration


def split_audio_by_segments(audio_path, segment_time_ranges, output_dir, video_duration_s=None):
    """按全局时间范围切分音频，使用 ffmpeg subprocess 切分。

    Args:
        audio_path: 完整音频文件路径
        segment_time_ranges: list of (start_ms, end_ms)，end_ms 为 None 表示到视频末尾
        output_dir: 输出目录
        video_duration_s: 视频总时长(秒)，用于计算最后一个 segment 的结束时间

    Returns: list of (segment_index, audio_file_path)
    """
    import shutil

    # 获取音频总时长
    audio_duration_s = _get_audio_duration(str(audio_path))

    # 优先使用 imageio-ffmpeg 自带的 ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, (start_ms, end_ms) in enumerate(segment_time_ranges):
        if end_ms is None:
            end_ms = int(video_duration_s * 1000) if video_duration_s else start_ms + 1

        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0
        if start_s >= audio_duration_s:
            continue
        end_s = min(end_s, audio_duration_s)
        duration_s = end_s - start_s

        out_path = os.path.join(output_dir, f"segment_{idx:02d}.wav")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(audio_path),
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        results.append((idx, out_path))

    return results


def split_video_by_segments(video_path, segment_time_ranges, output_dir, video_duration_s=None):
    """按全局时间范围切分视频，使用 ffmpeg subprocess 切分。

    Args:
        video_path: 原始视频文件路径
        segment_time_ranges: list of (start_ms, end_ms)，end_ms 为 None 表示到视频末尾
        output_dir: 输出目录 (output/segments)
        video_duration_s: 视频总时长(秒)
    """
    import shutil

    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    for idx, (start_ms, end_ms) in enumerate(segment_time_ranges):
        if end_ms is None:
            end_ms = int(video_duration_s * 1000) if video_duration_s else start_ms + 1

        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0
        if video_duration_s and start_s >= video_duration_s:
            continue
        if video_duration_s:
            end_s = min(end_s, video_duration_s)
        duration_s = end_s - start_s

        seg_dir = os.path.join(output_dir, f"{idx:02d}")
        os.makedirs(seg_dir, exist_ok=True)
        out_path = os.path.join(seg_dir, "video.mp4")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _get_audio_duration(audio_path):
    """获取音频时长（秒）"""
    from moviepy import AudioFileClip
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    audio.close()
    return duration
