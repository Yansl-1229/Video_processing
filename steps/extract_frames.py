"""从视频中按固定间隔提取帧"""
from pathlib import Path
import cv2


def extract_frames(video_path: Path, output_dir: Path, step_seconds: float = 0.5,
                   image_format: str = "png", jpeg_quality: int = 95):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = None
    if fps and fps > 0 and frame_count and frame_count > 0:
        duration_ms = int((frame_count / fps) * 1000)
    step_ms = int(step_seconds * 1000)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t = 0
    if duration_ms is None:
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, t)
            ret, frame = cap.read()
            if not ret:
                break
            name = f"frame_{t:09d}.{image_format}"
            path = output_dir / name
            if image_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            else:
                cv2.imwrite(str(path), frame)
            paths.append(str(path))
            t += step_ms
    else:
        last_t = (duration_ms // step_ms) * step_ms
        while t <= last_t:
            cap.set(cv2.CAP_PROP_POS_MSEC, t)
            ret, frame = cap.read()
            if ret:
                name = f"frame_{t:09d}.{image_format}"
                path = output_dir / name
                if image_format.lower() in ("jpg", "jpeg"):
                    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                else:
                    cv2.imwrite(str(path), frame)
                paths.append(str(path))
            t += step_ms
    cap.release()
    list_file = output_dir / "images.txt"
    list_file.write_text("\n".join(paths), encoding="utf-8")
    return paths
