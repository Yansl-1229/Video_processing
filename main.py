"""PPT 讲解视频处理系统 - 主入口

用法:
    conda activate YOLO
    python main.py --video "视频.mp4" [--step 0.5] [--output ./output]
"""
import argparse
import os
import sys
import time
from pathlib import Path

from steps.streaming_extract import extract_frames_with_similarity, deduplicate_and_segment
from steps.extract_audio import extract_full_audio, split_audio_by_segments, split_video_by_segments
from steps.transcribe import transcribe_audio
from steps.summarize import summarize_transcripts
from steps.match_ppt import process_filtered_frames, process_segments
from steps.ppt_to_images import pdf_to_images
from steps.generate_document import generate_document


def get_video_duration(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frame_count and frame_count > 0:
        return frame_count / fps
    return None


def main():
    parser = argparse.ArgumentParser(description="PPT 讲解视频处理系统")
    parser.add_argument("--video", required=False, default=str(Path("1.线性代数基础一.mp4")), help="输入视频文件路径，默认 1.线性代数基础一.mp4")
    parser.add_argument("--step", type=float, default=2, help="帧提取间隔（秒），默认 0.5")
    parser.add_argument("--output", default="./output", help="输出目录，默认 ./output")
    parser.add_argument("--ppt-pdf", default="./GAMES001-Lecture01.pdf", help="PPT PDF 文件路径，提供后自动拆页为图片用于匹配")
    parser.add_argument("--ppt-dir",  help="PPT 图片目录（若提供 --ppt-pdf 则自动设置）")
    parser.add_argument("--min-similarity", type=float, default=0.9,
                        help="最小相似度阈值，低于此值保留原始帧 (默认 0.0)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    filtered_dir = output_dir / "filtered"
    segments_dir = output_dir / "segments"
    audio_dir = output_dir / "audio"

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量，语音转录和总结将失败")

    # ========== Step 1: 流式提取帧 + 计算 CLIP 相似度 ==========
    print(f"\n{'='*60}")
    print(f"Step 1: 流式提取帧并计算 CLIP 相似度 (间隔 {args.step}s)")
    print(f"{'='*60}")
    t0 = time.time()
    frame_paths, csv_path, n_pairs = extract_frames_with_similarity(
        str(video_path), str(output_dir), step_seconds=args.step
    )
    print(f"  提取了 {len(frame_paths)} 帧, 计算了 {n_pairs} 对相似度 ({time.time()-t0:.1f}s)")

    # ========== Step 2: 去重 + PPT 页面切分 ==========
    print(f"\n{'='*60}")
    print("Step 2: 去重 + 按 PPT 页面切分")
    print(f"{'='*60}")
    t0 = time.time()
    segments, segment_time_ranges = deduplicate_and_segment(
        str(csv_path), str(frames_dir), str(filtered_dir), str(segments_dir), dup_threshold=0.8, threshold_low=0.6, min_gap=15, min_duration_s=60
    )
    print(f"  切分为 {len(segments)} 个页面 ({time.time()-t0:.1f}s)")
    for i, seg in enumerate(segments):
        print(f"    第 {i+1} 页: {len(seg)} 帧")

    # ========== Step 2.5: PPT PDF 拆页 ==========
    if args.ppt_pdf:
        pdf_path = Path(args.ppt_pdf)
        if pdf_path.exists():
            print(f"\n{'='*60}")
            print(f"Step 2.5a: 将 PPT PDF 拆页为图片")
            print(f"{'='*60}")
            t0 = time.time()
            pdf_to_images(str(pdf_path))
            ppt_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
            print(f"  PDF 拆页完成，图片保存在: {ppt_dir} ({time.time()-t0:.1f}s)")
        else:
            print(f"\n警告: PPT PDF 文件不存在: {pdf_path}，跳过")
            ppt_dir = Path(args.ppt_dir)
    else:
        ppt_dir = Path(args.ppt_dir)

    # ========== Step 2.5b: 帧图片替换为PPT图片 ==========
    ppt_dir = Path(ppt_dir)
    if ppt_dir.exists():
        print(f"\n{'='*60}")
        print(f"Step 2.5: 将帧图片替换为最相似的PPT图片")
        print(f"{'='*60}")
        t0 = time.time()
        # 处理 filtered 目录
        process_filtered_frames(str(output_dir), str(ppt_dir), min_similarity=args.min_similarity)
        # 处理 segments 目录
        process_segments(str(output_dir), str(ppt_dir), min_similarity=args.min_similarity)
        print(f"  PPT 匹配完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"\n警告: PPT 目录不存在，跳过 PPT 匹配: {ppt_dir}")

    # ========== Step 3: 提取音频并切分 ==========
    print(f"\n{'='*60}")
    print("Step 3: 提取音频并按页面切分")
    print(f"{'='*60}")
    t0 = time.time()
    full_audio_path = audio_dir / "full.wav"
    video_duration = get_video_duration(video_path)
    full_audio_duration = extract_full_audio(video_path, full_audio_path)
    print(f"  完整音频: {full_audio_duration:.1f}s ({time.time()-t0:.1f}s)")

    t0 = time.time()
    audio_segments = split_audio_by_segments(
        str(full_audio_path), segment_time_ranges, str(audio_dir), video_duration_s=video_duration
    )
    print(f"  切分为 {len(audio_segments)} 段音频 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    split_video_by_segments(
        str(video_path), segment_time_ranges, str(segments_dir), video_duration_s=video_duration
    )
    print(f"  切分为 {len(segments)} 段视频 ({time.time()-t0:.1f}s)")

    # ========== Step 4: 语音转文字 ==========
    print(f"\n{'='*60}")
    print("Step 4: 语音转文字 (qwen3.5-omni-flash)")
    print(f"{'='*60}")
    transcripts = []
    for idx, audio_path in audio_segments:
        t0 = time.time()
        print(f"  转录第 {idx+1} 页音频...", end=" ", flush=True)
        text = transcribe_audio(audio_path, api_key=api_key)
        transcripts.append((idx, text))
        # 保存每页转录结果
        seg_dir = segments_dir / f"{idx:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        (seg_dir / "transcript.txt").write_text(text, encoding="utf-8")
        print(f"({time.time()-t0:.1f}s)")

    # 保存所有转录结果到一个文件
    all_transcripts_path = output_dir / "all_transcripts.txt"
    with open(all_transcripts_path, "w", encoding="utf-8") as f:
        for idx, text in transcripts:
            f.write(f"=== 第 {idx+1} 页 ===\n")
            f.write(text)
            f.write("\n\n")
    print(f"\n  所有转录结果已保存到: {all_transcripts_path}")

    # ========== Step 5: 整体总结 ==========
    print(f"\n{'='*60}")
    print("Step 5: 生成课程总结和知识点 (qwen3-max)")
    print(f"{'='*60}")
    t0 = time.time()
    summary = summarize_transcripts(transcripts, api_key=api_key)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"  总结已保存到: {summary_path} ({time.time()-t0:.1f}s)")

    # ========== Step 6: 逐页提取知识点 ==========
    print(f"\n{'='*60}")
    print("Step 6: 逐页提取知识点 (qwen-plus)")
    print(f"{'='*60}")
    t0 = time.time()
    from steps.segment_summarize import summarize_segments
    summarize_segments(segments_dir, api_key=api_key)
    print(f"  逐页知识点提取完成 ({time.time()-t0:.1f}s)")

    # ========== Step 7: 生成多模态文档 ==========
    print(f"\n{'='*60}")
    print("Step 7: 生成多模态课程文档")
    print(f"{'='*60}")
    t0 = time.time()
    doc_path = generate_document(
        str(output_dir), segments_dir=str(segments_dir), segments=segments
    )
    print(f"  多模态文档已生成: {doc_path} ({time.time()-t0:.1f}s)")

    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"{'='*60}")
    print(f"  帧目录:      {frames_dir}")
    print(f"  过滤帧:      {filtered_dir}")
    print(f"  PPT 页面:    {segments_dir}")
    print(f"  音频:        {audio_dir}")
    print(f"  全部转录:    {all_transcripts_path}")
    print(f"  课程总结:    {summary_path}")
    print(f"  逐页知识点:  segments/*/segment_summary.md")
    print(f"  多模态文档:  {doc_path}")


if __name__ == "__main__":
    main()
