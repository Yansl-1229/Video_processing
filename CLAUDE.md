# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPT 讲解视频处理系统。输入一份 PPT 讲解视频，自动完成：
1. 按 PPT 页面切分视频片段，提取每页对应的音频并转录为文字
2. 对整个视频的讲解内容进行归纳总结，提取知识点

核心入口：`main.py`，串联 `steps/` 下的 5 个处理步骤。

## Environment & Dependencies

项目使用 conda 环境 `YOLO`，所有脚本需在该环境下运行：

```bash
conda activate YOLO
pip install -r requirements.txt
```

需要设置 `DASHSCOPE_API_KEY` 环境变量（语音转录和总结均需调用 DashScope API）。

## Usage

```bash
conda activate YOLO
python main.py --video "视频.mp4" [--step 0.5] [--output ./output]
```

参数：
- `--video`：输入视频路径（必填）
- `--step`：帧提取间隔秒数，默认 0.5
- `--output`：输出目录，默认 `./output`

## Output Structure

```
output/
  frames/              # 提取的帧
  filtered/            # 去重后的帧
  segments/            # 按 PPT 页面切分
    00/
      frames/          # 该页的帧
      transcript.txt   # 该页的讲解文字
    01/
      ...
  audio/               # 音频文件
    full.wav           # 完整音频
    segment_00.wav     # 每页音频
  adjacent_similarity.csv
  all_transcripts.txt  # 全部转录文本
  summary.md           # 课程总结 + 知识点
```

## Architecture

Pipeline 是一系列数据转换步骤，每步读写文件系统：

1. **流式帧提取 + CLIP 相似度** (`steps/streaming_extract.py`) — 单次遍历视频，帧提取和 CLIP 编码在同一循环中完成（不写 PNG 再读回），输出 `frames/` 和 `adjacent_similarity.csv`
2. **去重 + PPT 页面切分** (`steps/streaming_extract.py`) — 读取 CSV，去除重复帧（>0.99），按低相似度（<0.62）切分为 PPT 页面，过滤短片段（<60s 合并），段内再去重，输出 `filtered/` 和 `segments/`
3. **音频/视频切分** (`steps/extract_audio.py`) — moviepy 提取完整音频，ffmpeg 切分音频和视频
4. **语音转录** (`steps/transcribe.py`) — `qwen3.5-omni-flash`，OpenAI 兼容接口，base64 编码音频；长音频自动分段
5. **总结** (`steps/summarize.py`) — `qwen3-max`，拼接所有转录文本，生成总结和知识点列表

### 性能优化说明

原流程 Step 1-4 需要遍历视频 2 次（提取帧 → 读帧算相似度），优化后只遍历视频 1 次。
`extract_frames_with_similarity()` 在 OpenCV 读帧循环中直接做 CLIP 编码，省去写 PNG 再读回的 I/O。
旧模块 `extract_frames.py`、`clip_similarity.py`、`deduplicate.py`、`split_segments.py` 保留作参考，不再被 main.py 调用。

### API 调用方式

- 语音转录：`openai.OpenAI` 兼容接口，`model="qwen3.5-omni-flash"`，`modalities=["text"]`，音频以 base64 传入 `input_audio`
- 文本总结：`dashscope.Generation.call(model="qwen3-max")`

## Legacy Files

项目中保留了旧的化学实验 SOP 分析脚本（`1_*` 到 `6_*`、`detection_track.py`、`weights/`），这些是重构前的代码，当前项目不再使用。

## Key Thresholds

| 参数 | 值 | 用途 |
|------|-----|------|
| 帧提取间隔 | 0.5s | 默认值，PPT 讲解场景适用 |
| 去重阈值 | 0.99 | 相似度高于此值视为重复帧 |
| 切分阈值 | 0.62 | 相似度低于此值视为 PPT 翻页 |
| 切分最小间隔 | 15 | 两个切分点之间的最小行数 |
| 切分最小时长 | 60s | 短于此值的片段合并到相邻片段 |
| 段内去重阈值 | 0.97 | 每个 PPT 页面内部的去重阈值 |
