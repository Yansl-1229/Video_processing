# PPT 讲解视频处理系统

## 项目简介

PPT 讲解视频处理系统是一个自动化工具，专门用于处理 PPT 讲解类视频。该系统能够自动完成以下核心功能：

1. 按 PPT 页面自动切分视频片段
2. 提取每页 PPT 对应的音频并转录为文字
3. 对整个视频的讲解内容进行归纳总结，提取知识点

## 环境配置

### 环境要求

- **Conda 环境**: YOLO
- **Python 版本**: 3.x（具体版本根据实际环境而定）

### 安装步骤

1. 激活 Conda 环境：
```bash
conda activate YOLO
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 环境变量配置

需要设置 `DASHSCOPE_API_KEY` 环境变量，用于调用通义千问 API 进行语音转录和文本总结：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

## 使用说明

### 基本用法

```bash
conda activate YOLO
python main.py --video "视频.mp4" [--step 3] [--output ./output]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video` | string | `1.线性代数基础一.mp4` | 输入视频文件路径 |
| `--step` | float | 3 | 帧提取间隔（秒） |
| `--output` | string | `./output` | 输出目录 |

### 使用示例

```bash
# 使用默认参数处理示例视频
python main.py

# 处理自定义视频，使用0.5秒的帧提取间隔
python main.py --video "我的讲解视频.mp4" --step 0.5

# 指定输出目录
python main.py --video "课程视频.mp4" --output ./my_output
```

## 输出结构

系统运行后会在输出目录中生成以下结构：

```
output/
├── frames/              # 提取的原始帧
│   ├── frame_000000000.png
│   ├── frame_000003000.png
│   └── ...
├── filtered/            # 去重后的帧
├── segments/            # 按 PPT 页面切分的内容
│   ├── 00/
│   │   ├── frames/      # 该页的帧
│   │   └── transcript.txt  # 该页的讲解文字
│   ├── 01/
│   │   └── ...
│   └── ...
├── audio/               # 音频文件
│   ├── full.wav         # 完整音频
│   ├── segment_00.wav   # 第1页音频
│   ├── segment_01.wav   # 第2页音频
│   └── ...
├── adjacent_similarity.csv  # 相邻帧相似度数据
├── all_transcripts.txt      # 全部转录文本汇总
└── summary.md           # 课程总结和知识点
```

## 系统架构

### 处理流程

系统采用流水线式处理架构，共分为 5 个主要步骤：

#### Step 1: 流式帧提取 + CLIP 相似度计算
- **模块**: `steps/streaming_extract.py`
- **功能**: 单次遍历视频，在同一循环中完成帧提取和 CLIP 编码
- **输出**: `frames/` 目录和 `adjacent_similarity.csv`

#### Step 2: 去重 + PPT 页面切分
- **模块**: `steps/streaming_extract.py`
- **功能**: 读取相似度数据，去除重复帧，按低相似度切分为 PPT 页面，过滤短片段
- **输出**: `filtered/` 目录和 `segments/` 目录

#### Step 3: 音频/视频切分
- **模块**: `steps/extract_audio.py`
- **功能**: 提取完整音频，按 PPT 页面切分音频和视频
- **输出**: `audio/` 目录和 `segments/` 目录

#### Step 4: 语音转文字
- **模块**: `steps/transcribe.py`
- **功能**: 使用 qwen3-omni-flash 模型进行语音转录
- **输出**: 每页的 `transcript.txt` 和 `all_transcripts.txt`

#### Step 5: 生成课程总结
- **模块**: `steps/summarize.py`
- **功能**: 使用 qwen3-max 模型生成总结和知识点列表
- **输出**: `summary.md`

### 核心阈值参数

| 参数 | 值 | 用途 |
|------|-----|------|
| 帧提取间隔 | 3s（可配置） | 帧提取的时间间隔 |
| 去重阈值 | 0.99 | 相似度高于此值视为重复帧 |
| 切分阈值 | 0.62 | 相似度低于此值视为 PPT 翻页 |
| 切分最小间隔 | 15 帧 | 两个切分点之间的最小帧数 |
| 切分最小时长 | 60s | 短于此值的片段合并到相邻片段 |
| 段内去重阈值 | 0.97 | 每个 PPT 页面内部的去重阈值 |

### API 调用说明

- **语音转录**: 使用通义千问 `qwen3-omni-flash` 模型
- **文本总结**: 使用通义千问 `qwen3-max` 模型

## 性能优化

### 优化亮点

1. **单次视频遍历**: 原流程需要遍历视频 2 次（提取帧 → 读帧算相似度），优化后只遍历视频 1 次
2. **流式处理**: `extract_frames_with_similarity()` 在 OpenCV 读帧循环中直接做 CLIP 编码，省去写 PNG 再读回的 I/O 开销
3. **高效去重**: 使用 CLIP 相似度进行智能去重，保留关键帧

### 遗留文件

项目中保留了以下旧模块作为参考，不再被 `main.py` 调用：
- `steps/extract_frames.py`
- `steps/clip_similarity.py`
- `steps/deduplicate.py`
- `steps/split_segments.py`

## 文件说明

### 主要文件

- **main.py**: 系统主入口，串联所有处理步骤
- **CLAUDE.md**: 开发指南和项目说明
- **README.md**: 项目说明文档（本文件）

### 模块目录

- **steps/**: 核心处理模块目录
  - `__init__.py`: 模块初始化文件
  - `streaming_extract.py`: 流式帧提取和相似度计算（当前使用）
  - `extract_audio.py`: 音频提取和切分
  - `transcribe.py`: 语音转录
  - `summarize.py`: 文本总结
  - 其他 `.py` 文件：旧版本模块（参考用）

## 注意事项

1. **API Key**: 确保已正确设置 `DASHSCOPE_API_KEY` 环境变量，否则语音转录和总结功能将无法使用
2. **视频格式**: 建议使用常见的视频格式（如 MP4、AVI 等）
3. **磁盘空间**: 处理过程中会生成大量帧图片，确保有足够的磁盘空间
4. **处理时间**: 处理时间取决于视频长度、帧率和硬件性能

## 常见问题

### Q: 如何获取 DASHSCOPE_API_KEY？
A: 请访问阿里云通义千问开放平台（https://dashscope.aliyun.com/）注册账号并获取 API Key。

### Q: 处理速度很慢怎么办？
A: 可以尝试增大 `--step` 参数的值，减少提取的帧数，但这可能会影响切分精度。

### Q: PPT 页面切分不准确怎么办？
A: 可以调整 `steps/streaming_extract.py` 中的切分阈值参数（如 `split_threshold`、`min_segment_duration` 等）。

## 许可证

（根据实际情况填写许可证信息）

## 贡献指南

（根据实际情况填写贡献指南）
