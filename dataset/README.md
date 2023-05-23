# Movie101 数据集

![Movie101 Dataset](https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/Movie101_dataset.png "Movie101 Dataset")

Movie101包含101部电影的视频（从[西瓜视频](https://www.ixigua.com/channel/barrier_free)爬取），以及带有时间戳的解说文本（由ASR获取并人工修正）和演员台词（由OCR获取）。

根据电影片段解说 (MCN) 任务和时序解说定位 (TNG) 任务的要求，我们构建了两个版本：
- Movie101-N，包含段落解说（两段演员对话之间的解说被合并成段落），用于narrating任务；
- Movie101-G，用于grounding任务。

## Movie101-raw

### annotation

- lines.json  
每部电影的演员台词及其时间戳。

- narration_clips.json  
未合并的解说文本和时间戳。

- narration_paragraphs.json  
基于演员对话合并的解说文本和时间戳。

- public_split.json  
训练集、验证集和测试集划分。

- metadata.json  
每部电影的元信息，包括类别、简介、演员表等。

### videos

可以使用`Movie101-raw/scripts`中提供的脚本下载电影视频，默认下载到`Movie101-raw/videos`。（感谢Ziheng Wang提供脚本）

```bash
python ./scripts/video_download.py
```

## Movie101-N & Movie101-G

划分后的各集合数据规模：

| 数据集 | 训练 | 验证 | 测试 |
|---------|-------|------------|------|
| Movie101-N | 11,325 | 1,416 | 1,368 |
| Movie101-G | 24,508 | 2,768 | 2,898 |