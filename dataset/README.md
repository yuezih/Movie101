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

电影及url列表：`video_src.json`

我们曾尝试提供一个从原始网站爬取视频的脚本，但爬取过程非常复杂，难以保证其便捷性。因此，我们决定仿照[LSMDC电影数据集](https://sites.google.com/site/describingmovies)的做法，允许研究者通过签署同意书来直接从我们这里访问数据。如果您需要访问我们数据集的视频，请按照以下步骤操作：

1. 点击下载[同意书](https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/AccessMovie101.pdf)；
2. 签署该同意书（需要同时签署中文版和英文版）；
3. 向`yzihao@ruc.edu.cn`发送邮件，邮件需包含以下内容：
    - 您的姓名、单位，以及您的导师的姓名
    - 附件：已签署的同意书

我们将在收到邮件后尽快回复您（尽量当天回复），告知您如何访问数据。

## Movie101-N & Movie101-G

划分后的各集合数据规模：

| 数据集 | 训练 | 验证 | 测试 |
|---------|-------|------------|------|
| Movie101-N | 11,325 | 1,416 | 1,368 |
| Movie101-G | 24,508 | 2,768 | 2,898 |