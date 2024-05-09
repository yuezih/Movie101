<div>
  <h2 align="center">
    🎬 Movie101 Benchmark
  </h2>
</div>

<p align="center">
    <a >
       <img alt="Issues" src="https://img.shields.io/github/issues/yuezih/Movie101?color=blueviolet" />
  	</a>
    <a >
       <img alt="Forks" src="https://img.shields.io/github/forks/yuezih/Movie101?color=orange" />
  	</a>
    <a >
       <img alt="Stars" src="https://img.shields.io/github/stars/yuezih/Movie101?color=ff69b4" />
  	</a>
    <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
    <br />
</p>


![Movie101 Dataset](assets/Movie101_dataset.png "Movie101 Dataset")


> [English README](README_en.md)  

有关Movie101的详细介绍请看论文: 
[[arXiv]](https://arxiv.org/abs/2305.12140) [[中文版论文]](assets/Movie101_zh.pdf)。

本仓库包含了Movie101的`数据集`、`基线模型代码`和`评测代码`。**更多介绍可以在仓库下的子文件夹中找到。** 

Movie101是一个大规模的AI中文电影理解基准，包含了101部电影。这些电影来自[西瓜视频](https://www.ixigua.com/channel/barrier_free)的无障碍影院，配备有音频描述（AD）。我们通过自动化流程和人工修正从原始视频中获取了音频描述和演员台词，并爬取了电影的相关信息。Movie101数据集包含了30,174个解说片段，总计92小时。

Movie101基准包含两个任务：电影片段解说 (Movie CLip Narrating, MCN) 和 时序解说定位 (Temporal Narration Grounding, TNG)。

- MCN任务要求模型根据电影视频生成解说文本来描述当前剧情。现实生活中的电影解说没有时间戳来告诉模型需要在哪里生成解说，因此，为了贴近现实应用场景，MCN要求模型在没有演员说话时生成解说。为此，我们重新组织了Movie101数据集，将两段对话之间的分散解说片段合并成一个更长的片段，一共得到14,109个的长片段。此外，为了更好地评估模型生成解说的质量，我们还设计了一个特定于电影解说的新指标MNScore（Movie Narration Score）。
- TNG任务要求模型根据一段文本描述在电影中定位目标片段的起止时间。



希望Movie101能够支持更多有关电影理解的探索。



如何引用Movie101：

```
@inproceedings{yue-etal-2023-movie101,
    title = "Movie101: A New Movie Understanding Benchmark",
    author={Zihao Yue and Qi Zhang and Anwen Hu and Liang Zhang and Ziheng Wang and Qin Jin},
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    url = "https://aclanthology.org/2023.acl-long.257",
    doi = "10.18653/v1/2023.acl-long.257",
    pages = "4669--4684",
}
```

```
@misc{yue2024movie101v2,
      title={Movie101v2: Improved Movie Narration Benchmark}, 
      author={Zihao Yue and Yepeng Zhang and Ziheng Wang and Qin Jin},
      year={2024},
      eprint={2404.13370},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

有关 EMNLP 2022 Findings 论文 [MovieUN: A Dataset for Movie Understanding and Narrating](https://aclanthology.org/2022.findings-emnlp.135/) 的疑问请查看我们的[FAQ: MovieUN](assets/FAQ_MovieUN/FAQ_MovieUN.md)。