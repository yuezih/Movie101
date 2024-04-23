<div>
  <h2 align="center">
    Movie101
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

![Movie101 Dataset](assets/example.png "Movie101 Dataset")

## About

**Movie101** is a large-scale benchmark providing video-aligned movie narration texts to facilitate research on AI movie understanding, like narration generation and temporal grounding. **Movie101v2** builds upon Movie101 with bilingual narrations, increased scale, and improved data quality.

Find more details in our papers:

> [Movie101v2: Improved Movie Narration Benchmark](https://arxiv.org/abs/2404.13370) 🔥  
> [Movie101: A New Movie Understanding Benchmark](https://arxiv.org/abs/2305.12140) (ACL 2023)

To access the Movie101/Movie101v2 data:

- **Annotation**: available at the corresponding folders.  
- **Video**: visit our [Movie101 homepage](https://movie101.github.io).

This repository also contains the evaluation scripts and baseline codes.

If you find Movie101 useful, please consider citing our papers:

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