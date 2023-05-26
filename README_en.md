<div>
  <h2 align="center">
    <!-- <img src="https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/pigeon.png" width="40" /> -->
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

![Movie101 Dataset](https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/Movie101_dataset.png "Movie101 Dataset")

> [Chinese README](README.md)

Explore Movie101 in our paper [Movie101: A New Movie Understanding Benchmark](https://arxiv.org/abs/2305.12140). We are preparing a Chinese version of the paper for Chinese readers.

**Movie101 is a large-scale benchmark for AI Chinese movie understanding**, encompassing 101 movies. We collect the movies from the barrier-free channel on the [Xigua Video](https://www.ixigua.com/channel/barrier_free) platform, where standard movies are remastered with audio descriptions (ADs). Through automatic processes and manual correction, we obtain the ADs and actor lines from the raw videos. We also crawl rich meta information relevant to the movies. Eventually, Movie101 comprises 30,174 narration clips, totaling 92 hours.

The Movie101 benchmark includes two tasks: *Movie Clip Narrating (MCN)* and *Temporal Narration Grounding (TNG)*. 

- The MCN task requires the model to generate narration text based on movie videos to describe the current plot. In real-life movie narration, there are no timestamps to tell the model where to generate the narration. Therefore, to closely align with real-world application scenarios, MCN requires the model to generate narration when no actor is speaking. For this purpose, we reorganize the Movie101 dataset, merging the scattered narration clips between two dialogues into a longer clip, yielding a total of 14,109 long clips. Furthermore, to better evaluate the quality of model-generated narrations, we also designed a new metric specific to movie narrating, namely MNScore (Movie Narration Score).
- The TNG task requires the model to locate the start and end times of target clips in the movie based on a text description.

For both tasks, we demonstrate the performance of several existing baseline models and our new baselines provided in the paper.

This repository contains the Movie101 `dataset`, `baseline model codes`, and `evaluation scripts`. **More details can be found in the subfolders within this repository.** We hope that our proposed Movie101 benchmark can inspire more explorations into narrating and understanding movies.

If you find Movie101 helpful, please consider citing our paper:

```
@misc{yue2023movie101,
      title={Movie101: A New Movie Understanding Benchmark}, 
      author={Zihao Yue and Qi Zhang and Anwen Hu and Liang Zhang and Ziheng Wang and Qin Jin},
      year={2023},
      eprint={2305.12140},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```