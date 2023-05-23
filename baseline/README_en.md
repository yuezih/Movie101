# Movie101 Baselines

This directory contains the official baseline codes for the Movie101 benchmark, including the RMN model for the Movie Clip Narration (MCN) task and the CNCLIP+RNL framework for the temporal narration localization task.

![Movie101 baselines](https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/Movie101_baselines.png "Movie101 baselines")

## Movie Clip Narrating (MCN)

For this task, we provide the code for the RMN model from the paper (located in `./RMN`). This model is based on the [OVP](https:/github.com/syuqings/video-paragraph) model, with the addition of genre encoding and a pointer network for generating character names. The other two baseline models in the paper are the [Vanilla Transformer](https://github.com/jayleicn/recurrent-transformer) and [OVP](https:/github.com/syuqings/video-paragraph).

| Model | EMScore | BERTScore | RoleF1 | MNScore |
|-------|---------|-----------|--------|---------|
| VT    | 0.153   | 0.150     | 0      | 12.55   |
| OVP   | 0.155   | 0.159     | 0      | 13.18   |
| RMN   | 0.154   | 0.188     | 0.238  | 19.07   |

## Temporal Narration Grounding (TNG)

For this task, we provide the code for the CNCLIP+RNL framework from the paper (located in `./CNCLIP+RNL`). This first divides the movie into equal-length 20s segments, then retrieves the most relevant segment based on a text query (Global Shot Retrieval), which is then expanded to 200s for precise localization (Local Temporal Grounding) using the RNL model. The retrieval model uses a fine-tuned [CNCLIP](https://github.com/OFA-Sys/Chinese-CLIP), while the RNL model is based on [IA-Net](https://github.com/liudaizong/IA-Net). For Local Temporal Grounding, an additional baseline model in the paper is [2D-TAN](https://github.com/chenjoya/2dtan).

| Model | Rank@1, IoU0.3 | Rank@1, IoU0.5 | Rank@5, IoU0.3 | Rank@5, IoU0.5 |
|-------|----------------|----------------|----------------|----------------|
| 2D-TAN| 28.85         | 18.60          | 52.17          | 43.82          |
| IA-Net| 25.16         | 17.98          | 57.11          | 42.68          |
| RNL   | 27.54         | 20.22          | 59.52          | 45.69          |