# Role-pointed Movie Narrator

**The code is being gradually completed.**

TODO:
- upload features for downloading

1. Download the video features

```bash
python ./data/feature/download.py
```
This script will download our prepared feature for the RMN model to `./data/feature`. The feature of each frame is concatenated by CLIP feature (512d), S3D feature (1024d) and 3 portrait features extracted by ArcFace (512d * 3). 

2. Train the model with config files `./results/model.json` and `./results/path.json`

```bash
cd driver
bash run.sh
```
