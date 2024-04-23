# Role-pointed Movie Narrator

> Our code implementation is tedious, if you encounter problems when running our code, please feel free to raise issues or contact us directly, and we are willing to provide technical help.

## Preparation

### Download the video features  

We provide our extracted feature within a `.zip` file at [TeraBox: Movie101_features.zip](https://terabox.com/s/1iYf154IsDKCYDEw72SWfUw). This link allows you to download the file directly from the TeraBox client. 
<!-- If you want to download on Linux, some open-source tools (e.g., [TeraBox-Downloader](https://github.com/snoofox/TeraBox-Downloader)) may be helpful.   -->

Download and unzip the feature file to `./data/feature`. It contains:
- `./clip/`: CLIP feature (512d) for each frame
- `./s3d/`: S3D feature (1024d) for each frame
- `./frame_face/`: 3 face features (512d * 3) extracted from each frame
- `protrait_face.hdf5`: a dict that maps role_id (see `./data/metadata/meta_anno.json`) to the corresponding role portrait feature (512d)

### Narration Annotation

For data augmentation, we add fine-grained clips from Movie101-raw to the training set of Movie101-N (`./data/narration_role/train_merge.json`). 

## Training

Train the model with config files `./results/model.json` and `./results/path.json`

```bash
cd driver
bash run.sh
```

## Evaluation

Replace the resume file (model ckpt) in the `./driver/eval.sh`.

```bash
cd driver
bash eval.sh
```

## Inference

We now (2023.12.03) provide an example script for single video clip inference, see `./driver/inference.py`.

```bash
python inference.py \
--movie_id 6965768652251628068 \
--starttime 1000 \
--endtime 1010
```

Our trained model checkpoint can be downloaded from [TeraBox: model-ckpt](https://terabox.com/s/1FQ562_B2U_11F3X1ND2Iyw).