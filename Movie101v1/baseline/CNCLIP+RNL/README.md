# CNCLIP+RNL

**Since the pipeline is complex, it's a lot of work to organize the code and I'm still working on that. If you urgently need it, please let me know.**

The goal of the Temporal Narration Grounding (TNG) task is to locate the start and end times of a relevant clip in an entire movie given a piece of narration text. The baseline provided in this paper divides this process into two stages. The first is a coarse-grained Global Shot Retrieval, which initially divides the movie into equal-length shot of 20 seconds each and then retrieves the most relevant shot. The second is a fine-grained Local Temporal Grounding, which involves precise localization around the retrieved shot. Specifically, the retrieved 20-second shot is expanded to an 200-second clip, and then temporal localization is performed.

# Global Shot Retrieval

This stage is achieved by CNCLIP, a Chinese image-text pre-training model, which we first need to make video retrieval-capable. Therefore, we constructed a temporary dataset Movie101-R (temp) for training a retrieval model, as described in the paper. Then, CNCLIP is fine-tuned on Movie101-R (temp). Finally, the fine-tuned model is used to retrieve the most relevant shot from a movie for a given narration text.

Since the implementation of our changes on CNCLIP is simple, however, the CNCLIP model itself requires a complex process of preparing data, we recommend referring to the [official CNCLIP code](https://github.com/OFA-Sys/Chinese-CLIP).

Based on that, we use CNCLIP's image encoder to independently encode 10 video frames and mean pool the frame features to a video feature, to replace the original image feature in CNCLIP. 

Our fine-tuned checkpoint is available at [TeraBox: cnclip-huge-movie101](https://terabox.com/s/1-5J0OTi9Cx7-uwiNqgMaNg). The folder contains a sequence of 3 files, as the size limit for a single file in TeraBox is 4GB, while our ckpt file sized 11GB. Use the Linux command to merge them:

```bash
cat cnclip-huge-movie101.pt.part-* > cnclip-huge-movie101.pt
```

# Local Temporal Grounding

This stage is achieved by RNL. Our implementation is based on [IA-Net](https://github.com/liudaizong/IA-Net).

## Preparation

### Download the Features

See `../RMN/README.md` for video feature downloading.

Text query features are available at [TeraBox: Movie101_text_features](https://terabox.com/s/1Nup2PBsew8IZM_FIAbN_mQ). Use this comman to merge the files and unzip:

```bash
cat grounding_text_feature.zip.part-* > grounding_text_feature.zip
unzip -o -d YOUR/PATH grounding_text_feature.zip
```

## Training

```bash
cd RNL
bash train.sh
```

## Evaluation

```bash
cd RNL
bash eval.sh
```