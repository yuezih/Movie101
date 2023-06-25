# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats', 
        type=str, 
        required=True,
        help="Specify the path of image features."
    )  
    parser.add_argument(
        '--text-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )      
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help="Specify the k value of top-k predictions."
    )   
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=32768,
        help="Specify the image-side batch size when computing the inner products, default to 8192"
    )  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    with open(args.image_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    image_feature = torch.from_numpy(image_feats_array).cuda() # (N, L, D)

    print("Begin to load text features...")
    text_ids = []
    text_feats = []
    text_masks = []
    with open(args.text_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_ids.append(obj['text_id'])
            text_feats.append(obj['feature'])
            text_masks.append(obj['mask'])
    text_feat_array = np.array(text_feats, dtype=np.float32)
    text_feature = torch.from_numpy(text_feat_array).cuda() # (N, L, D)

    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                embeddings of reference sentences, BxKxd,
                B: batch size, K: longest length, d: bert dimenison.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                reference sentences.
        - :param: `hyp_embedding` (torch.Tensor):
                embeddings of candidate sentences, BxKxd,
                B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                candidate sentences.
    """
    # ref_embedding and hyp_embedding are aleady L2-normalized.
    # build text_masks for bert attention
    F1_list = []
    batch_size = image_feature.size(0)
    for i in range(batch_size):
        sentence_feat = text_feature[i] # (L, D)
        image_feat = image_feature[i] # (N, D)
        sim = torch.einsum('ld,md->lm', sentence_feat, image_feat) # (L, N)
        sentence_len = int(sum(text_masks[i]))
        sim = sim[:sentence_len, :] # (L, N)
        image_len = image_feat.size(0) # (N)

        word_precision, matched_indices = sim.max(dim=1)
        word_recall = sim.max(dim=0)[0]

        P = (word_precision).sum(dim=0)/sentence_len
        R = word_recall.sum(dim=0)/image_len
        F = 2 * P * R / (P + R)
        F1_list.append(F)
    F1s = [float(f) for f in F1_list]
    print(f"Mean F1 score: {np.mean(F1s)}")