import collections
import logging
import os

import torch

from torch.utils.data import DataLoader

import criteria
from dataloaders.clip_loader import get_dataset
from models_.gcn_final import Model
# from models_.latent_attention import Model
from optimizer.adam_optimizer import AdamOptimizer
from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils import load_word2vec, AverageMeter, TimeMeter, CountMeter
import numpy as np
import pdb
import json
from tqdm import tqdm

def save_json(data, filename):
    with open(filename, "w",encoding='utf-8') as f:
        return json.dump(data, f,ensure_ascii=False)

gt_anno = json.load(open('/data5/yzh/MovieUN_v2/IANET/code/gt_times.json','r'))
pred_anno = json.load(open('/data5/yzh/MovieUN_v2/IANET/code/ia_net_preds.json','r'))

meters_5 = collections.defaultdict(lambda: CountMeter())
for idx in range(len(gt_anno)):
    gt_windows = np.array(gt_anno[idx]['gt_times'])
    predict_windows = np.array(pred_anno[idx]['predicted_times'])
    predict_score = np.array(pred_anno[idx]['scores'])
    topn_IoU_matric = criteria.compute_IoU_recall(predict_score, predict_windows, gt_windows)
    meters_5['mIoU'].update(topn_IoU_matric, 1)

IoU_threshs = [0.1, 0.3, 0.5, 0.7]
top_n_list = [1,5]
topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
res_json = {}
for i in range(2):
    for j in range(4):
        print('{}, {:.4f}'.format('IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j]), topn_IoU_matric[i,j]/count), end=' | ')
        res_json['IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j])] = topn_IoU_matric[i,j]/count
meters_5['mIoU'].reset()
save_json(res_json, '/data5/yzh/MovieUN_v2/IANET/code/output/grounding/results.json')
print()