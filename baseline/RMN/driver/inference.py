'''
Author: Zihao Yue
Date: 2023-12-03
Description:
    To address the request in issue #3, we provide an example inference script for the RMN model.
    To adapt the script, we add a new function 'self.infer_with_single_video_feature()' on line 171 of 'baseline/RMN/models/transformer.py'
    However, the script has not been tested and may contain bugs (but they should be easy to fix).

Input: video information (movie_id, timestamp)
Output: narration

Usage:
python inference.py \
--movie_id 6965768652251628068 \
--starttime 1000 \
--endtime 1010
'''

from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.abspath('..'))
import argparse
import json
import time
import pdb
import random
import numpy as np

import torch
import models.transformer
from models.transformer import DECODER
import readers.caption_data as dataset
import framework.run_utils
import framework.logbase
import torch.utils.data as data
from torch.nn import functional as F
import h5py

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def infer(_model, movie_id, starttime, endtime, data_path):

    movie_id = str(movie_id)
    clip_start = int(starttime) # e.g., 1000 (第1000秒)
    clip_end = int(endtime) # e.g., 2000 (第2000秒)

    # model_cfg = models.transformer.TransModelConfig()
    # model_cfg.load(data_path['model_config'])
    # _model = models.transformer.TransModel(model_cfg, _logger=None)
    # _model.load_checkpoint(data_path['ckpt'])

    # get video feature
    clip_movie = np.load(os.path.join(data_path['clip'], '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
    s3d_movie = np.load(os.path.join(data_path['s3d'], '%s-4.npz' % movie_id))['features'].astype(np.float32)
    frame_face_movie = np.load(os.path.join(data_path['frame_face'], '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
    clip_feature = torch.from_numpy(clip_movie)[clip_start:clip_end]
    s3d_feature = torch.from_numpy(s3d_movie)[clip_start:clip_end]
    frame_face_feature = torch.from_numpy(frame_face_movie)[clip_start:clip_end]
    features = torch.cat((clip_feature, s3d_feature, frame_face_feature), dim=1)
    features = F.normalize(features,dim=1).unsqueeze(0).cuda()
    feature_mask = torch.ones(1, features.size(0)).cuda()

    # get role name
    role_anno = json.load(open(data_path['role_anno']))
    movie_with_face_list = json.load(open(data_path['movie_with_face']))
    face_feature_h5 = h5py.File(data_path['meta_face'], 'r')
    stoi = json.load(open(data_path['stoi']))
    itos = json.load(open(data_path['itos']))
    role_name_list = []
    role_name_seq = []
    role_feature = np.zeros((10, 512), np.float32)

    if movie_id in role_anno and movie_id in movie_with_face_list:
      for role_id in role_anno[movie_id]:
        role_feature[len(role_name_list)] = np.array(face_feature_h5[role_id]['features'], np.float32)
        role_name = role_anno[movie_id][role_id]['rolename']
        role_name_sent = [stoi.get(w, 0) for w in role_name][:5] + [-1] * max(0, 5-len(role_name))
        role_name_list.append(role_name)
        role_name_seq.append(role_name_sent)
        if len(role_name_list) == 10:
          break
    
    role_feature = torch.from_numpy(role_feature).unsqueeze(0).cuda()

    role_name_seq += [[-1] * 5] * (10 - len(role_name_list))
    role_name_seq = np.array(role_name_seq)
        
    rolename_len = len(role_name_list)    

    output = _model.infer_with_single_video_feature(features, feature_mask, role_feature)

    # decoding
    def int2sent_role(intseq, rolename_list):
      ex_out = []
      for ind in intseq:
        ind = ind.item()
        if ind >= len(itos):
          role_idx = ind - len(itos)
          for role_token in rolename_list[role_idx]:
            if role_token != -1:
              ex_out.append(itos[str(role_token)])
            else:
              break
        else:
          ex_out.append(itos.get(str(ind), '<unk>'))
      return ex_out

    sents = int2sent_role(output.detach()[0], role_name_seq)
    # remove <bos>, <eos> and <pad>
    sents = [w for w in sents if w not in ['<sos>', '<eos>', '<pad>']]
    sentence = ''.join(sents)

    return sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_cfg_file')
    parser.add_argument('path_cfg_file')
    parser.add_argument('--movie_id', default=None)
    parser.add_argument('--starttime', default=None)
    parser.add_argument('--endtime', default=None)
    parser.add_argument('--ckpt', default=None)
    opts = parser.parse_args()

    set_seeds(12345)

    data_path = {
        'clip': 'baseline/RMN/data/feature/clip',
        's3d': 'baseline/RMN/data/feature/s3d',
        'frame_face': 'baseline/RMN/data/feature/frame_face',
        'stoi': 'baseline/RMN/data/vocab/c2id.json',
        'itos': 'baseline/RMN/data/vocab/id2c.json',
        'role_anno': 'baseline/RMN/data/metadata/meta_anno.json',
        'movie_with_face': 'baseline/RMN/data/metadata/movie_with_portraits.json',
        'meta_face': 'baseline/RMN/data/feature/protrait_face.hdf5',
        'model_config': 'baseline/RMN/results/model.json',
        'ckpt': 'path/to/your/ckpt/e.g./roleaware.90.th',
    }

    model_cfg = models.transformer.TransModelConfig()
    model_cfg.load(opts.model_cfg_file)
    _model = models.transformer.TransModel(model_cfg, _logger=None)

    if opts.ckpt is not None:
       data_path['ckpt'] = opts.ckpt
    result = _model.infer(_model, opts.movie_id, opts.starttime, opts.endtime, data_path)
    print(result)


if __name__ == '__main__':
    main()
