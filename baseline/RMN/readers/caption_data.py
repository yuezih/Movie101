from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import math
import torch.utils.data
import h5py
import torch.nn.functional as F

UNK, PAD, BOS, EOS = 0, 1, 2, 3


class CaptionDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_root, cap_file, word2int, int2word,
    max_words_in_sent=150, is_train=False, _logger=None):
    super(CaptionDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.ref_captions = json.load(open(cap_file)) # gt
    self.names = list(self.ref_captions.keys())
    
    self.data_root = os.path.join(ft_root.split('/')[:-1])
    self.role_anno = json.load(open(os.path.join(self.data_root, 'metadata/meta_anno.json')))
    self.face_feature_h5 = h5py.File(os.path.join(self.data_root, 'metadata/portrait_face.hdf5'), 'r')
    self.movie_with_face_list = json.load(open(os.path.join(self.data_root, 'metadata/movie_with_portraits.json')))
    # self.movie2tags = json.load(open('metadata/movie_tag_anno.json'))
    # self.tag2id = json.load(open('metadata/tag_vocab.json'))
    
    self.stoi = json.load(open(word2int))
    self.itos = json.load(open(int2word))
    self.ft_root = ft_root
    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train


  def temporal_pad_or_trim_feature(self, ft, max_len, transpose=False, average=False):
    length, dim_ft = ft.shape
    # pad
    if length <= max_len:
      ft_new = np.zeros((max_len, dim_ft), np.float32)
      ft_new[:length] = ft
    # trim
    else:
      if average:
        indices = np.round(np.linspace(0, length, max_len+1)).astype(np.int32)
        ft_new = [np.mean(ft[indices[i]: indices[i+1]], axis=0) for i in range(max_len)]
        ft_new = np.array(ft_new, np.float32)
      else:
        indices = np.round(np.linspace(0, length - 1, max_len)).astype(np.int32)
        ft_new = ft[indices]
    if transpose:
      ft_new = ft_new.transpose()
    return ft_new

  def pad_sent(self, x):
    max_len = self.max_words_in_sent
    padded = [BOS] + x[:max_len] + [EOS] + [PAD] * max(0, max_len - len(x))
    length = 1+min(len(x), max_len)+1
    return np.array(padded), length

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>')
    batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def int2sent_role(self, batch, rolename):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
      rolename = rolename.tolist()
    batch_out = []
    for ex in range(len(batch)):
      ex_out = []
      rolename_list = rolename[ex]
      for ind in batch[ex]:
        if ind >= len(self.itos):
          role_idx = ind - len(self.itos)
          for role_token in rolename_list[role_idx]:
            if role_token != -1:
              ex_out.append(self.itos[str(role_token)])
            else:
              break
        else:
          ex_out.append(self.itos.get(str(ind), '<unk>'))
      batch_out.append(ex_out)
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence

    batch_out = [trim(ex, '<eos>') for ex in batch_out] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>')

    batch_out = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch_out]
    return batch_out


  def __len__(self):
    # if self.is_train:
    #   return len(self.captions)
    # else:
    #   return len(self.names)
    return len(self.ref_captions)

  def __getitem__(self, idx):
    outs = {}

    name = self.names[idx]
    example = self.ref_captions[name]

    start = int(example["timestamps"][0][0])
    end = int(example["timestamps"][0][1])
    sentence = example["sentences"][0]
    movie_id = example["movie_id"]

    max_v_l = 40

    raw_feat = self.get_video_features(movie_id, start, end)[:max_v_l]

    # tags = self.movie2tags[movie_id]
    # tags = [self.tag2id[t] for t in tags]
    # tags_len = len(tags)
    # tags += [0] * (4 - tags_len)
    # tags = np.array(tags[:4])

    feat_len = raw_feat.shape[0]
    feat_dim = raw_feat.shape[1]

    video_feature = np.zeros((max_v_l, feat_dim), np.float32)  # only video features and padding
    video_feature[:feat_len] = raw_feat[:]

    # rolename, max_len = 10
    role_name_list = []
    role_name_seq = []
    role_feature = np.zeros((10, 512), np.float32)

    if movie_id in self.role_anno and movie_id in self.movie_with_face_list:
      for role_id in self.role_anno[movie_id]:
        role_feature[len(role_name_list)] = np.array(self.face_feature_h5[role_id]['features'], np.float32)
        role_name = self.role_anno[movie_id][role_id]['rolename']
        role_name_sent = [self.stoi.get(w, UNK) for w in role_name][:5] + [-1] * max(0, 5-len(role_name))
        role_name_list.append(role_name)
        role_name_seq.append(role_name_sent)
        if len(role_name_list) == 10:
          break
    
    role_name_seq += [[-1] * 5] * (10 - len(role_name_list))
    role_name_seq = np.array(role_name_seq)
        
    rolename_len = len(role_name_list)

    outs['ft_len'] = feat_len
    outs['img_ft'] = video_feature
    outs['name'] = name

    outs['rolename_seq'] = role_name_seq
    outs['rolename_len'] = rolename_len
    outs['role_face'] = role_feature

    # outs['tags'] = tags
    # outs['tags_len'] = tags_len

    if self.is_train:
      outs['ref_sents'] = sentence
      sent_id = []
      i = 0
      while i < len(sentence):
        if sentence[i] == '@':
          sent_id.append(int(sentence[i+1])+4177)
          i += 1
        else:
          sent_id.append(self.stoi.get(sentence[i], UNK))
        i += 1
      # sent_id, sent_len = self.pad_sent(self.sent2int(sentence))
      padded, padded_len = self.pad_sent(sent_id)
      outs['caption_ids'] = padded
      outs['id_len'] = padded_len
    return outs

  def get_video_features(self, movie_id, clip_start, clip_end):
      
      clip_movie = np.load(os.path.join('../data/feature', 'clip', '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
      s3d_movie = np.load(os.path.join('../data/feature', 's3d', '%s-4.npz' % movie_id))['features'].astype(np.float32)
      clip_frames = torch.from_numpy(clip_movie)
      s3d_frames = torch.from_numpy(s3d_movie)

      clip_feature = clip_frames[clip_start:clip_end]
      s3d_feature = s3d_frames[clip_start:clip_end]
      frame_face = self.movie_frame_face[movie_id][clip_start:clip_end]

      features = torch.cat((clip_feature, s3d_feature, frame_face), dim=1)
      features = F.normalize(features,dim=1)
      
      # feat_name = movie_id + '_' + str(clip_start) + '_' + str(clip_end) + '.npy'
      # features = np.load(os.path.join(self.ft_root, 'clip_s3d', feat_name))
      # features = torch.from_numpy(features)
      return features