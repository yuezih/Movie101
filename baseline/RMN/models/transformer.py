from __future__ import print_function
from __future__ import division

import os
import numpy as np
import collections
import json
import h5py
import random
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import framework.configbase
import framework.modelbase
import modules.transformer
import modules.video_encoder
import metrics.evaluation
import metrics.criterion

DECODER = 'transformer'
VISENC = 'vis_enc'


class TransModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super(TransModelConfig, self).__init__()

  def load(self, cfg_file):
    with open(cfg_file) as f:
      data = json.load(f)
    for key, value in data.items():
      if key != 'subcfgs':
        setattr(self, key, value)
    # initialize config objects
    for subname, subcfg_type in self.subcfg_types.items():
      if subname == DECODER:
        self.subcfgs[subname] = modules.transformer.__dict__[subcfg_type]()
      elif subname == VISENC:
        self.subcfgs[subname] = modules.video_encoder.__dict__[subcfg_type]()
      self.subcfgs[subname].load_from_dict(data['subcfgs'][subname])

      
class TransModel(framework.modelbase.ModelBase):
  def build_submods(self):
    submods = {}
    submods[DECODER] = modules.transformer.Transformer(self.config.subcfgs[DECODER])
    submods[VISENC] = modules.video_encoder.VideoEncoder(self.config.subcfgs[VISENC])
    return submods

  def build_loss(self):
    xe_loss = metrics.criterion.UnlikelihoodLoss(0.1, 4187 ,1)
    rl_loss = metrics.criterion.RewardLoss(self.config.subcfgs[DECODER].document_freq)
    reconstruct = metrics.criterion.L2Loss()
    sparsity = metrics.criterion.L1Loss()
    return (xe_loss, rl_loss, reconstruct, sparsity)

  def forward_loss(self, batch_data, TRG, step=None):
    # role name tokens
    # rolename = batch_data['rolename_seq']
    rolename_len = batch_data['rolename_len'].cuda()
    rolename_mask = self.attn_mask(rolename_len, max_len=10).unsqueeze(-2)
    # role face feature
    roleface = batch_data['role_face'].cuda()
    # video feature (with face, (feat_len, 1536+512*3))
    img_fts = torch.FloatTensor(batch_data['img_ft']).cuda() # (batch, seq_len, dim)
    ft_len = torch.LongTensor(batch_data['ft_len']).cuda() # (batch)
    img_fts = img_fts[:,:max(ft_len)] # (batch, seq_len, dim)
    # target sentence
    trg = torch.LongTensor(batch_data['caption_ids']).cuda() # (batch, seq_len)
    trg = trg[:,:max(batch_data['id_len'])]
    # # tags
    # tags = batch_data['tags'].cuda()
    # tags_len = batch_data['tags_len'].cuda()
    # tags_mask = self.attn_mask(tags_len, max_len=4).unsqueeze(-2)


    trg_input = trg[:, :-1]
    src_mask, trg_mask = self.create_masks(ft_len, img_fts.size(1), trg_input)
    outputs, key_enc, select = self.submods[DECODER](img_fts, trg_input, src_mask, trg_mask, roleface, rolename_mask)
    outputs = nn.LogSoftmax(dim=-1)(outputs)
    # outputs = torch.log(outputs)

    ys = trg[:, 1:].contiguous().view(-1)
    norm = trg[:, 1:].ne(1).sum().item()
    xe_loss = self.criterion[0](outputs.view(-1, outputs.size(-1)), ys, norm)
    # print('xe_loss:', xe_loss)
    
    if self.config.subcfgs[DECODER].keyframes:
      # reconstruct the orginal semantic feature vector
      v_key = self.submods[VISENC](key_enc, src_mask.squeeze(1))
      v_org = self.submods[VISENC](img_fts, src_mask.squeeze(1))
      recon_loss = self.criterion[2](v_key, v_org)
      thres = torch.min(src_mask.squeeze(1).sum(dim=-1).float()/src_mask.size(-1),torch.tensor(0.5).cuda())
      spar_loss = self.criterion[3](select.mean(dim=-1), thres)
      xe_loss += 0.5 * recon_loss + 0.5 * spar_loss

    if self.config.subcfgs[DECODER].rl:
      txt_ids, seqLogprobs, _ = self.submods[DECODER].sample(img_fts, src_mask, decoding='sample')
      sents = TRG.int2sent(txt_ids[:,1:].detach())
      sample_masks = torch.zeros((txt_ids.size(0), txt_ids.size(1)-1), dtype=torch.float32).cuda()
      for i in range(txt_ids.size(0)):
        index = (txt_ids[i]==3).nonzero() # the id of <EOS> is 3
        if len(index) == 0:
          sample_masks[i, :] = 1.
        else:
          sample_masks[i, :index[0].item()] = 1.

      self.submods[DECODER].eval()
      with torch.no_grad():
        greedy_ids, _, _ = self.submods[DECODER].sample(img_fts, src_mask, decoding='greedy')
        greedy_sents = TRG.int2sent(greedy_ids[:,1:].detach())

      gt_sents = batch_data['ref_sents']
      sample, greedy, gt = {}, {}, {}
      for i in range(len(sents)):
        sample[i] = [sents[i]+' <EOS>']
        greedy[i] = [greedy_sents[i]+' <EOS>']
        gt[i] = [gt_sents[i]+' <EOS>']
      self.submods[DECODER].train()
      rl_loss = self.criterion[1](seqLogprobs[:,1:], sample_masks, greedy, sample, gt)
      return 0.1 * xe_loss+ 2 * rl_loss
    else:
      return xe_loss

  def evaluate(self, tst_reader):
    pred_sents = []
    for batch_data in tqdm(tst_reader):
      img_fts = torch.FloatTensor(batch_data['img_ft']).cuda()
      ft_len = torch.LongTensor(batch_data['ft_len']).cuda()
      img_fts = img_fts[:,:max(ft_len)]
      img_mask = self.attn_mask(ft_len, max_len=img_fts.size(1)).unsqueeze(-2)

      # role name tokens and face feature
      rolename = batch_data['rolename_seq'].cuda() # [[3017,25,-1,-1,-1],[101,817,-1,-1,-1],[-1,-1,-1,-1,-1]]
      rolename_len = batch_data['rolename_len'].cuda()
      rolename_mask = self.attn_mask(rolename_len, max_len=10).unsqueeze(-2)
      roleface = batch_data['role_face'].cuda()

      # tags = batch_data['tags'].cuda()
      # tags_len = batch_data['tags_len'].cuda()
      # tags_mask = self.attn_mask(tags_len, max_len=4).unsqueeze(-2)

      output, _, attn = self.submods[DECODER].sample(img_fts, img_mask, rolename, roleface, rolename_mask, decoding='greedy')
      sents_per_batch = tst_reader.dataset.int2sent_role(output.detach(), rolename)
      # pdb.set_trace()
      pred_sents.extend(sents_per_batch)
    # pdb.set_trace()
    score = metrics.evaluation.compute(pred_sents, tst_reader.dataset.names, tst_reader.dataset.ref_captions)
    return score, pred_sents

  def validate(self, val_reader):
    self.eval_start()
    metrics, _ = self.evaluate(val_reader)
    torch.cuda.empty_cache()
    return metrics

  def test(self, tst_reader, tst_pred_file, tst_model_file=None):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()
    metrics, pred_data = self.evaluate(tst_reader)
    with open(tst_pred_file, 'w') as f:
      json.dump(pred_data, f)
    return metrics
  
  def infer_with_single_video_feature(self, features, feature_mask, role_feature):
    output, _, attn = self.submods[DECODER].sample(features, feature_mask, None, role_feature, None, decoding='greedy')
    return output

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

  def create_masks(self, ft_len, max_len, trg):
    if trg is not None:
      trg_mask = (trg != 1).unsqueeze(-2)
      size = trg.size(1) # get seq_len for matrix
      np_mask = self.nopeak_mask(size)
      trg_mask = trg_mask & np_mask  
    else:
      trg_mask = None
    src_mask = self.attn_mask(ft_len, max_len=max_len).unsqueeze(-2)
    return src_mask, trg_mask

  def attn_mask(self, lengths, max_len=None):
    ''' Creates a boolean mask from sequence lengths.
        lengths: LongTensor, (batch, )
    '''
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    return ~(torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .ge(lengths.unsqueeze(1)))