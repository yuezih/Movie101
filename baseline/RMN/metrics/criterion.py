import torch
import torch.nn as nn
import torch.nn.functional as F

from cap_eval.cider.cider import Cider
from cap_eval.meteor.meteor import Meteor
from cap_eval.bleu.bleu import Bleu
import framework.configbase
import framework.ops

import numpy as np
import json


class LabelSmoothingLoss(nn.Module):
  """
  With label smoothing,
  KL-divergence between q_{smoothed ground truth prob.}(w)
  and p_{prob. computed by model}(w) is minimized.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    assert 0.0 < label_smoothing <= 1.0
    self.padding_idx = ignore_index
    super(LabelSmoothingLoss, self).__init__()
    smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    one_hot = torch.full((tgt_vocab_size,), smoothing_value).cuda()
    one_hot[self.padding_idx] = 0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))
    self.confidence = 1.0 - label_smoothing

  def forward(self, output, target, norm):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    model_prob = self.one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
    loss = F.kl_div(output, model_prob, reduction='sum')
    return loss.div(float(norm))


class UnlikelihoodLoss(nn.Module):
  """
  Enhancing the LabelSmoothingLoss with unlikelihood training objective.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    assert 0.0 < label_smoothing <= 1.0
    self.padding_idx = ignore_index
    super(UnlikelihoodLoss, self).__init__()
    smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    one_hot = torch.full((tgt_vocab_size,), smoothing_value).cuda()
    one_hot[self.padding_idx] = 0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))
    self.confidence = 1.0 - label_smoothing

  def forward(self, output, target, norm):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    model_prob = self.one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
    loss = F.kl_div(output, model_prob, reduction='sum')

    with torch.no_grad():
      # Make 'the triangle'.
      ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
      ctx_cands_ = (ctx_cands.tril(-1) + self.padding_idx)
      ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
      ctx_cands = ctx_cands.tril(-1) + ctx_cands_
      # Don't include the target for that timestep as a negative target.
      ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), self.padding_idx)
      negative_targets = torch.zeros_like(output).scatter_(1, ctx_cands, 1)
      negative_targets[:,self.padding_idx] = 0

    # Compute the unlikelihood loss
    one_minus_probs = torch.clamp((1.0 - output.exp()), min=1e-5)
    custom_loss = -torch.log(one_minus_probs)*negative_targets
    custom_loss = custom_loss.sum()
    return loss.div(float(norm))+custom_loss.div(float(norm))


class L2Loss(nn.Module):
  """
  Compute the l2 distance
  """
  def __init__(self,):
    super(L2Loss, self).__init__()
    
  def forward(self, h_pred, h_target):
    return torch.norm(h_target - h_pred, p=2)


class L1Loss(nn.Module):
  """
  Compute the l1 distance
  """
  def __init__(self,):
    super(L1Loss, self).__init__()
    
  def forward(self, h_pred, h_target):
    return torch.mean(torch.abs(h_target - h_pred))


class RewardLoss(nn.Module):
  """
  Compute the self-critical loss with hybrid rewards (1.0 * cider + 0.3 * diverse)
  """
  def __init__(self, document_freq):
    super(RewardLoss,self).__init__()
    self.scorers = {}
    self.scorers['cider'] = Cider()
    self.scorer_names = ['cider', 'diverse']
    self.weights = np.array([1, 0.3])
    self.document_freq = json.load(open(document_freq))

  def token_ngram_freq(self, sents, len_size):
    freqs = np.zeros(shape=[len(sents),len_size])
    for id in sents:
      words = sents[id][0].split()[:-1]
      tmp, count = np.zeros(len_size,), np.zeros(len_size,)
      for j in range(len(words)-4):
        if '.' in words[j:j+4] or ',' in words[j:j+4]:
          continue
        ngram = ' '.join(words[j:j+4]) 
        if ngram in self.document_freq:
          tmp[j:j+4] += 1.0/self.document_freq[ngram]
          count[j:j+4] += 1
      count[count==0] = 1
      freqs[int(id)] = tmp/count
    return freqs

  def global_ngram_freq(self, sents):
    freqs = []
    for id in sents:
      words = sents[id][0].split()[:-1]
      tmp, count = 0, 0
      for j in range(len(words)-4):
        if '.' in words[j:j+4] or ',' in words[j:j+4]:
          continue
        ngram = ' '.join(words[j:j+4]) 
        if ngram in self.document_freq:
          tmp += 1.0/self.document_freq[ngram] 
          count += 1
      freqs.append(tmp/max(count, 1))
    return np.expand_dims(np.array(freqs),1)

  def calc_reward(self, greedy_sents, sample_sents, ref_sents, len_size):
    batch_size = len(greedy_sents)
    rewards = np.zeros(shape=[batch_size,len_size], dtype=np.float32)
    for scorer_name, weight in zip(self.scorer_names, self.weights):
      if scorer_name == 'diverse':
        sample_scores = self.token_ngram_freq(sample_sents, len_size)
        greedy_scores = self.global_ngram_freq(greedy_sents)
      else:
        scorer = self.scorers[scorer_name]
        _, greedy_scores = scorer.compute_score(ref_sents, greedy_sents)
        _, sample_scores = scorer.compute_score(ref_sents, sample_sents)
        greedy_scores = np.array(greedy_scores)
        sample_scores = np.array(sample_scores)

      if scorer_name != 'diverse':
        rewards += np.expand_dims(weight * (sample_scores - greedy_scores),1)
      else:
        rewards += weight * (sample_scores - greedy_scores)
    rewards = torch.FloatTensor(rewards).cuda().data
    return rewards

  def forward(self, sample_word_logprobs, sample_word_masks, greedy_sents, sample_sents, ref_sents):
    rewards = self.calc_reward(greedy_sents, sample_sents, ref_sents, sample_word_logprobs.size(1))
    logprobs = torch.sum(sample_word_logprobs * rewards * sample_word_masks)
    loss = - logprobs / torch.sum(sample_word_masks)
    return loss
