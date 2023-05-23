from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

import framework.ops
from framework.ops import l2norm
import framework.configbase


class VideoEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super(VideoEncoderConfig, self).__init__()
    self.rnn_type = 'gru'  # lstm, gru
    self.ft_dim = 4096
    self.hidden_size = 512
    self.num_layers = 1
    self.bidirectional = True
    self.dropout = 0.2


class VideoEncoder(nn.Module):
  def __init__(self, config):
    super(VideoEncoder, self).__init__()
    self.config = config
    self.embedding = nn.Linear(self.config.ft_dim, self.config.hidden_size*2)
    self.rnn = framework.ops.rnn_factory(self.config.rnn_type,
      input_size=self.config.hidden_size*2, hidden_size=self.config.hidden_size, 
      num_layers=self.config.num_layers, dropout=self.config.dropout,
      bidirectional=self.config.bidirectional, bias=True, batch_first=True)
    input_size = self.config.hidden_size*2 if self.config.bidirectional else self.config.hidden_size
    self.fc = nn.Linear(input_size, 1024)
    self.dropout = nn.Dropout(p=self.config.dropout)
    self.init_weights()

  def xavier_init_fc(self, fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

  def init_weights(self):
    """Xavier initialization for the fully connected layer
    """
    self.xavier_init_fc(self.embedding)
    self.xavier_init_fc(self.fc)

  def forward(self, inputs, seq_masks, init_states=None):
    # outs.size = (batch, seq_len, num_directions * hidden_size)
    #seq_masks = framework.ops.sequence_mask(seq_lens, max_len=inputs.size(1)).float()
    embeds = self.embedding(inputs)
    seq_lens = seq_masks.sum(dim=1).long()
    self.rnn.flatten_parameters()
    outs, states = framework.ops.calc_rnn_outs_with_sort(self.rnn, embeds, seq_lens, init_states)
    outs = torch.sum(outs * seq_masks.float().unsqueeze(-1), 1) / seq_lens.unsqueeze(-1).float()
    outs = self.dropout(outs)
    embeds = l2norm(torch.tanh(self.fc(outs)))
    return embeds
