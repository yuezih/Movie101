from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def l2norm(inputs, dim=-1):
  # inputs: (batch, dim_ft)
  norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
  zeros = torch.zeros(norm.size()).cuda()
  zeros[norm == 0] = 1
  inputs = inputs / (norm+zeros)
  return inputs


def sequence_mask(lengths, max_len=None):
  ''' Creates a boolean mask from sequence lengths.
  '''
  # lengths: LongTensor, (batch, )
  batch_size = lengths.size(0)
  max_len = max_len or lengths.max()
  return (torch.arange(0, max_len)
          .type_as(lengths)
          .repeat(batch_size, 1)
          .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
  # Use pytorch version when available.
  rnn = getattr(nn, rnn_type.upper())(**kwargs)
  return rnn


def calc_rnn_outs_with_sort(rnn, inputs, seq_lens, init_states=None):
  '''
  inputs: FloatTensor, (batch, seq_len, dim_ft)
  seq_lens: LongTensor, (batch,)
  '''
  seq_len = inputs.size(1)
  # sort
  sorted_seq_lens, seq_sort_idx = torch.sort(seq_lens, descending=True)
  _, seq_unsort_idx = torch.sort(seq_sort_idx, descending=False)
  # pack
  inputs = torch.index_select(inputs, 0, seq_sort_idx)
  packed_inputs = pack_padded_sequence(inputs, sorted_seq_lens, batch_first=True)
  if init_states is not None:
    if isinstance(init_states, tuple):
      new_states = []
      for i, state in enumerate(init_states):
        new_states.append(torch.index_select(state, 1, seq_sort_idx))
      init_states = tuple(new_states)
    else:
      init_states = torch.index_select(init_states, 1, seq_sort_idx)
  # rnn
  packed_outs, states = rnn(packed_inputs, init_states)
  # unpack
  outs, _ = pad_packed_sequence(packed_outs, batch_first=True, 
    total_length=seq_len, padding_value=0)
  # unsort
  # outs.size = (batch, seq_len, num_directions * hidden_size)     
  outs = torch.index_select(outs, 0, seq_unsort_idx)   
  if isinstance(states, tuple):
    # states: (num_layers * num_directions, batch, hidden_size)
    new_states = []
    for i, state in enumerate(states):
      new_states.append(torch.index_select(state, 1, seq_unsort_idx))
    states = tuple(new_states)
  else:
    states = torch.index_select(states, 1, seq_unsort_idx)

  return outs, states
