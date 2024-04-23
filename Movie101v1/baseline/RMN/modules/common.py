import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
import pdb
import numpy as np


class PositionalEncoder(nn.Module):
  def __init__(self, d_model, max_seq_len=200, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    # create constant 'pe' matrix with values dependant on pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model, 2):
        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  
  def forward(self, x, step=None):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    seq_len = x.size(1)
    if seq_len == 1 and step is not None:
      pe = Variable(self.pe[:,step-1], requires_grad=False).cuda()
    else:
      pe = Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
    x = x + pe
    return self.dropout(x)


class Norm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.size = d_model   
    # create two learnable parameters to calibrate normalisation
    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    self.eps = eps
    
  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
      / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9) 
  scores = F.softmax(scores, dim=-1) 
  if dropout is not None:
    scores = dropout(scores)     
  output = torch.matmul(scores, v)
  return output, scores

def raw_attention(q, k, v, d_k, mask=None, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9) 
  # scores = F.softmax(scores, dim=-1) 
  if dropout is not None:
    scores = dropout(scores)     
  output = torch.matmul(scores, v)
  return output, scores

class RawMultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super().__init__() 
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)  
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def shape(self, x):
    bs = x.size(0)
    return x.view(bs, -1, self.h, self.d_k).transpose(1,2)
    
  def forward(self, q, k, v, mask=None, layer_cache=None, attn_type=None):
    if layer_cache is not None:
      if attn_type == "self":
        k = self.shape(self.k_linear(k)) # (batch, key_len=1, dim_embed)
        v = self.shape(self.v_linear(v))
        if layer_cache['self_keys'] is not None:
          k = torch.cat((layer_cache['self_keys'], k), dim=2)
        if layer_cache['self_values'] is not None:
          v = torch.cat((layer_cache['self_values'], v), dim=2)
        layer_cache['self_keys'] = k
        layer_cache['self_values'] = v
    else:
      k = self.shape(self.k_linear(k)) # (batch, key_len, dim_embed)
      v = self.shape(self.v_linear(v))

    bs = q.size(0)
    q = self.shape(self.q_linear(q))
    # calculate attention using function we will define next
    scores, attn = raw_attention(q, k, v, self.d_k, mask, self.dropout)
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
    output = self.out(concat)
    return output, attn

class MultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super().__init__() 
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)  
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def shape(self, x):
    bs = x.size(0)
    return x.view(bs, -1, self.h, self.d_k).transpose(1,2)
    
  def forward(self, q, k, v, mask=None, layer_cache=None, attn_type=None):
    if layer_cache is not None:
      if attn_type == "self":
        k = self.shape(self.k_linear(k)) # (batch, key_len=1, dim_embed)
        v = self.shape(self.v_linear(v))
        if layer_cache['self_keys'] is not None:
          k = torch.cat((layer_cache['self_keys'], k), dim=2)
        if layer_cache['self_values'] is not None:
          v = torch.cat((layer_cache['self_values'], v), dim=2)
        layer_cache['self_keys'] = k
        layer_cache['self_values'] = v
    else:
      k = self.shape(self.k_linear(k)) # (batch, key_len, dim_embed)
      v = self.shape(self.v_linear(v))

    bs = q.size(0)
    q = self.shape(self.q_linear(q))
    # calculate attention using function we will define next
    scores, attn = attention(q, k, v, self.d_k, mask, self.dropout)
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
    output = self.out(concat)
    return output, attn


class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout=0.1):
    super().__init__() 
    # We set d_ff as a default to 2048
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
  def forward(self, x):
    x = self.dropout(F.relu(self.linear_1(x)))
    x = self.linear_2(x)
    return x


def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
