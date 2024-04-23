import torch
import torch.nn as nn
from modules.common import *
import pdb

class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_size, d_model)
  def forward(self, x):
    return self.embed(x)

class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1, keyframes=False):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.ff = FeedForward(d_model, dropout=dropout)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.keyframes = keyframes

  def forward(self, x, mask):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn(x2,x2,x2,mask)[0])
    x2 = self.norm_2(x)
    if self.keyframes:
      select = self.dropout_2(torch.sigmoid(self.ff(x2)))
      x = x * select
      return x, select
    else:
      x = x + self.dropout_2(self.ff(x2))
      return x, None

class TagsEncoder(nn.Module):
  def __init__(self, d_model, N, heads, dropout):
    super().__init__()
    # self.N = N
    # self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.tags_embedder = Embedder(42, d_model)
    self.norm = Norm(d_model)
    
  def forward(self, tags):
    x = self.tags_embedder(tags)
    # for i in range(self.N):
    #   x, select = self.layers[i](x, tags_mask)
    return self.norm(x)

class FaceEncoder(nn.Module):
  def __init__(self, d_model, N, heads, dropout):
    super().__init__()
    N = 1
    self.N = N
    # self.token_embed = Embedder(4301, d_model)
    self.face_embed = nn.Linear(512, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)

  def forward(self, roleface):
    x = self.face_embed(roleface)
    x = self.pe(x)
    for i in range(self.N):
      x, select = self.layers[i](x, None)
    return self.norm(x)


class Encoder(nn.Module):
  def __init__(self, ft_dim, d_model, N, heads, dropout, keyframes=False):
    super().__init__()
    self.N = N
    # self.embed = nn.Linear(ft_dim, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout, keyframes), N)
    self.norm = Norm(d_model)
    self.keyframes = keyframes

  def forward(self, src, mask):
    # x = self.embed(src)
    x = self.pe(src)
    for i in range(self.N):
      x, select = self.layers[i](x, mask)
      
    if self.keyframes:
      # select key frame features
      select = select.mean(dim=-1, keepdim=True) * mask.transpose(-1, -2).float()
      org_frame = src * select
      return self.norm(x), org_frame, select.squeeze(-1)
    else:
      return self.norm(x), None, None

  def get_keyframes(self, src, mask):
    x = self.embed(src)
    x = self.pe(x)
    for i in range(self.N):
      x, select = self.layers[i](x, mask)
    select = select.mean(dim=-1, keepdim=True) * mask.transpose(-1, -2).float()
    select = select.squeeze(-1)
    thres = min(75, src.size(1))
    indices = select.topk(thres, 1)[1].sort()[0]
    x = torch.gather(x, 1, indices.unsqueeze(-1).expand(x.size(0),-1,x.size(-1)))
    mask = torch.gather(mask, 2, indices.unsqueeze(1).expand(x.size(0),1,-1))
    return self.norm(x), mask

class VideoEncoder(nn.Module):
  def __init__(self, d_model, N, heads, dropout):
    super().__init__()
    self.video_embed = nn.Linear(1536, d_model)
    self.norm = Norm(d_model)

  def forward(self, feature):
    x = self.video_embed(feature)
    return self.norm(x)
