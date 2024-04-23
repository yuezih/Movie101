from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

def qkv_attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2,-1)) / sqrt(d_k)
	# if mask is not None:
	# 	scores.data.masked_fill_(mask.eq(0), -65504.0)
	
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul(p_attn, value), p_attn

class DenseCoAttn(nn.Module):

	def __init__(self, dim1, dim2, num_attn, dropout, is_multi_head=False):
		super(DenseCoAttn, self).__init__()
		dim = min(dim1, dim2)
		self.linears = nn.ModuleList([nn.Linear(dim1, dim, bias=False),
									  nn.Linear(dim2, dim, bias=False)])
		self.d_k = dim // num_attn
		self.h = num_attn
		self.is_multi_head = is_multi_head
		self.attn = None
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])

	def forward(self, value1, value2, mask1=None, mask2=None):
		batch = value1.size(0)
		dim1, dim2 = value1.size(-1), value2.size(-1)

		query1, query2 = [l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2) 
			for l, x in zip(self.linears, (value1, value2))]

		if self.is_multi_head:
			weighted1, attn1 = qkv_attention(query2, query1, query1, mask=mask1, dropout=self.dropouts[0])
			weighted1 = weighted1.transpose(1, 2).contiguous()
			weighted2, attn2 = qkv_attention(query1, query2, query2, mask=mask2, dropout=self.dropouts[1])
			weighted2 = weighted2.transpose(1, 2).contiguous()
		else:
			weighted1, attn1 = qkv_attention(query2, query1, value1.unsqueeze(1), mask=mask1, 
				dropout=self.dropouts[0])
			weighted1 = weighted1.mean(dim=1)
			weighted2, attn2 = qkv_attention(query1, query2, value2.unsqueeze(1), mask=mask2, 
				dropout=self.dropouts[1])
			weighted2 = weighted2.mean(dim=1)
		self.attn = [attn1, attn2]

		return weighted1, weighted2

class NormalSubLayer1(nn.Module):

	def __init__(self, dim1, dim2, num_attn, dropout, dropattn=0):
		super(NormalSubLayer1, self).__init__()
		self.dense_coattn = DenseCoAttn(dim1, dim2, num_attn, dropattn)
		self.linears = nn.ModuleList([
			nn.Sequential(
				nn.Linear(dim1 + dim2, dim1),
				nn.Tanh(),
				nn.Dropout(p=dropout),
			),
			nn.Sequential(
				nn.Linear(dim1 + dim2, dim2),
				nn.Tanh(),
				nn.Dropout(p=dropout),
			)
		])

		self.linears_re = nn.ModuleList([
			nn.Sequential(
				nn.Linear(dim1 + dim2, dim1),
				nn.Sigmoid(),
				nn.Dropout(p=dropout),
			),
			nn.Sequential(
				nn.Linear(dim1 + dim2, dim2),
				nn.Sigmoid(),
				nn.Dropout(p=dropout),
			)
		])

	def forward(self, data1, data2, mask1, mask2):
		weighted1, weighted2 = self.dense_coattn(data1, data2, mask1, mask2)
		gate1 = self.linears_re[0](torch.cat([data1, weighted2], dim=2))
		gate2 = self.linears_re[1](torch.cat([data2, weighted1], dim=2))
		data1 = gate1 * data1 + (1 - gate1) * self.linears[0](torch.cat([data1, weighted2], dim=2))
		data2 = gate2 * data2 + (1 - gate2) * self.linears[1](torch.cat([data2, weighted1], dim=2))
		
		return data1, data2