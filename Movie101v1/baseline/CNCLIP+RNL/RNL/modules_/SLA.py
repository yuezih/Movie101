import torch
import torch.nn as nn
import torch.nn.functional as F


class stack_latent_attention(nn.Module):
    def __init__(self, d_model, first=False):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_h = nn.Linear(d_model, d_model, bias=False)

        self.query_v = nn.Linear(d_model, d_model, bias=True)
        self.key_h = nn.Linear(d_model, d_model, bias=False)

        if first:
            self.W_z = nn.Linear(d_model, d_model, bias=False)
        else:
            self.W_z = nn.Linear(d_model * 2, d_model, bias=False)

        self.W_u = nn.Linear(d_model, 1, bias=False)
        

    def forward(self, v, h, z, v_mask=None, h_mask=None):
        #v: bx200x512, h: bx20x512, z: bx200x1024
        v_v = self.W_v(v)
        h_h = self.W_h(h)
        z_z = self.W_z(z)

        query = self.query_v(v)
        key = self.key_h(h)

        if h_mask is not None:
            h_mask = h_mask.unsqueeze(-1)  # [nb, 20, 1]
            key = key.masked_fill(h_mask == 0, -1e30)
        h_v = torch.bmm(query, torch.transpose(key, 1, 2).contiguous()) # [b, 200, 20]
        h_v = torch.bmm(F.softmax(h_v, -1), h_h) # [b, 200, 512]

        s = torch.sigmoid(v_v+h_v+z_z) #s: bx200x512
        f = s.unsqueeze(2) + s.unsqueeze(1) #f: bx200x200x512
        f = self.W_u(f).squeeze(-1) #f: bx200x200
        if v_mask is not None:
            v_mask = v_mask.unsqueeze(1)  # [nb, 1, 200]
            f = f.masked_fill(v_mask == 0, -1e30)
        f = F.softmax(f, -1)
        f = torch.matmul(f, v) #f: bx200x512

        h_new = f 
        z_new = torch.cat((s, f), -1)
        
        return h_new, z_new
