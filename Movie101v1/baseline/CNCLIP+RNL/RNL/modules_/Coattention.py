import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_.position import PositionalEncoding, RelTemporalEncoding
# class CoAttention(nn.Module):
#     def __init__(self, d_model1, d_model2, d_model): #200 20
#         super().__init__()
#         self.W_b = nn.Parameter(torch.randn(d_model, d_model))
#         self.W_v = nn.Parameter(torch.randn(d_model, d_model))
#         self.W_q = nn.Parameter(torch.randn(d_model, d_model))
#         self.w_hv = nn.Parameter(torch.randn(d_model, 1))
#         self.w_hq = nn.Parameter(torch.randn(d_model, 1))

#     def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
#         # node_mask = node_mask.unsqueeze(2)
#         # x2 = x2.masked_fill(node_mask == 0, -1e30) #128, 20, 512
#         x1 = torch.transpose(x1,1,2).contiguous() #128, 200, 512 -> 128, 512, 200
 
#         C = torch.matmul(x2, torch.matmul(self.W_b, x1)) #128 20 200

#         H_v = F.tanh(torch.matmul(self.W_v, x1) + torch.matmul(torch.matmul(self.W_q, x2.permute(0, 2, 1)), C))                            # B x k x 200
#         H_q = F.tanh(torch.matmul(self.W_q, x2.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, x1), C.permute(0, 2, 1)))           # B x k x 20

#         #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 200
#         #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x 20

#         a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 200
#         a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x 20

#         v = (x1 *a_v).permute(0, 2, 1)# B x 200 512
#         q = x2 * a_q.permute(0, 2, 1) # B x 512

#         # v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
#         # q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

#         return q, v

class CoAttention(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        self.q_proj = nn.Linear(512, 512, bias = False)
        #self.gate1 = nn.Conv1d(512, 1, kernel_size=1, bias = False)
        #self.gate_s1 = nn.Sigmoid()
        #self.gate2 = nn.Conv1d(512, 1, kernel_size=1, bias = False)
        #self.gate_s2 = nn.Sigmoid()

    def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
        Q = self.q_proj(x1)
        D = x2 #* (node_mask.unsqueeze(-1))

        D_t = torch.transpose(D, 1, 2).contiguous() #128, 512, 20
        L = torch.bmm(Q, D_t) #128, 200, 20

        Q_t = torch.transpose(Q, 1, 2).contiguous() # 128, 512, 200
        A_D = F.softmax(L, dim=2)
        C_D = torch.bmm(Q_t, A_D) # 128, 512, 20

        A_Q_ = F.softmax(L, dim = 1)
        A_Q = torch.transpose(A_Q_, 1, 2).contiguous() #128, 20, 200
        C_Q = torch.bmm(D_t, A_Q) # 128, 512, 200

        #C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        #C_D = C_D * self.gate_s2(self.gate2(C_D))

        C_Q = torch.transpose(C_Q, 1, 2).contiguous()
        C_D = torch.transpose(C_D, 1, 2).contiguous()

        # C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        # C_D = C_D * self.gate_s2(self.gate2(C_D))

        return C_Q, C_D, A_D, A_Q

class CoAttention_intra(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        #self.rte = RelTemporalEncoding(n_hid=d_model2,max_len=d_model1)
    def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
        '''
        b, t, f = x1.shape[:]
        for tt in range(t):
            curr = x1[:,tt,:].unsqueeze(1).contiguous()
            time = torch.arange(0, t).unsqueeze(0).contiguous()
            time -= tt
            time = torch.abs(time)
            res = self.rte(x1,time)
            L = torch.bmm(curr, torch.transpose(res, 1, 2).contiguous())
            A = F.softmax(L, dim=2)
            x2[:,tt,:] = torch.bmm(A,res).squeeze(1).contiguous()
        return x2, x2, A, A
        '''
        Q = x1#PositionalEncoding(x1.shape[-1],0,x1.shape[-2])(x1)#self.q_proj(x1)
        D = x2#PositionalEncoding(x1.shape[-1],0,x1.shape[-2])(x2)#x2 #* (node_mask.unsqueeze(-1))

        D_t = torch.transpose(D, 1, 2).contiguous() #128, 512, 20
        #L = torch.bmm(Q, D_t)
        L = torch.bmm(PositionalEncoding(x1.shape[-1],0.2,x1.shape[-2])(Q), torch.transpose(PositionalEncoding(x1.shape[-1],0.2,x1.shape[-2])(D), 1, 2).contiguous()) #128, 200, 20

        Q_t = torch.transpose(Q, 1, 2).contiguous() # 128, 512, 200
        A_D = F.softmax(L, dim=2)
        C_D = torch.bmm(Q_t, A_D) # 128, 512, 20

        A_Q_ = F.softmax(L, dim = 1)
        A_Q = torch.transpose(A_Q_, 1, 2).contiguous() #128, 20, 200
        C_Q = torch.bmm(D_t, A_Q) # 128, 512, 200

        #C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        #C_D = C_D * self.gate_s2(self.gate2(C_D))

        C_Q = torch.transpose(C_Q, 1, 2).contiguous()
        C_D = torch.transpose(C_D, 1, 2).contiguous()

        # C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        # C_D = C_D * self.gate_s2(self.gate2(C_D))
       
        return C_Q, C_D, A_D, A_Q

# class CoAttention(nn.Module):
#     def __init__(self, d_model1, d_model2):
#         super().__init__()
#         self.linear_e = nn.Linear(d_model1, d_model2, bias = False)
#         self.gate1 = nn.Conv1d(d_model2, 1, kernel_size=1, bias = False)
#         self.gate_s1 = nn.Sigmoid()
#         self.gate2 = nn.Conv1d(d_model1, 1, kernel_size=1, bias = False)
#         self.gate_s2 = nn.Sigmoid()

#     def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
#         x1_t = torch.transpose(x1,1,2).contiguous() #128, 200, 512 -> 128, 512, 200
#         x1_corr = self.linear_e(x1_t) #128, 512, 200 -> 128, 512, 20
#         A = torch.bmm(x1_corr, x2) #128, 512, 20 -> 128, 512, 512
#         B = F.softmax(torch.transpose(A,1,2),dim=1)

#         x1_att = torch.bmm(x2, B).contiguous()
#         x1_mask = self.gate1(x1_att)
#         x1_mask = self.gate_s1(x1_mask)
#         x1_att = x1_att * x1_mask

#         x2_att = torch.bmm(x1, A).contiguous()
#         x2_mask = self.gate2(x2_att)
#         x2_mask = self.gate_s2(x2_mask)
#         x2_att = x2_att * x2_mask

#         return x1_att, x2_att
