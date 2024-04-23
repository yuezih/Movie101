import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from modules_.graph_convolution import GraphConvolution
from modules_.tanh_attention import TanhAttention
from modules_.dynamic_rnn import DynamicGRU
from modules_.cross_gate import CrossGate
from modules_.ConvGRU import ConvGRUCell
from modules_.Coattention import CoAttention, CoAttention_intra
from modules_.message import Message
from modules_.multihead_attention import MultiHeadAttention
from modules_.position import RelTemporalEncoding
from modules_.coa import NormalSubLayer1
from modules_.sea import NormalSubLayer2
from utils import generate_anchors
import pdb

class VideoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.args = args
        if args.dataset == 'TACOS':
            self.tran = nn.Linear(4096, args.frame_dim)
        if args.dataset == 'Charades':
            self.tran = nn.Linear(1024, args.frame_dim)
        else:
           self.tran = nn.Linear(1536, 512)
        self.max_num_frames = args.max_num_frames
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(args.frame_dim, args.num_heads)
            for _ in range(args.num_attn_layers)
        ])
        self.rnn = DynamicGRU(args.frame_dim, args.d_model >> 1, bidirectional=True, batch_first=False)
        self.attn_width = 3
        self.self_attn_mask = torch.empty(self.max_num_frames, self.max_num_frames) \
            .float().fill_(float(-1e10)).cuda()
        for i in range(0, self.max_num_frames):
            low = i - self.attn_width
            low = 0 if low < 0 else low
            high = i + self.attn_width + 1
            high = self.max_num_frames if high > self.max_num_frames else high
            # attn_mask[i, low:high] = 0
            self.self_attn_mask[i, low:high] = 0

    def forward(self, x, mask):
        if self.args.dataset in ['TACOS', 'Charades']:
            x = self.tran(x)
        else:
           x_face = self.tran(x[:,:,1024:])
           x = x[:,:,:1024]
        x = x.transpose(0, 1)
        length = mask.sum(dim=-1)

        for a in self.attn_layers:
            res = x
            x, _ = a(x, x, x, None, attn_mask=self.self_attn_mask)
            x = F.dropout(x, self.dropout, self.training)
            x = res + x

        x = self.rnn(x, length, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)
        x = x.transpose(0, 1)
        x = x + x_face
        return x


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words
  
        self.rnn = DynamicGRU(args.word_dim, args.d_model >> 1, bidirectional=True, batch_first=True)
        self.tran_face = nn.Linear(512, args.d_model)
        #self.gcn_layers = nn.ModuleList([
        #    GraphConvolution(args.word_dim)
        #    for _ in range(args.num_gcn_layers)
        #])

    def forward(self, x, mask, face_feats, face_mask):
        length = mask.sum(dim=-1)
        # face_x = self.tran_face(face_feats)
        face_x = face_feats
        x = self.rnn(x, length, self.max_num_words)
        x = F.dropout(x, self.dropout, self.training)
        x = torch.cat([x, face_x], 1)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dropout = args.dropout
        self.max_num_frames = args.max_num_frames

        self.anchors = generate_anchors(dataset=args.dataset)
        self.num_anchors = self.anchors.shape[0]
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, args.max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

        # VideoEncoder
        self.video_encoder = VideoEncoder(args)

        # SentenceEncoder
        self.sentence_encoder = SentenceEncoder(args)

        self.co1 = NormalSubLayer1(args.d_model, args.d_model, 4, 0)
        self.se1 = NormalSubLayer2(args.d_model, args.d_model, 4, 0)

        self.co2 = NormalSubLayer1(args.d_model, args.d_model, 4, 0)
        self.se2 = NormalSubLayer2(args.d_model, args.d_model, 4, 0)

        self.co3 = NormalSubLayer1(args.d_model, args.d_model, 4, 0)
        self.se3 = NormalSubLayer2(args.d_model, args.d_model, 4, 0)

        
        self.v2s = TanhAttention(args.d_model)
        # self.s2v = TanhAttention(args.d_model)
        #self.cross_gate = CrossGate(args.d_model)
        # self.fc = Bilinear(args.d_model, args.d_model, args.d_model)
        # 
        # self.s2v = nn.Conv1d(args.max_num_words, args.max_num_frames, kernel_size=1)
        
        self.rnn = DynamicGRU(args.d_model << 1, args.d_model >> 1, bidirectional=True, batch_first=True)

        self.fc_score = nn.Conv1d(args.d_model, self.num_anchors, kernel_size=1, padding=0, stride=1)
        self.fc_reg = nn.Conv1d(args.d_model, self.num_anchors << 1, kernel_size=1, padding=0, stride=1)

        # loss function
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.SmoothL1Loss()

    def forward(self, frames, frame_mask, words, word_mask,
                label, label_mask, gt,
                node_pos, node_mask, adj_mat, face_feats, face_mask):
        frames_len = frame_mask.sum(dim=-1)
        # pdb.set_trace()
        frames = F.dropout(frames, self.dropout, self.training)
        words = F.dropout(words, self.dropout, self.training)
        # pdb.set_trace()
        frames = self.video_encoder(frames, frame_mask)
        x = self.sentence_encoder(words, word_mask, face_feats, face_mask)
        word_mask = torch.cat([word_mask,face_mask],-1)
        node_mask = torch.cat([node_mask,face_mask],-1)
        
        #attentive
        frames1, x1 = self.co1(frames, x, frame_mask, node_mask)
        frames1, x1 = self.se1(frames1, x1, frame_mask, node_mask)

        frames1, x1 = self.co2(frames1, x1, frame_mask, node_mask)
        frames1, x1 = self.se2(frames1, x1, frame_mask, node_mask)

        frames1, x1 = self.co3(frames1, x1, frame_mask, node_mask)
        frames1, x1 = self.se3(frames1, x1, frame_mask, node_mask)
        
        #frames1, x1 = frames, x
        #a1, a2 = 1, 1
        # interactive
        # x1 = self.atten(frames1, x1, node_mask, last=True)
        x1 = self.v2s(frames1, x1, node_mask)
        #frames1, x1 = self.cross_gate(frames1, x1)
        # x1 = self.s2v(x1)
        x = torch.cat([frames1, x1], -1) #x1
        x = self.rnn(x, frames_len, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)
        #import pdb;pdb.set_trace()
        # loss
        predict = torch.sigmoid(self.fc_score(x.transpose(-1, -2))).transpose(-1, -2)
        # [batch, max_num_frames, num_anchors]
        reg = self.fc_reg(x.transpose(-1, -2)).transpose(-1, -2)
        reg = reg.contiguous().view(-1, self.args.max_num_frames * self.num_anchors, 2)
        # [batch, max_num_frames, num_anchors, 2]
        predict_flatten = predict.contiguous().view(predict.size(0), -1) * label_mask.float()
        cls_loss = self.criterion1(predict_flatten, label)
        # gt_box: [batch, 2]
        proposals = torch.from_numpy(self.proposals).type_as(gt).float()  # [max_num_frames, num_anchors, 2]
        proposals = proposals.view(-1, 2)
        if not self.training:
            train_flag = 0

            if train_flag == 0:
                # if test, open this
                batch_now = reg.shape[0]
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
                predict_box = proposals
                predict_reg = reg # [nb, 2]
                refine_box = predict_box + predict_reg
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
                reg_loss = self.criterion2(refine_box, gt.float())
                loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
                predict_flatten = (predict.contiguous().view(predict.size(0), -1) * label_mask.float())
            
            else:
                # if train, open this
                indices = torch.argmax(predict_flatten, -1)
                predict_box = proposals[indices]  # [nb, 2]
                predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
                refine_box = predict_box + predict_reg
                reg_loss = self.criterion2(refine_box, gt.float())
                if self.args.dataset in ['TACOS', 'Charades']:
                   loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
                else:
                   loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        else:
            # indices = torch.argmax(label, -1)
            indices = torch.where(adj_mat > 0)
            batch_now = reg.shape[0]
            if self.args.dataset == 'TACOS':
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            elif self.args.dataset == 'Charades':
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            else:
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            predict_box = proposals[indices]  # [nb, 2]
            predict_reg = reg[indices]  # [nb, 2]
            refine_box = predict_box + predict_reg
            if self.args.dataset == 'TACOS':
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            elif self.args.dataset == 'Charades':
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            else:         
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            gt = gt[indices]
            reg_loss = self.criterion2(refine_box, gt.float())
            if self.args.dataset in ['TACOS', 'Charades']:
                loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            else:
                loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        # predict_box = proposals[indices]  # [nb, 2]
        # predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
        # refine_box = predict_box + predict_reg
        # reg_loss = self.criterion2(refine_box, gt.float())
        # loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        # if detail:
        #     return refine_box, loss, predict_flatten, reg, proposals
        return refine_box, loss, predict_flatten, loss, loss
