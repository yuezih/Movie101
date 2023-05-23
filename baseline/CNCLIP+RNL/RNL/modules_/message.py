import torch
import torch.nn as nn
import torch.nn.functional as F


class Message(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model1, d_model2, bias=False)
        self.fc_gate2 = nn.Linear(d_model1, d_model2, bias=False)
        #self.fc_gate1 = nn.Conv1d(d_model1, d_model2, kernel_size = 1, bias=False)
        #self.fc_gate2 = nn.Conv1d(d_model2, d_model2, kernel_size = 1, bias=False)
        #self.bn1 = nn.BatchNorm1d(d_model2)
        #self.bn2 = nn.BatchNorm1d(d_model2)
        #self.re1 = nn.ReLU()
        #self.re2 = nn.ReLU()

    def forward(self, x1):
        g1 = self.fc_gate1(x1)
        #g1 = self.re1(self.bn1(self.fc_gate1(x1)))
        #g2 = self.re2(self.bn2(self.fc_gate2(g1)))
        return g1
