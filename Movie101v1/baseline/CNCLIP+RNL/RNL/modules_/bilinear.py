import torch
import torch.nn as nn


class Bilinear(nn.Module):
    def __init__(self, in_features1, in_features2, out_feature):
        super().__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_feature = out_feature
        self.d_model = min(in_features1, in_features2)
        self.U = nn.Conv1d(200, 200,1)
        self.V = nn.Conv1d(20, 200,1)
        self.P = nn.Linear(self.d_model, out_feature)

    def forward(self, x, y, activate_fn=None):
        if activate_fn is not None:
            return self.P(activate_fn(self.U(x) * self.V(y)))
        return self.P(self.U(x) * self.V(y))
