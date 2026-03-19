from conf import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCTRescalingLayer(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # pool along seq dim
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        b, s, d = x.shape
        y = self.avg_pool(x.permute(0, 2, 1))  # [Batch, Dim, 1]
        y = y.view(b, d)  # [Batch, Dim]
        scale = self.fc(y).view(b, 1, d)  # [Batch, 1, Dim]
        return x * scale  # channel (dct coe)-wise scaling
