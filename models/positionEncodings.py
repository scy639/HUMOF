import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionEncoding_A(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.position_encoding = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.normal_(self.position_encoding, std=0.02)

    def forward(self, x):
        # x.shape = (B, sequence_length==J, dim==3*DCT)
        return x + self.position_encoding


