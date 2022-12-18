import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttention as mha


class PositionalEmbedding(nn.Module):
    def __init__(self, denominator=10000, d_model=512) -> None:
        super().__init__()
        self.denominator = denominator
        self.d_model = d_model

    def forward(self, x):
        i = x.size()[2]
        inp = torch.Tensor([pos / (self.denominator ** (2*i/self.d_model)) for pos in range(i)])
        pe = torch.Tensor([torch.cos(inp[pos]) if pos//2==0 else torch.sin(inp[pos]) for pos in range(i)])
        return pe + x

class FeedForward(nn.Module):

    def __init__(self, d_in=512, d_hid=512):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x

