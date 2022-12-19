import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention as mha

class PositionalEncoding(nn.Module):
    def __init__(self, denominator=10000, d_model=512) -> None:
        super().__init__()
        self.denominator = denominator
        self.d_model = d_model

    def forward(self, x):
        i = x.size()[2]
        inp = torch.Tensor([pos / (self.denominator**(2*i/self.d_model)) for pos in range(i)])
        pe = torch.Tensor([torch.cos(inp[pos]) if pos%2==1 else torch.sin(inp[pos]) for pos in range(i)])
        return pe + x

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_in=512, d_hid=512) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.w_1 = nn.Linear(self.d_in, self.d_hid)
        self.w_2 = nn.Linear(self.d_hid, self.d_in)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.w_2(x)
        return x