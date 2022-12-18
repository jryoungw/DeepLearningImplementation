import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, isMasked:bool=False) -> None:
        super().__init__()
        self.isMasked = isMasked

    def forward(self, q, k, v):
        assert q.size() == k.size() == v.size()
        assert len(q.size()) == 3
        bs, _, dk = q.size()
        matmul = torch.matmul(q.transpose(2,1), k)
        scaled = matmul / np.sqrt(dk)
        if self.isMasked:
            scaled = scaled.masked_fill(scaled==0, 1e-4)
        softmax = torch.softmax(scaled, dim=1)
        final = torch.matmul(softmax, v.transpose(2,1)).transpose(2,1)
        return final


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 isMasked:bool=False, 
                 h:int=8,
                 dim:int=512) -> None:
        super().__init__()
        self.isMasked = isMasked
        self.h = h
        self.inp_dim = dim
        self.out_dim = dim
        self.linear_q = nn.Linear(self.inp_dim, self.out_dim)
        self.linear_k = nn.Linear(self.inp_dim, self.out_dim)
        self.linear_v = nn.Linear(self.inp_dim, self.out_dim)
        self.SDPA = ScaledDotProductAttention(self.isMasked)

    def forward(self, q, k, v):
        q_out = self.linear_q(q)
        k_out = self.linear_k(k)
        v_out = self.linear_v(v)
        out = self.SDPA(q_out, k_out, v_out)
        return out