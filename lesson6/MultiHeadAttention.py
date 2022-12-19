import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, isMask:bool=False) -> None:
        super().__init__()
        self.isMask = isMask

    def forward(self, q, k, v):
        assert q.size() == k.size() == v.size()
        assert len(q.size()) == 3
        bs, _, d_k = q.size()
        # k.transpose(2,1) : (bs, d_k, _) shape
        # q                : (bs, _, d_k)
        # k.transpose(2,1) : (bs, d_k, _)
        # matmul           : (bs, _, _)
        # v                : (bs, _, d_k)
        matmul = torch.matmul(q, k.transpose(2,1))
        scaled = matmul / np.sqrt(d_k)
        if self.isMask:
            scaled = scaled.masked_fill(scaled==0, 1e-4)
        softmax = torch.softmax(scaled, dim=1)
        # final            : (bs, _, d_k)
        final = torch.matmul(softmax, v) # (bs, _, d_k)
        return final

class MultiHeadAttention(nn.Module):
    def __init__(self, isMasked:bool=False, h:int=8, dim:int=512) -> None:
        super().__init__()
        self.isMasked = isMasked
        self.h = h
        self.dim = dim
        self.linear_v = nn.Linear(self.dim, self.dim)
        self.linear_k = nn.Linear(self.dim, self.dim)
        self.linear_q = nn.Linear(self.dim, self.dim)
        self.SDPA = ScaledDotProductAttention(self.isMasked)

    def forward(self, q, k, v):
        q_out = self.linear_q(q)
        k_out = self.linear_k(k)
        v_out = self.linear_v(v)
        out = self.SDPA(q_out, k_out, v_out)
        return out
