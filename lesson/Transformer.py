import torch
import torch.nn as nn
from Sublayers import *

class Encoder(nn.Module):
    def __init__(self,
                 N=6,
                 isMasked=False,
                 h=8,
                 dim=512,
                 out_dim=512) -> None:
        super().__init__()
        self.N = N
        self.isMasked = isMasked
        self.h = h
        self.dim = dim
        self.out_dim = out_dim
        self.encode_mha = [mha(isMasked=self.isMasked, h=self.h, dim=self.dim) for idx in range(self.h)]
        self.posenc = PositionalEmbedding()
        self.ff = FeedForward()

    def foward(self, inp):
        inter = self.posenc(inp)
        for _ in range(self.N):
            q = inter
            k = inter
            v = inter
            cat = []
            for index in range(self.h):
                cat.append(self.encode_mha[index](q, k, v))
            x = torch.cat(cat)
            x += inter
            self.layernorm1 = nn.LayerNorm(x.size()[1:])
            inter = self.layernorm1(x)
            x = self.ff(inter)
            x += inter
            self.layernorm2 = nn.LayerNorm(x.size()[1:])
            inter = self.layernorm2(x)
        return inter


class Decoder(nn.Module):
    def __init__(self,
                 N=6,
                 isMasked=False,
                 h=8,
                 dim=512,
                 out_dim=512) -> None:
        super().__init__()
        self.N = N
        self.isMasked = isMasked
        self.h = h
        self.dim = dim
        self.out_dim = out_dim
        self.decode_mha = [mha(isMasked=self.isMasked, h=self.h, dim=self.dim) for idx in range(self.h)]
        self.decode_mha_msk = [[mha(isMasked=True, h=self.h, dim=self.dim) for idx in range(self.h)]]
        self.posenc = PositionalEmbedding()
        self.ff = FeedForward()

    def foward(self, inp, enc_out):
        inter = self.posenc(inp)
        for _ in range(self.N):
            q = inter
            k = inter
            v = inter
            # Masked MHA
            cat = []
            for index in range(self.h):
                cat.append(self.decode_mha_msk[index](q, k, v))
            x = torch.cat(cat)
            x += inter
            self.layernorm1 = nn.LayerNorm(x.size()[1:])
            # MHA
            cat = []
            for index in range(self.h):
                cat.append(self.decode_mha[index](enc_out, enc_out, v))
            x = torch.cat(cat)
            x += inter
            self.layernorm2 = nn.LayerNorm(x.size()[1:])
            inter = self.layernorm2(x)
            # FFN
            x = self.ff(inter)
            x += inter
            self.layernorm3 = nn.LayerNorm(x.size()[1:])
            inter = self.layernorm3(x)
        return inter

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, inp):
        enc_out = self.enc(inp)
        out = self.dec(inp, enc_out)
        return out