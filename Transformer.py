import torch
import torch.nn as nn
from Sublayers import *


class Encoder(nn.Module):
    def __init__(self, 
                 N=6, 
                 isMasked:bool=False, 
                 h=8, 
                 dim=512, 
                 out_dim=256) -> None:
        super().__init__()
        self.N = N
        self.isMasked = isMasked
        self.h = h
        self.dim = dim
        self.out_dim = out_dim
        self.encoder_mha = mha(isMasked=self.isMasked, h=self.h, dim=self.dim)
        self.posenc = PositionalEmbedding()
        self.ff = FeedForward()


    def forward(self, inp):
        inter = self.posenc(inp)
        for _ in range(self.N):
            q = inter
            k = inter
            v = inter
            x = self.encoder_mha(q, k, v)
            x += inter
            inter = x
            self.layernorm = nn.LayerNorm(x.size()[1:])
            x = self.layernorm(x)
            x = self.ff(x)
            x += inter
            self.layernorm2 = nn.LayerNorm(x.size()[1:])
            inter = self.layernorm2(x)
        return inter


class Decoder(nn.Module):
    def __init__(self, 
                 N=6, 
                 isMasked:bool=True, 
                 h=8, 
                 dim=512, 
                 out_dim=256) -> None:
        super().__init__()
        self.N = N
        self.isMasked = isMasked
        self.h = h
        self.dim = dim
        self.out_dim = out_dim
        self.decoder_mha_masked = mha(isMasked=self.isMasked, h=self.h, dim=self.dim)
        self.decoder_mha = mha(isMasked=False, h=self.h, dim=self.dim)
        self.posenc = PositionalEmbedding()
        self.ff = FeedForward()


    def forward(self, inp, enc_x):
        inter = self.posenc(inp)
        for _ in range(self.N):
            q = inter
            k = inter
            v = inter
            x = self.decoder_mha_masked(q, k, v)
            x += inter
            inter = x
            self.layernorm = nn.LayerNorm(x.size()[1:])
            x = self.layernorm(x)
            inter = x
            x = self.decoder_mha(enc_x, enc_x, x)
            x += inter
            self.layernorm2 = nn.LayerNorm(x.size()[1:])
            x = self.layernorm2(x)
            x = self.ff(x)
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
        enc_x = self.enc(x)
        out = self.dec(enc_x, x)
        return out
            


if __name__=='__main__':
    transformer = Transformer()
    x = torch.rand((4, 16, 512))
    print(x.size())
    out = transformer(x)
    print(out.size())