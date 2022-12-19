import torch
import torch.nn as nn
import torch.nn.functional as F

class NLNN(nn.Module):
    def __init__(self,
                 in_channels,
                 mode='embedded_gaussian') -> None:
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        assert self.mode in ['embedded_gausian', 'gaussian', 'dot', 'concat']
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding='same')
        self.conv4 = nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1, padding='same')

    def forward(self, x):
        bs, c, h, w = x.size()
        theta = self.conv1(x)
        phi = self.conv2(x)
        g = self.conv3(x)
        theta = theta.view(bs, -1, c)
        phi = phi.view(bs, c, -1)
        matmul = torch.matmul(theta, phi)
        if self.mode == 'gaussian':
            xi = x.view(bs, -1, c)
            xj = x.view(bs, c, -1)
            matmul = torch.exp(torch.matmul(xi, xj))
            C = torch.sum(matmul, dim=1)
            out = matmul / C
        elif self.mode == 'embedded_gaussain':
            matmul = torch.exp(matmul)
            out = matmul / torch.sum(matmul, dim=1)
        elif self.mode == 'dot':
            N = h * w
            out = matmul / N
        else:
            cat = torch.cat([theta, phi], dim=1)
            bs_cat, c_cat, h_cat, w_cat = cat.size()
            self.W = nn.Linear(c_cat, c, bias=False)
            out = F.relu(self.W(cat)) / (h * w)
        out = torch.matmul(out, g)
        out = out.view(bs, c, h, w)
        out = self.conv4(out)
        out += x
        return out
        