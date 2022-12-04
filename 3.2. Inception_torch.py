import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, 
                 kernel1:int=1, 
                 kernel2:int=3, 
                 kernel3:int=5) -> None:
        super(InceptionModule, self).__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3


class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LocalRespNorm(nn.Module):
    def __init__(self, local=1, alpha=1.0, beta=0.75, k=1, channel_wise=False) -> None:
        super().__init__()
        self.channel_wise = channel_wise
        self.local = local
        self.alpha = alpha
        self.beta = beta
        self.k = k
        if self.channel_wise:
            self.avg = nn.AvgPool3d(kernel_size=(self.local, 1, 1), stride=1, padding=(int((self.local - 1.)/2), 0, 0))
        else:
            self.avg = nn.AvgPool2d(kernel_size=self.local, stride=1, padding=int((self.local-1.)/2))
        
    def forward(self, x):
        if self.channel_wise:
            div = (x**2).unsqueeze(1)
            div = self.avg(div).squeeze(1)
            div = (div * self.alpha + self.k).pow(self.beta)
        else:
            div = x**2
            div = self.avg(div)
            div = (div * self.alpha + self.k).pow(self.beta)
        x /= div
        return x


class BaseInception(nn.Module):
    def __init__(self, dim, size, config) -> None:
        super().__init__()
        self.dim = dim
        self.size = size
        self.config = config

        self.conv1 = nn.Conv2d(self.size, self.config[0][0], kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(self.size, self.config[1][0], kernel_size=1, stride=1, padding=0)
        self.relu3_1 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(self.config[1][0], self.config[1][1], kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()

        self.conv5_1 = nn.Conv2d(self.size, self.config[2][0], kernel_size=1, stride=1, padding=0)
        self.relu5_1 = nn.ReLU()
        self.conv5_5 = nn.Conv2d(self.config[2][0], self.config[2][1], kernel_size=5, stride=1, padding=2)
        self.relu5_5 = nn.ReLU()
        
        self.max1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)

        self.convmax = nn.Conv2d(self.size, config[3][1], kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

    def forward(self, inp):
        x1 = self.conv1(inp)
        x1 = self.relu1(x1)

        x2 = self.conv3_1(inp)
        x2 = self.relu3_1(x2)
        x2 = self.conv3_3(x2)
        x2 = self.relu3_3(x2)

        x3 = self.conv5_1(inp)
        x3 = self.relu5_1(x3)
        x3 = self.conv5_5(x3)
        x3 = self.relu5_5(x3)

        x4 = self.max1(inp)
        x4 = self.convmax(x4)
        x4 = self.relu4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=self.dim)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels:int=3, inter_channels:int=64, alpha_const=0.00109999) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.alpha_const = alpha_const

        self.conv1 = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = LocalRespNorm(local=11, alpha=self.alpha_const, beta=0.5, k=2)

        self.conv2 = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.inter_channels, self.inter_channels * 3, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.lrn3 = LocalRespNorm(local=11, alpha=self.alpha_const, beta=0.5, k=2)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep_3a = BaseInception(1, 192, [[64], [96, 128], [16, 32], [3, 32]])
        self.incep_3b = BaseInception(1, 256, [[128], [128, 192], [32, 96], [3, 64]])
        self.max_inc3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.incep_4a = BaseInception(1, 480, [[192], [ 96,204], [16, 48], [3, 64]])
        self.incep_4b = BaseInception(1, 508, [[160], [112,224], [24, 64], [3, 64]])
        self.incep_4c = BaseInception(1, 512, [[128], [128,256], [24, 64], [3, 64]])
        self.incep_4d = BaseInception(1, 512, [[112], [144,288], [32, 64], [3, 64]])
        self.incep_4e = BaseInception(1, 528, [[256], [160,320], [32,128], [3,128]])
        self.max_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep_5a = BaseInception(1, 832, [[256], [160,320], [48,128], [3,128]])
        self.incep_5b = BaseInception(1, 832, [[384], [192,384], [48,128], [3,128]])
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 1000)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1(x)
        x = self.lrn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.lrn3(x)
        x = self.max3(x)

        x = self.incep_3a(x)
        x = self.incep_3b(x)
        x = self.max_inc3(x)

        x = self.incep_4a(x)
        x = self.incep_4b(x)
        x = self.incep_4c(x)
        x = self.incep_4d(x)
        x = self.incep_4e(x)
        x = self.max_inc4(x)

        x = self.incep_5a(x)
        x = self.incep_5b(x)
        x = self.avg_pool(x)

        x = x.view(-1, 1024)

        x = self.dropout(x)
        x = self.linear(x)

        x = self.sm(x)

        return x

if __name__=='__main__':
    module = Inception()
    # print(module)
    x = torch.rand((1,3,224,224))
    out = module(x)
    print(out.size())