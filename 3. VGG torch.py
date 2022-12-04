import torch
import torch.nn as nn
import numpy as np
import sys

TRAIN_MEAN = 123456

class VGG(nn.Module):
    def __init__(self, 
                 config='D',
                 batch_size=16,
                 in_channels:int=3,
                 out_channels:int=64, 
                 class_num:int=1000) -> None:
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.batch_size=batch_size
        self.out_channels = out_channels
        self.class_num = class_num
        self.config = config
        configs = ['A', 'B', 'C', 'D', 'E']
        assert config in configs, "Configuration (config) shold be one of "+\
                                 f"{configs}. Given configurationo : {self.config}."
        self.config_dict = {"A":1, "B":2, "C":3, "D":4, "E":5}
        if self.config_dict[self.config]>0:
            pass
        if self.config_dict[self.config]>1:
            self.conv12 = nn.Conv2d(self.out_channels*1,
                                    self.out_channels*1,
                                    kernel_size=3,
                                    stride=1)
            self.conv22 = nn.Conv2d(self.out_channels*2,
                                    self.out_channels*2,
                                    kernel_size=3,
                                    stride=1)
        if self.config_dict[self.config]>2:
            self.conv33 = nn.Conv2d(self.out_channels*4,
                                   self.out_channels*4,
                                   kernel_size=1,
                                   stride=1)
            self.conv43 = nn.Conv2d(self.out_channels*8,
                                   self.out_channels*8,
                                   kernel_size=1,
                                   stride=1)
            self.conv53 = nn.Conv2d(self.out_channels*8,
                                   self.out_channels*8,
                                   kernel_size=1,
                                   stride=1)
        if self.config_dict[self.config]>3:
            self.conv33 = nn.Conv2d(self.out_channels*4,
                                    self.out_channels*4,
                                    kernel_size=3,
                                    stride=1)
            self.conv43 = nn.Conv2d(self.out_channels*8,
                                    self.out_channels*8,
                                    kernel_size=3,
                                    stride=1)
            self.conv53 = nn.Conv2d(self.out_channels*8,
                                    self.out_channels*8,
                                    kernel_size=3,
                                    stride=1)
        if self.config_dict[self.config]>4:
            self.conv34 = nn.Conv2d(self.out_channels*4,
                                    self.out_channels*4,
                                    kernel_size=3,
                                    stride=1)
            self.conv44 = nn.Conv2d(self.out_channels*8,
                                    self.out_channels*8,
                                    kernel_size=3,
                                    stride=1)
            self.conv54 = nn.Conv2d(self.out_channels*8,
                                    self.out_channels*8,
                                    kernel_size=3,
                                    stride=1)
        
        self.conv11 = nn.Conv2d(self.in_channels,
                                self.out_channels,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.max1 = nn.MaxPool2d(kernel_size=2,
                                stride=2)
        self.conv21 = nn.Conv2d(self.out_channels,
                                self.out_channels*2,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.max2 = nn.MaxPool2d(kernel_size=2,
                                stride=2)
        self.conv31 = nn.Conv2d(self.out_channels*2,
                                self.out_channels*4,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.conv32 = nn.Conv2d(self.out_channels*4,
                                self.out_channels*4,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.max3 = nn.MaxPool2d(kernel_size=2,
                                stride=2)
        self.conv41 = nn.Conv2d(self.out_channels*4,
                                self.out_channels*8,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.conv42 = nn.Conv2d(self.out_channels*8,
                                self.out_channels*8,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.max4 = nn.MaxPool2d(kernel_size=2,
                                stride=2)
        self.conv51 = nn.Conv2d(self.out_channels*8,
                                self.out_channels*8,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.conv52 = nn.Conv2d(self.out_channels*8,
                                self.out_channels*8,
                                kernel_size=3,
                                stride=1,
                                padding='same')
        self.max5 = nn.MaxPool2d(kernel_size=2,
                                stride=2)
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu21 = nn.ReLU()
        self.relu22 = nn.ReLU()
        self.relu31 = nn.ReLU()
        self.relu32 = nn.ReLU()
        self.relu33 = nn.ReLU()
        self.relu34 = nn.ReLU()
        self.relu41 = nn.ReLU()
        self.relu42 = nn.ReLU()
        self.relu43 = nn.ReLU()
        self.relu44 = nn.ReLU()
        self.relu51 = nn.ReLU()
        self.relu52 = nn.ReLU()
        self.relu53 = nn.ReLU()
        self.relu54 = nn.ReLU()
        self.fc1relu = nn.ReLU()
        self.fc2relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.sm = nn.Softmax()
    def forward(self, x):
        x = self.conv11(x)
        x = self.relu11(x)
        if self.config_dict[self.config]>1:
            x = self.conv12(x)
            x = self.relu12(x)
        x = self.max1(x)
        x = self.conv21(x)
        x = self.relu21(x)
        if self.config_dict[self.config]>1:
            x = self.conv22(x)
            x = self.relu22(x)
        x = self.max2(x)
        x = self.conv31(x)
        x = self.relu31(x)
        x = self.conv32(x)
        x = self.relu32(x)
        if self.config_dict[self.config]>2:
            x = self.conv33(x)
            x = self.relu33(x)
        if self.config_dict[self.config]>4:
            x = self.conv34(x)
            x = self.relu34(x)
        x = self.max3(x)
        x = self.conv41(x)
        x = self.relu41(x)
        x = self.conv42(x)
        x = self.relu42(x)
        if self.config_dict[self.config]>3:
            x = self.conv43(x)
            x = self.relu43(x)
        if self.config_dict[self.config]>4:
            x = self.conv44(x)
            x = self.relu44(x)
        x = self.max4(x)
        x = self.conv51(x)
        x = self.relu51(x)
        x = self.conv52(x)
        x = self.relu52(x)
        if self.config_dict[self.config]>3:
            x = self.conv53(x)
            x = self.relu53(x)
        if self.config_dict[self.config]>4:
            x = self.conv54(x)
            x = self.relu54(x)
        x = self.max5(x)
        x = x.view(self.batch_size, -1)
        width = x.size()[1]
        self.fc1 = nn.Linear(width, 4096)
        x = self.fc1(x)
        x = self.fc1relu(x)
        x = self.fc2(x)
        x = self.fc2relu(x)
        x = self.fc3(x)
        out = self.sm(x)
        return out

if __name__=='__main__':
    _, config = sys.argv
    batch_size = 16
    x = torch.rand((batch_size, 3, 224, 224))
    vgg = VGG(config, batch_size)
    out = vgg(x)
    print(out.size())