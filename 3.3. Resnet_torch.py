import torch
import torch.nn as nn
import sys

class shallowBlock(nn.Module):
    def __init__(self, 
                 in_filters:int,
                 filters:int,
                 kernel_size:int=3, 
                 stride:int=1,
                 filter_expand:int=1,
                 first:bool=None
                 ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.filter_expand = filter_expand
        self.in_filters = in_filters
        self.filters = filters
        self.start_filters = self.in_filters * 2
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_filters, self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=self.kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        self.shortcut = nn.Identity()
        if (self.stride == 2) or (self.in_filters != self.filters * self.filter_expand):
            self.shortcut = nn.Sequential(*[nn.Conv2d(self.in_filters, self.filters * self.filter_expand, kernel_size=1, stride=self.stride, bias=False),
                                            nn.BatchNorm2d(self.filters * self.filter_expand)])
        self.relu3 = nn.ReLU()
        
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.shortcut(inp)
        x += shortcut
        x = self.relu3(x)
        return x

class deepBlock(nn.Module):
    def __init__(self, 
                 in_filters:int,
                 filters:int,
                 kernel_size:int=1, 
                 stride:int=1,
                 filter_expand:int=4,
                 first:bool=True
                 ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_filters = in_filters if first else in_filters * filter_expand
        self.filters = filters
        self.stride = stride
        self.filter_expand = filter_expand
        self.conv1 = nn.Conv2d(self.in_filters, self.filters, kernel_size=self.kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=self.kernel_size*3, stride=self.stride, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.filters, self.filters * self.filter_expand, kernel_size=self.kernel_size, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=self.filters * self.filter_expand)
        self.shortcut = nn.Identity()
        if (self.stride == 2) or (self.in_filters != self.filters * self.filter_expand):
            self.shortcut = nn.Sequential(*[nn.Conv2d(self.in_filters, self.filters * self.filter_expand, kernel_size=1, stride=self.stride, bias=False),
                                            nn.BatchNorm2d(self.filters * self.filter_expand)])
        self.relu3 = nn.ReLU()
        
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        shortcut = self.shortcut(inp)
        x += shortcut
        x = self.relu3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, batch_size, in_channel:int=3, inter_channel:int=64, mode:str='resnet18') -> None:
        super().__init__()
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        self.batch_size = batch_size
        self.filter_expand = 4
        assert mode in ['resnet' + str(i) for i in [18, 34, 50, 101, 152]], \
            f"Only supports resnet-18, 34, 50, 101, 152. Given input : {mode}"
        if mode=='resnet18':
            self.module_list = [2, 2, 2, 2]
        elif mode=='resnet34' or mode=='resnet50':
            self.module_list = [3, 4, 6, 3]
        elif mode=='resnet101':
            self.module_list = [3, 4, 23, 3]
        else:
            self.module_list = [3, 8, 36, 3]

        if mode in ['resnet18', 'resnet34']:
            block = shallowBlock
            isdeep = False
        else:
            block = deepBlock
            isdeep = True
        
        block_list= []
        for idx, L in enumerate(self.module_list):
            first_block = True if idx==0 else False
            filter_num = 64 * (2**idx)

            for l in range(L):
                if l==0 and first_block:
                    block_list.append(block(in_filters=filter_num, filters=filter_num, stride=1, first=isdeep))
                elif l==0 and not first_block:
                    if isdeep:
                        isdeep = False
                    block_list.append(block(in_filters=filter_num//2, filters=filter_num, stride=2, first=isdeep))
                else:
                    if isdeep:
                        isdeep = False
                    block_list.append(block(in_filters=filter_num, filters=filter_num, stride=1, first=isdeep))
            
        block_list.append(nn.AvgPool2d(4))

        self.conv = nn.Conv2d(self.in_channel, self.inter_channel, stride=2, kernel_size=7, padding=3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*(block_list[:self.module_list[0]]))
        self.layer2 = nn.Sequential(*(block_list[self.module_list[0]:sum(self.module_list[0:2])]))
        self.layer3 = nn.Sequential(*(block_list[sum(self.module_list[0:2]):sum(self.module_list[0:3])]))
        self.layer4 = nn.Sequential(*(block_list[sum(self.module_list[0:3]):sum(self.module_list)]))
        self.sm = nn.Softmax()

    def forward(self, x):
        print(x.size())
        x = self.conv(x)
        print(x.size())
        x = self.maxp(x)
        print(x.size())
        x = self.layer1(x)
        print(x.size())
        x = self.layer2(x)
        print(x.size())
        x = self.layer3(x)
        print(x.size())
        x = self.layer4(x)
        print(x.size())
        x = nn.Linear(x.view(self.batch_size, -1).size()[1], 1000)(x.view(self.batch_size, -1))
        out = self.sm(x)
        return out

if __name__=='__main__':
    _, mode = sys.argv
    module = ResNet(8, mode=mode)
    # print(module)
    x = torch.rand((1,3,224,224))
    out = module(x)
    print(out.size())