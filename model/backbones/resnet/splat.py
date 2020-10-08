

import torch

from torch import nn
from torch.nn import functional as F
from xtorch import xnn


class SplAtConv2d(xnn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', radix=2, reduction_factor=4):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(out_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.conv = xnn.Conv2d(out_channels * radix, kernel_size, stride, padding, dilation, groups * radix, bias, padding_mode)
        self.bn0 = xnn.BatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = xnn.Conv2d(inter_channels, 1, groups=groups)
        self.bn1 = xnn.BatchNorm2d()
        self.fc2 = xnn.Conv2d(out_channels * radix, 1, groups=groups)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        split = torch.chunk(x, self.radix, 1)
        gap = sum(split)
        gap = F.adaptive_avg_pool2d(gap, (1, 1))
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        atten = torch.chunk(atten, self.radix, 1)
        out = sum([att * split for (att, split) in zip(atten, split)])
        return out


class rSoftMax(xnn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        shape = x.shape
        if self.radix > 1:
            x = x.view(x.size(0), self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(shape)
        else:
            x = torch.sigmoid(x)
        return x
