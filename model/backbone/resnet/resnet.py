

import os
import torch

from torch import nn
from xtorch import xnn
from config import DIR_WEIGHT
from utils.init import weight_init

from .splat import SplAtConv2d

# all pre-trained on image-net

weights = {
    # from pytorch
    'resnet18': os.path.join(DIR_WEIGHT, 'resnet', 'resnet18-5c106cde.pth'),
    'resnet34': os.path.join(DIR_WEIGHT, 'resnet', 'resnet34-333f7ec4.pth'),
    'resnet50': os.path.join(DIR_WEIGHT, 'resnet', 'resnet50-19c8e357.pth'),
    'resnet101': os.path.join(DIR_WEIGHT, 'resnet', 'resnet101-5d3b4d8f.pth'),
    'resnet152': os.path.join(DIR_WEIGHT, 'resnet', 'resnet152-b121ed2d.pth'),
    'resnext50': os.path.join(DIR_WEIGHT, 'resnet', 'resnext50_32x4d-7cdf4587.pth'),
    'resnext101': os.path.join(DIR_WEIGHT, 'resnet', 'resnext101_32x8d-8ba56ff5.pth'),

    # from https://github.com/zhanghang1989/ResNeSt
    'resnest50': os.path.join(DIR_WEIGHT, 'resnet', 'resnest50-528c19ca.pth'),
    'resnest101': os.path.join(DIR_WEIGHT, 'resnet', 'resnest101-22405ba7.pth'),
    'resnest200': os.path.join(DIR_WEIGHT, 'resnet', 'resnest200-75117900.pth'),
    'resnest269': os.path.join(DIR_WEIGHT, 'resnet', 'resnest269-0cc87c48.pth'),

    'resnet50-d': os.path.join(DIR_WEIGHT, 'resnet', 'resnet50_v1d.pth'),
    'resnet101-d': os.path.join(DIR_WEIGHT, 'resnet', 'resnet101_v1d.pth'),
    'resnet152-d': os.path.join(DIR_WEIGHT, 'resnet', 'resnet152_v1d.pth'),
}


class BasicBlock(xnn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, *args, **kwargs):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(xnn.Conv2d(planes, 3, stride, 1, bias=False),
                                  xnn.BatchNorm2d(),
                                  nn.ReLU(inplace=True),
                                  xnn.Conv2d(planes, 3, 1, 1, bias=False),
                                  xnn.BatchNorm2d())
        self.downsample = downsample

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.body(x) + (x if self.downsample is None else self.downsample(x)))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, radix=1, cardinality=1, bottleneck_width=64, avd=False,
                 avd_first=False, dilation=1, is_first=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64)) * cardinality
        avd = avd and (stride > 1 or is_first)

        body = [xnn.Conv2d(group_width, kernel_size=1, bias=False), xnn.BatchNorm2d(), nn.ReLU(inplace=True)]
        if avd:
            avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
            if avd_first:
                body.append(avd_layer)
        if radix > 1:
            body.append(SplAtConv2d(group_width, 3, stride, dilation, dilation, cardinality, bias=False, radix=radix))
        else:
            body.append(xnn.Conv2d(group_width, 3, stride, dilation, dilation, cardinality, bias=False))
            body.append(xnn.BatchNorm2d())
            body.append(nn.ReLU(inplace=True))
        if avd and not avd_first:
            body.append(avd_layer)
        body.append(xnn.Conv2d(planes * self.expansion, 1, bias=False))
        body.append(xnn.BatchNorm2d())
        self.body = nn.Sequential(*body)
        self.downsample = downsample

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.body(x) + (x if self.downsample is None else self.downsample(x)))


class Backbone(xnn.Module):
    def __init__(self, block, layers, name=None, fetch_feature=False, radix=1, groups=1, bottleneck_width=64,
                 dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, avd=False, avd_first=False):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(Backbone, self).__init__()

        self.name = name
        self.fetch_feature = fetch_feature

        if deep_stem:
            head = [xnn.Conv2d(stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                    xnn.BatchNorm2d(),
                    nn.ReLU(inplace=True),
                    xnn.Conv2d(stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                    xnn.BatchNorm2d(),
                    nn.ReLU(inplace=True),
                    xnn.Conv2d(stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False)]
        else:
            head = [xnn.Conv2d(64, kernel_size=7, stride=2, padding=3, bias=False)]
        self.head = nn.Sequential(*head, xnn.BatchNorm2d(), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def init(self):
        if self.name in weights:
            print('load pre-training weights for', self.name)
            weight = torch.load(weights[self.name])
            ret = self.load_state_dict(weight, strict=False)
            print(ret)
        else:
            self.apply(weight_init['kaiming_normal'])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(
                        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(xnn.Conv2d(planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(xnn.Conv2d(planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(xnn.BatchNorm2d())
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(planes, stride, downsample, self.radix, self.cardinality, self.bottleneck_width, self.avd,
                      self.avd_first, 1, is_first))
        elif dilation == 4:
            layers.append(
                block(planes, stride, downsample, self.radix, self.cardinality, self.bottleneck_width, self.avd,
                      self.avd_first, 2, is_first))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(planes, 1, None, self.radix, self.cardinality, self.bottleneck_width, self.avd, self.avd_first,
                      dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4] if self.fetch_feature else x4


def resnet18(fetch_feature=False):
    return Backbone(BasicBlock, (2, 2, 2, 2), 'resnet18', fetch_feature)


def resnet34(fetch_feature=False):
    return Backbone(BasicBlock, (3, 4, 6, 3), 'resnet34', fetch_feature)


def resnet50(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 6, 3), 'resnet50', fetch_feature)


def resnet101(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 23, 3), 'resnet101', fetch_feature)


def resnet152(fetch_feature=False):
    return Backbone(Bottleneck, (3, 8, 36, 3), 'resnet152', fetch_feature)


def resnest50(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 6, 3), 'resnest50', fetch_feature, radix=2, deep_stem=True, stem_width=32,
                    avg_down=True, avd=True, avd_first=False)


def resnest101(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 23, 3), 'resnest101', fetch_feature, radix=2, deep_stem=True, stem_width=64,
                    avg_down=True, avd=True, avd_first=False)


def resnest200(fetch_feature=False):
    return Backbone(Bottleneck, (3, 24, 36, 3), 'resnest200', fetch_feature, radix=2, deep_stem=True, stem_width=64,
                    avg_down=True, avd=True, avd_first=False)


def resnest269(fetch_feature=False):
    return Backbone(Bottleneck, (3, 30, 48, 8), 'resnest269', fetch_feature, radix=2, deep_stem=True, stem_width=64,
                    avg_down=True, avd=True, avd_first=False)


def resnext50_32x4d(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 6, 3), 'resnext50-32x4d', fetch_feature, groups=32, bottleneck_width=4)


def resnext101_32x8d(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 23, 3), 'resnext101-32x8d', fetch_feature, groups=32, bottleneck_width=8)


def resnet18_d(fetch_feature=False):
    return Backbone(BasicBlock, (2, 2, 2, 2), 'resnet18-d', fetch_feature, deep_stem=True, stem_width=32)


def resnet34_d(fetch_feature=False):
    return Backbone(BasicBlock, (3, 4, 6, 3), 'resnet34-d', fetch_feature, deep_stem=True, stem_width=32)


def resnet50_d(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 6, 3), 'resnet50-d', fetch_feature, deep_stem=True, stem_width=32)


def resnet101_d(fetch_feature=False):
    return Backbone(Bottleneck, (3, 4, 23, 3), 'resnet101-d', fetch_feature, deep_stem=True, stem_width=32)


def resnet152_d(fetch_feature=False):
    return Backbone(Bottleneck, (3, 8, 36, 3), 'resnet152-d', fetch_feature, deep_stem=True, stem_width=32)
