

import os
import torch

from torch import nn
from xtorch import xnn
from config import DIR_WEIGHT
from utils.init import weight_init

# all pre-trained on image-net

weights = {
    # from YOLO-v3
    'darknet21': os.path.join(DIR_WEIGHT, 'darknet', 'darknet21.pth'),
    'darknet53': os.path.join(DIR_WEIGHT, 'darknet', 'darknet53.pth'),
}


def CBR(plane, kernel_size, stride=1, padding=0):
    return nn.Sequential(xnn.Conv2d(plane, kernel_size, stride, padding, bias=False),
                         xnn.BatchNorm2d(),
                         nn.ReLU(inplace=True))


class BasicBlock(xnn.Module):
    def __init__(self, plane):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(CBR(plane // 2, kernel_size=1, stride=1, padding=0),
                                  CBR(plane, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Backbone(xnn.Module):
    def __init__(self, layers, name=None, fetch_feature=False):
        super(Backbone, self).__init__()
        self.name = name
        self.fetch_feature = fetch_feature
        self.head = CBR(32, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList([self._make_layer(64 * 2 ** i, blocks) for i, blocks in enumerate(layers)])

    @staticmethod
    def _make_layer(plane, blocks):
        layers = [CBR(plane, kernel_size=3, stride=2, padding=1)]
        for i in range(0, blocks):
            layers.append(BasicBlock(plane))
        return nn.Sequential(*layers)

    def init(self):
        if self.name in weights:
            print('load pre-training weights for', self.name)
            weight = torch.load(weights[self.name])
            ret = self.load_state_dict(weight, strict=False)
            print(ret)
        else:
            self.apply(weight_init['normal'])

    def forward(self, x):
        feature = self.head(x)
        features = []
        for layer in self.layers:
            feature = layer(feature)
            if self.fetch_feature:
                features.append(feature)
        return features if self.fetch_feature else feature


def darknet21(fetch_feature=False):
    return Backbone([1, 1, 2, 2, 1], 'darknet21', fetch_feature)


def darknet53(fetch_feature=False):
    return Backbone([1, 2, 8, 8, 4], 'darknet53', fetch_feature)
