# -*- coding: utf-8 -*-
# File   : modules.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:03
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from torch import nn
from xtorch import xnn


class FeaturePyramidNet(xnn.Module):
    def __init__(self, depth, plane):
        super(FeaturePyramidNet, self).__init__()
        self.link = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(depth):
            self.link.append(nn.Sequential(xnn.Conv2d(plane, 1, 1, 0, bias=False),
                                           xnn.BatchNorm2d()))
            if i != depth:
                self.fuse.append(nn.Sequential(nn.ReLU(inplace=True),
                                               xnn.Conv2d(plane, 3, 1, 1, bias=False),
                                               xnn.BatchNorm2d()))

    def forward(self, features):
        features = [self.link[i](feature) for i, feature in enumerate(features)]
        for i in range(len(features))[::-1]:
            if i != len(features) - 1:
                features[i] = self.fuse[i](features[i] + nn.Upsample(scale_factor=2)(features[i + 1]))
        features = [nn.ReLU(inplace=True)(feature) for feature in features]
        return features


class PredictHead(xnn.Module):
    def __init__(self, plane, num_anchors, num_classes):
        super(PredictHead, self).__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(xnn.Conv2d(plane, 3, 1, 1, bias=False),
                                  xnn.BatchNorm2d(),
                                  nn.ReLU(inplace=True),
                                  xnn.Conv2d(num_anchors * num_classes, 3, 1, 1))

    def forward(self, x):
        x = self.body(x)
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)


class DetPredict(xnn.Module):
    def __init__(self, depth, plane, num_anchors, num_classes, num_loc_params):
        super(DetPredict, self).__init__()
        self.heads_cls = nn.ModuleList()
        self.heads_loc = nn.ModuleList()
        for i in range(depth):
            self.heads_cls.append(PredictHead(plane, num_anchors[i], num_classes))
            self.heads_loc.append(PredictHead(plane, num_anchors[i], num_loc_params))

    def forward(self, features):
        predict_cls, predict_loc = [], []
        for i, feature in enumerate(features):
            predict_cls.append(self.heads_cls[i](feature))
            predict_loc.append(self.heads_loc[i](feature))
        predict_cls = torch.cat(predict_cls, dim=1)
        predict_loc = torch.cat(predict_loc, dim=1)
        return predict_cls, predict_loc
