# -*- coding: utf-8 -*-
# File   : rdd.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:58
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from torch import nn
from xtorch import xnn
from utils.init import weight_init

from .utils.modules import FeaturePyramidNet, DetPredict
from .utils.priorbox import LFUPriorBox
from .utils.loss import calc_loss
from .utils.detect import detect


class RDD(xnn.Module):
    def __init__(self, backbone, cfg):
        super(RDD, self).__init__()

        cfg.setdefault('iou_thresh', [0.4, 0.5])
        cfg.setdefault('variance', [0.1, 0.2, 0.1])
        cfg.setdefault('balance', 0.5)

        cfg.setdefault('conf_thresh', 0.01)
        cfg.setdefault('nms_thresh', 0.5)
        cfg.setdefault('top_n', None)

        cfg.setdefault('extra', 0)
        cfg.setdefault('fpn_plane', 256)
        cfg.setdefault('extra_plane', 512)

        self.backbone = backbone
        self.prior_box = LFUPriorBox(cfg['prior_box'])
        self.num_levels = self.prior_box.num_levels
        self.num_classes = cfg['num_classes']
        self.iou_thresh = cfg['iou_thresh']
        self.variance = cfg['variance']
        self.balance = cfg['balance']

        self.conf_thresh = cfg['conf_thresh']
        self.nms_thresh = cfg['nms_thresh']
        self.top_n = cfg['top_n']

        self.extra = cfg['extra']
        self.fpn_plane = cfg['fpn_plane']
        self.extra_plane = cfg['extra_plane']

        self.fpn = FeaturePyramidNet(self.num_levels, self.fpn_plane)
        self.predict = DetPredict(self.num_levels, self.fpn_plane, self.prior_box.num_prior_boxes, self.num_classes, 5)

        if self.extra > 0:
            self.extra_layers = nn.ModuleList()
            for i in range(self.extra):
                self.extra_layers.append(nn.Sequential(xnn.Conv2d(self.extra_plane, 3, 2, 1, bias=False),
                                                       xnn.BatchNorm2d(),
                                                       nn.ReLU(inplace=True)))

    def init(self):
        self.apply(weight_init['normal'])
        self.backbone.init()

    def restore(self, path):
        weight = torch.load(path)
        self.load_state_dict(weight, strict=True)

    def forward(self, images, targets=None):
        features = list(self.backbone(images))
        features = features[-(self.num_levels - self.extra):]
        if self.extra > 0:
            for layer in self.extra_layers:
                features.append(layer(features[-1]))
        features = self.fpn(features)

        pred_cls, pred_loc = self.predict(features)
        anchors = self.prior_box.get_anchors(images.shape[2:]).to(images)
        if self.training:
            if targets is not None:
                return calc_loss(pred_cls, pred_loc, targets, anchors, self.iou_thresh, self.variance, self.balance)
        else:
            pred_cls, pred_loc = pred_cls.detach(), pred_loc.detach()
            top_n = (images.size(2) // 32) * (images.size(3) // 32) if self.top_n is None else self.top_n
            return detect(pred_cls, pred_loc, anchors, self.variance, self.conf_thresh, self.nms_thresh, top_n)
