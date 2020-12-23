# -*- coding: utf-8 -*-
# File   : priorbox.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:03
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from collections import OrderedDict

from utils.misc import LFUCache


class PriorBox:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prior_boxes = OrderedDict()

        for stride, size, aspects, scales in zip(cfg['strides'], cfg['sizes'], cfg['aspects'], cfg['scales']):
            self.prior_boxes[stride] = self._get_prior_box(stride, size, aspects, scales, cfg.get('old_version', False))

    @staticmethod
    def _get_prior_box(stride, size, aspects, scales, old_version=False):
        boxes = []
        if old_version:
            # To be compatible with previous weights
            pair = [[aspect, scale] for scale in scales for aspect in aspects]
        else:
            pair = [[aspect, scale] for aspect in aspects for scale in scales]
        for aspect, scale in pair:
            length = stride * size * scale
            if aspect == 1:
                boxes.append([length, length])
            else:
                boxes.append([length * aspect ** 0.5, length / aspect ** 0.5])
                boxes.append([length / aspect ** 0.5, length * aspect ** 0.5])
        return boxes

    @staticmethod
    def _get_anchors(img_size, prior_boxes):
        h, w = img_size
        anchors = []
        for stride, prior_box in prior_boxes:
            assert w % stride == 0 and h % stride == 0
            fmw, fmh = w // stride, h // stride
            prior_box = torch.tensor(prior_box, dtype=torch.float)
            offset_y, offset_x = torch.meshgrid([torch.arange(fmh), torch.arange(fmw)])
            offset_x = offset_x.to(prior_box) + 0.5
            offset_y = offset_y.to(prior_box) + 0.5
            offset = torch.stack([offset_x, offset_y], dim=-1) * stride
            offset = offset[:, :, None, :].repeat(1, 1, prior_box.size(0), 1)
            prior_box = prior_box[None, None, :, :].repeat(fmh, fmw, 1, 1)
            anchors.append(torch.cat([offset, prior_box], dim=-1).reshape(-1, 4))
        anchors = torch.cat(anchors)
        return anchors

    def get_anchors(self, img_size):
        return self._get_anchors(img_size, self.prior_boxes.items())


class LFUPriorBox:
    def __init__(self, prior_box_cfg, capacity=3):
        self.prior_box = PriorBox(prior_box_cfg)
        self.num_levels = len(self.prior_box.prior_boxes)
        self.num_prior_boxes = [len(prior_boxes) for prior_boxes in self.prior_box.prior_boxes.values()]
        self.lfu_cache = LFUCache(capacity)

    def get_anchors(self, img_size):
        name = 'anchors-%d-%d' % tuple(img_size)
        anchors = self.lfu_cache.get(name, None)
        if anchors is None:
            anchors = self.prior_box.get_anchors(img_size)
            self.lfu_cache.put(name, anchors)
        return anchors
