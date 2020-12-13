# -*- coding: utf-8 -*-
# File   : detect.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:58
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from collections import Counter
from utils.box.bbox import decode
from utils.box.rbbox import rbbox_batched_nms as nms


def detect(pred_cls, pred_loc, anchors, variance, conf_thresh, nms_thresh, top_n):
    scores = torch.sigmoid(pred_cls)
    bboxes = decode(pred_loc, anchors[None], variance)
    indexes_img, indexes_anchor, indexes_cls = torch.where(scores > conf_thresh)

    bboxes = bboxes[indexes_img, indexes_anchor]
    scores = scores[indexes_img, indexes_anchor, indexes_cls]
    labels = indexes_cls

    start = 0
    dets = [None] * pred_cls.size(0)
    for image_id, n in sorted(Counter(indexes_img.tolist()).items()):
        bboxes_ = bboxes[start: start + n]
        scores_ = scores[start: start + n]
        labels_ = labels[start: start + n]
        keeps = nms(bboxes_, scores_, labels_, nms_thresh)[:top_n]
        dets[image_id] = [bboxes_[keeps], scores_[keeps], labels_[keeps]]
        start += n

    return dets
