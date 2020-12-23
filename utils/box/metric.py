# -*- coding: utf-8 -*-
# File   : metric.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:08
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import numpy as np

from collections import defaultdict, Counter

from .rbbox_np import rbbox_iou


def get_ap(recall, precision):
    recall = [0] + list(recall) + [1]
    precision = [0] + list(precision) + [0]
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    ap = sum((recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(recall)) if recall[i] != recall[i - 1])
    return ap * 100


def get_ap_07(recall, precision):
    ap = 0.
    for t in np.linspace(0, 1, 11, endpoint=True):
        mask = recall >= t
        if np.any(mask):
            ap += np.max(precision[mask]) / 11
    return ap * 100


def get_det_aps(detect, target, num_classes, iou_thresh=0.5, use_07_metric=False):
    # [[index, bbox, score, label], ...]
    aps = []
    for c in range(num_classes):
        target_c = list(filter(lambda x: x[3] == c, target))
        detect_c = filter(lambda x: x[3] == c, detect)
        detect_c = sorted(detect_c, key=lambda x: x[2], reverse=True)
        tp = np.zeros(len(detect_c))
        fp = np.zeros(len(detect_c))
        target_count = Counter([x[0] for x in target_c])
        target_count = {index: np.zeros(count) for index, count in target_count.items()}
        target_lut = defaultdict(list)
        for index, bbox, conf, label in target_c:
            target_lut[index].append(bbox)
        detect_lut = defaultdict(list)
        for index, bbox, conf, label in detect_c:
            detect_lut[index].append(bbox)
        iou_lut = dict()
        for index, bboxes in detect_lut.items():
            if index in target_lut:
                iou_lut[index] = rbbox_iou(np.stack(bboxes), np.stack(target_lut[index]))
        counter = defaultdict(int)
        for i, (index, bbox, conf, label) in enumerate(detect_c):
            count = counter[index]
            counter[index] += 1
            iou_max = -np.inf
            hit_j = 0
            if index in iou_lut:
                for j, iou in enumerate(iou_lut[index][count]):
                    if iou > iou_max:
                        iou_max = iou
                        hit_j = j
            if iou_max > iou_thresh and target_count[index][hit_j] == 0:
                tp[i] = 1
                target_count[index][hit_j] = 1
            else:
                fp[i] = 1
        tp_sum = np.cumsum(tp)
        fp_sum = np.cumsum(fp)
        npos = len(target_c)
        recall = tp_sum / npos
        precision = tp_sum / (tp_sum + fp_sum)
        aps.append((get_ap_07 if use_07_metric else get_ap)(recall, precision))
    return aps
