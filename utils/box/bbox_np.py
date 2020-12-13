# -*- coding: utf-8 -*-
# File   : bbox_np.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:08
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import numpy as np


def bbox_switch(bbox, in_type, out_type):  # 'xyxy', 'xywh'
    if in_type == 'xyxy' and out_type == 'xywh':
        bbox = np.concatenate([(bbox[..., 0: 2] + bbox[..., 2: 4]) / 2, bbox[..., 2: 4] - bbox[..., 0: 2]], axis=-1)
    elif in_type == 'xywh' and out_type == 'xyxy':
        bbox = np.concatenate([bbox[..., 0: 2] - bbox[..., 2: 4] / 2, bbox[..., 0: 2] + bbox[..., 2: 4] / 2], axis=-1)
    return bbox


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def xy42xywha(xy4, flag=0):  # bbox(4x2) represents a rectangle
    # flag=0, 0 <= a < 180
    # flag=1, 0 <= a < 180, w >= h
    # flag=2, -45 <= a < 45
    x, y = np.mean(xy4, axis=0)
    diff01 = xy4[0] - xy4[1]
    diff03 = xy4[0] - xy4[3]
    w = np.sqrt(np.square(diff01).sum())
    h = np.sqrt(np.square(diff03).sum())
    if w >= h:
        a = np.rad2deg(np.arctan2(diff01[1], diff01[0]))
    else:
        a = np.rad2deg(np.arctan2(diff03[1], diff03[0])) + 90
    if flag > 0:
        if w < h:
            w, h = h, w
            a += 90
    a = (a % 180 + 180) % 180
    if flag > 1:
        if 45 <= a < 135:
            w, h = h, w
            a -= 90
        elif a >= 135:
            a -= 180
    return np.stack([x, y, w, h, a])
