# -*- coding: utf-8 -*-
# File   : bbox.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:08
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from torchvision.ops.boxes import nms, batched_nms, box_iou


def bbox_switch(bbox, in_type, out_type):  # 'xyxy', 'xywh'
    if in_type == 'xyxy' and out_type == 'xywh':
        bbox = torch.cat([(bbox[..., 0: 2] + bbox[..., 2: 4]) / 2, bbox[..., 2: 4] - bbox[..., 0: 2]], dim=-1)
    elif in_type == 'xywh' and out_type == 'xyxy':
        bbox = torch.cat([bbox[..., 0: 2] - bbox[..., 2: 4] / 2, bbox[..., 0: 2] + bbox[..., 2: 4] / 2], dim=-1)
    return bbox


def bbox_iou(bbox1, bbox2, bbox_type='xyxy'):  # nx4, mx4 -> nxm
    bbox1 = bbox_switch(bbox1, bbox_type, 'xyxy')
    bbox2 = bbox_switch(bbox2, bbox_type, 'xyxy')
    return box_iou(bbox1, bbox2)


def bbox_nms(bboxes, scores, iou_thresh):
    return nms(bboxes, scores, iou_thresh)


def bbox_batched_nms(bboxes, scores, labels, iou_thresh):
    return batched_nms(bboxes, scores, labels, iou_thresh)


def encode(gt_bbox, det_bbox, anchor, variance):
    xy = (gt_bbox[..., 0: 2] - anchor[..., 0: 2]) / anchor[..., 2: 4] / variance[0]
    wh = torch.log(gt_bbox[..., 2: 4] / anchor[..., 2: 4]) / variance[1]
    a = gt_bbox[..., [4]] / 45 / variance[2]
    gt_bbox = torch.cat([xy, wh, a], dim=-1)
    det_bbox = torch.cat([det_bbox[..., :4], torch.tanh(det_bbox[..., [4]]) / variance[2]], dim=-1)
    return gt_bbox, det_bbox


def decode(det_bbox, anchor, variance):
    xy = det_bbox[..., 0: 2] * variance[0] * anchor[..., 2: 4] + anchor[..., 0: 2]
    wh = torch.exp(det_bbox[..., 2: 4] * variance[1]) * anchor[..., 2: 4]
    a = torch.tanh(det_bbox[..., [4]]) * 45
    return torch.cat([xy, wh, a], dim=-1)
