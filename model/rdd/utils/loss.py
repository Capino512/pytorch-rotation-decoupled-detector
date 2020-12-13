# -*- coding: utf-8 -*-
# File   : loss.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:59
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from torch import nn
from collections import OrderedDict
from torch.nn.functional import one_hot
from utils.box.bbox import bbox_switch, bbox_iou, encode


def match(bboxes, anchors, iou_thresh, batch=16):
    # Reduce GPU memory usage
    ious = torch.cat([bbox_iou(bboxes[i: i + batch], anchors) for i in range(0, bboxes.size(0), batch)])
    max_ious, bbox_indexes = torch.max(ious, dim=0)
    mask_neg = max_ious < iou_thresh[0]
    mask_pos = max_ious > iou_thresh[1]
    return mask_pos, mask_neg, bbox_indexes


def calc_loss_v1(pred_cls, pred_loc, targets, anchors, iou_thresh, variance, balance):
    device = pred_cls.device
    num_classes = pred_cls.size(-1)
    weight_pos, weight_neg = 2 * balance, 2 * (1 - balance)
    anchors_xyxy = bbox_switch(anchors, 'xywh', 'xyxy')

    criterion_cls = nn.BCEWithLogitsLoss(reduction='none')
    criterion_loc = nn.SmoothL1Loss(reduction='sum')
    loss_cls, loss_loc = torch.zeros([2], dtype=torch.float, device=device, requires_grad=True)
    num_pos = 0
    for i, target in enumerate(targets):
        if target:
            bboxes = target['bboxes'].to(device)
            labels = target['labels'].to(device)
            bboxes_xyxy = bbox_switch(bboxes[:, :4], 'xywh', 'xyxy')
            mask_pos, mask_neg, bbox_indexes = match(bboxes_xyxy, anchors_xyxy, iou_thresh)

            labels = labels[bbox_indexes]
            indexes_pos = bbox_indexes[mask_pos]
            bboxes_matched = bboxes[indexes_pos]
            anchors_matched = anchors[mask_pos]
            bboxes_pred = pred_loc[i][mask_pos]
            gt_bboxes, det_bboxes = encode(bboxes_matched, bboxes_pred, anchors_matched, variance)

            labels = one_hot(labels, num_classes=num_classes).float()
            labels[mask_neg] = 0
            loss_cls_ = criterion_cls(pred_cls[i], labels)
            loss_cls = loss_cls + loss_cls_[mask_pos].sum() * weight_pos + loss_cls_[mask_neg].sum() * weight_neg
            loss_loc = loss_loc + criterion_loc(gt_bboxes, det_bboxes)
            num_pos += mask_pos.sum().item()
        else:
            loss_cls = loss_cls + criterion_cls(pred_cls[i], torch.zeros_like(pred_cls[i])).sum()
    num_pos = max(num_pos, 1)
    return OrderedDict([('loss_cls', loss_cls / num_pos), ('loss_loc', loss_loc / num_pos)])


def calc_loss_v2(pred_cls, pred_loc, targets, anchors, iou_thresh, variance, balance):
    # Calculate the loss centrally, has only a small acceleration effect
    device = pred_cls.device
    num_classes = pred_cls.size(-1)
    weight_pos, weight_neg = 2 * balance, 2 * (1 - balance)
    criterion_cls = nn.BCEWithLogitsLoss(reduction='none')
    criterion_loc = nn.SmoothL1Loss(reduction='sum')

    num_bboxes = [target['bboxes'].size(0) if target else 0 for target in targets]
    bboxes = [target['bboxes'] for target in targets if target]
    labels = [target['labels'] for target in targets if target]
    if len(bboxes) > 0:
        bboxes = torch.cat(bboxes).to(device)
        labels = torch.cat(labels).to(device)
    else:
        loss_cls = criterion_cls(pred_cls, torch.zeros_like(pred_cls)).sum()
        return OrderedDict([('loss_cls', loss_cls), ('loss_loc', torch.tensor(0., requires_grad=True))])

    # Reduce GPU memory usage
    batch = 16
    iou = torch.cat([bbox_iou(bboxes[i: i + batch, :4], anchors, 'xywh') for i in range(0, bboxes.size(0), batch)])
    start = 0
    max_iou_merged, bbox_indexes_merged = [], []
    for i, num in enumerate(num_bboxes):
        if num == 0:
            max_iou = torch.zeros_like(pred_cls[i, :, 0])
            bbox_indexes = torch.zeros_like(pred_cls[i, :, 0], dtype=torch.long)
        else:
            max_iou, bbox_indexes = torch.max(iou[start: start + num], dim=0)  # a
        max_iou_merged.append(max_iou)
        bbox_indexes_merged.append(bbox_indexes + start)
        start += num
    max_iou_merged = torch.stack(max_iou_merged)
    bbox_indexes_merged = torch.stack(bbox_indexes_merged)
    masks_pos = max_iou_merged > iou_thresh[1]
    masks_neg = max_iou_merged < iou_thresh[0]
    labels_matched = labels[bbox_indexes_merged]
    labels_matched = one_hot(labels_matched, num_classes=num_classes)
    labels_matched[masks_neg] = 0
    bboxes_matched = bboxes[bbox_indexes_merged[masks_pos]]
    anchors_matched = anchors[None].repeat(len(targets), 1, 1)[masks_pos]
    loss_cls = criterion_cls(pred_cls, labels_matched.float())
    loss_cls = loss_cls[masks_pos].sum() * weight_pos + loss_cls[masks_neg].sum() * weight_neg
    gt_bboxes, det_bboxes = encode(bboxes_matched, pred_loc[masks_pos], anchors_matched, variance)
    loss_loc = criterion_loc(det_bboxes, gt_bboxes)
    num_pos = max(masks_pos.sum().item(), 1)
    return OrderedDict([('loss_cls', loss_cls / num_pos), ('loss_loc', loss_loc / num_pos)])


calc_loss = calc_loss_v1
