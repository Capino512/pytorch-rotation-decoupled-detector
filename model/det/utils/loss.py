

import torch

from torch import nn
from collections import OrderedDict
from torch.nn.functional import one_hot
from utils.box.bbox import bbox_switch, bbox_iou, encode


def match(bboxes, anchors, iou_thresh, batch=32):
    ious = []
    for i in range(0, len(bboxes), batch):
        ious.append(bbox_iou(bboxes[i:i + batch], anchors))
    ious = torch.cat(ious, dim=0)  # n * a
    ious_max0, indexes_max0 = torch.max(ious, dim=0)  # a
    mask_neg = ious_max0 < iou_thresh[0]
    mask_pos = ious_max0 > iou_thresh[1]
    return mask_pos, mask_neg, indexes_max0


def calc_loss(targets, pred_cls, pred_loc, anchors, num_classes, iou_thresh, variance):
    device = pred_cls.device
    anchors_xyxy = bbox_switch(anchors, 'xywh', 'xyxy')
    criterion_cls = nn.BCEWithLogitsLoss(reduction='none')
    criterion_loc = nn.SmoothL1Loss(reduction='sum')
    loss_cls, loss_loc = torch.zeros([2], dtype=torch.float, device=device, requires_grad=True)
    total_pos, total_pos_true = torch.zeros([2], dtype=torch.float, device=device, requires_grad=False)
    for i, target in enumerate(targets):
        if target:
            bboxes = target['bboxes']
            labels = target['labels']

            if pred_cls.is_cuda:
                bboxes = bboxes.cuda(device)
                labels = labels.cuda(device)
            bboxes_xyxy = bbox_switch(bboxes[:, :4], 'xywh', 'xyxy')
            mask_pos, mask_neg, indexes_max0 = match(bboxes_xyxy, anchors_xyxy, iou_thresh)

            labels = labels[indexes_max0]
            indexes_pos = indexes_max0[mask_pos]
            bboxes_pos = bboxes[indexes_pos]
            anchors_pos = anchors[mask_pos]
            bboxes_pred = pred_loc[i][mask_pos]
            bboxes_gt, bboxes_pred = encode(bboxes_pos, bboxes_pred, anchors_pos, variance)

            labels = one_hot(labels, num_classes=num_classes).float()
            labels[mask_neg] = 0
            mask_pos_neg = (mask_pos | mask_neg)
            loss_cls = loss_cls + criterion_cls(pred_cls[i], labels)[mask_pos_neg].sum()
            loss_loc = loss_loc + criterion_loc(bboxes_gt, bboxes_pred)
            num_pos = mask_pos.sum()
            total_pos += torch.clamp_min(num_pos, 1)
            total_pos_true += num_pos
        else:
            loss_cls = loss_cls + criterion_cls(pred_cls[i], torch.zeros_like(pred_cls[i])).sum()
            total_pos += 1
    loss_cls = loss_cls / total_pos
    if total_pos_true:
        loss_loc = loss_loc / total_pos_true
    return OrderedDict([('loss_cls', loss_cls), ('loss_loc', loss_loc)])
