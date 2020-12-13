

import numpy as np


try:
    from .ext.rbbox_overlap_gpu import rbbox_iou as rbbox_iou_gpu
    from .ext.rbbox_overlap_gpu import rbbox_nms as rbbox_nms_gpu


    def rbbox_iou(boxes1, boxes2, device=None):  # [x, y, w, h, a]
        if device is None:
            device = 0 if boxes1.device.type == 'cpu' else boxes1.device.index
        boxes1 = boxes1.reshape([-1, 5]).detach().cpu().numpy().astype(np.float32)
        boxes2 = boxes2.reshape([-1, 5]).detach().cpu().numpy().astype(np.float32)
        ious = rbbox_iou_gpu(boxes1, boxes2, device)
        return ious

    def rbbox_nms(boxes, scores, iou_thresh=0.5, device=None):
        if device is None:
            device = 0 if boxes.device.type == 'cpu' else boxes.device.index
        boxes = boxes.reshape([-1, 5]).detach().cpu().numpy().astype(np.float32)
        scores = scores.reshape([-1, 1]).detach().cpu().numpy().astype(np.float32)
        boxes = np.c_[boxes, scores]
        keeps = rbbox_nms_gpu(boxes, iou_thresh, device)
        return keeps

except ModuleNotFoundError as e:

    from .ext.rbbox_overlap_cpu import rbbox_iou_nxn as rbbox_iou_cpu
    from .ext.rbbox_overlap_cpu import rbbox_nms as rbbox_nms_cpu


    def rbbox_iou(boxes1, boxes2):
        boxes1 = boxes1.reshape([-1, 5]).detach().cpu().numpy().astype(np.float64)
        boxes2 = boxes2.reshape([-1, 5]).detach().cpu().numpy().astype(np.float64)
        ious = rbbox_iou_cpu(boxes1, boxes2)
        return ious


    def rbbox_nms(boxes, scores, iou_thresh=0.5):
        boxes = boxes.reshape([-1, 5]).detach().cpu().numpy().astype(np.float64)
        scores = scores.reshape([-1]).detach().cpu().numpy().astype(np.float64)
        keeps = rbbox_nms_cpu(boxes, scores, iou_thresh)
        return keeps


def rbbox_batched_nms(boxes, scores, labels, iou_thresh=0.5):
    if len(boxes) == 0:
        return np.empty([0], dtype=np.int)
    max_coordinate = boxes[:, 0:2].max() + boxes[:, 2:4].max()
    labels = labels.to(boxes)
    offsets = labels * (max_coordinate + 1)
    boxes = boxes.clone()
    boxes[:, :2] += offsets[:, None]
    return rbbox_nms(boxes, scores, iou_thresh)
