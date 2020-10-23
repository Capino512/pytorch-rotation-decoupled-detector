

import torch

from collections import Counter
from utils.box.bbox import decode
from utils.box.rbbox import rbbox_batched_nms as nms


def detect(pred_cls, pred_loc, anchors, variance, conf_thresh, nms_thresh, top_k):
    scores = torch.sigmoid(pred_cls)
    bboxes = decode(pred_loc, anchors[None], variance)
    indexes_img, indexes_anchor, indexes_cls = torch.where(scores > conf_thresh)

    bboxes = bboxes[indexes_img, indexes_anchor]
    scores = scores[indexes_img, indexes_anchor, indexes_cls]
    labels = indexes_cls

    start = 0
    dets = [None] * pred_cls.size(0)
    for image_id, n in sorted(Counter(indexes_img.tolist()).items()):
        loc = bboxes[start: start + n]
        conf = scores[start: start + n]
        label = labels[start: start + n]
        keeps = nms(loc, conf, label, nms_thresh)[:top_k]
        dets[image_id] = [loc[keeps], conf[keeps], label[keeps]]
        start += n

    return dets
