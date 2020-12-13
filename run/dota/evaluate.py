# -*- coding: utf-8 -*-
# File   : evaluate.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:10
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import sys

sys.path.append('.')

import os
import tqdm
import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import DOTA

from model.rdd import RDD
from model.backbone import resnet

from utils.box.bbox_np import xywha2xy4, xy42xywha
from utils.box.rbbox_np import rbbox_batched_nms
from utils.parallel import CustomDetDataParallel


@torch.no_grad()
def main():
    global checkpoint
    if checkpoint is None:
        dir_weight = os.path.join(dir_save, 'weight')
        indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
        current_step = max(indexes)
        checkpoint = os.path.join(dir_weight, '%d.pth' % current_step)

    batch_size = 32
    num_workers = 4

    image_size = 768
    aug = ops.Resize(image_size)
    dataset = DOTA(dir_dataset, 'test', aug)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }
    conf_thresh = 0.01
    nms_thresh = 0.5
    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
        'conf_thresh': conf_thresh,
        'nms_thresh': nms_thresh
    }

    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model.restore(checkpoint)
    if len(device_ids) > 1:
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    model.eval()

    ret_raw = defaultdict(list)
    for images, targets, infos in tqdm.tqdm(loader):
        images = images.cuda() / 255
        dets = model(images)
        for (det, info) in zip(dets, infos):
            if det:
                bboxes, scores, labels = det
                bboxes = bboxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                fname, x, y, w, h = os.path.splitext(os.path.basename(info['img_path']))[0].split('-')[:5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                bboxes = np.array([xywha2xy4(bbox) for bbox in bboxes])
                bboxes *= [w / image_size, h / image_size]
                bboxes += [x, y]
                bboxes = np.array([xy42xywha(bbox) for bbox in bboxes])
                ret_raw[fname].append([bboxes, scores, labels])

    print('merging results...')
    ret = []

    for fname, dets in ret_raw.items():
        bboxes, scores, labels = zip(*dets)
        bboxes = np.concatenate(list(bboxes))
        scores = np.concatenate(list(scores))
        labels = np.concatenate(list(labels))
        keeps = rbbox_batched_nms(bboxes, scores, labels, nms_thresh)
        ret.append([fname, [bboxes[keeps], scores[keeps], labels[keeps]]])

    print('converting to submission format...')
    ret_save = defaultdict(list)
    for fname, (bboxes, scores, labels) in ret:
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = xywha2xy4(bbox).ravel()
            line = '%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f' % (fname, score, *bbox)
            ret_save[dataset.label2name[label]].append(line)

    print('saving...')
    os.makedirs(os.path.join(dir_save, 'submission'), exist_ok=True)
    for name, dets in ret_save.items():
        with open(os.path.join(dir_save, 'submission', 'Task%d_%s.txt' % (1, name)), 'wt') as f:
            f.write('\n'.join(dets))

    print('finished')


if __name__ == '__main__':

    device_ids = [0, 1]
    torch.cuda.set_device(device_ids[0])
    backbone = resnet.resnet101

    dir_dataset = '<replace with your local path>'
    dir_save = '<replace with your local path>'
    checkpoint = None

    main()
