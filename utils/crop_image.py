# -*- coding: utf-8 -*-
# File   : crop_image.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:09
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import os
import json
import cv2 as cv
import numpy as np
import multiprocessing

from copy import deepcopy

from .box.bbox_np import xy42xywha
from .image import imread, imwrite


class Cropper:
    def __init__(self, size, overlap):
        self.sizes = sorted([size] if isinstance(size, int) else size)
        self.overlap = overlap

    @staticmethod
    def crop_bbox(objs, roi):
        sub_objs = []
        x1, y1, x2, y2 = roi
        roi = ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), 0
        for obj in objs:
            x, y, w, h, a = xy42xywha(np.array(obj['bbox'], dtype=np.float32))
            inter_points = cv.rotatedRectangleIntersection(roi, ((x, y), (w, h), a))[1]
            if inter_points is not None:
                order_points = cv.convexHull(inter_points, returnPoints=True)
                inter_area = cv.contourArea(order_points)
                iou = inter_area / (w * h)
                if iou > 0.5:
                    sub_bbox = cv.boxPoints(cv.minAreaRect(order_points)) - [x1, y1]
                    obj = deepcopy(obj)
                    obj['bbox'] = sub_bbox.tolist()
                    sub_objs.append(obj)
        return sub_objs

    def crop_with_anno(self, path_img, path_anno, out_dir_images, out_dir_annos, save_empty=False):
        print('crop:', path_img, path_anno)
        img = imread(path_img)
        ih, iw = img.shape[:2]
        name = os.path.splitext(os.path.basename(path_img))[0]
        anno = [] if path_anno is None else json.load(open(path_anno))
        for i, size in enumerate(self.sizes):
            if i > 0 and (max if save_empty else min)(iw, ih) < self.sizes[i - 1]:
                break
            stride = int(size * (1 - self.overlap))
            for x in range(0, iw, stride):
                for y in range(0, ih, stride):
                    w, h = size, size
                    if x + size > iw:
                        x = max(0, iw - size)
                        w = iw - x
                    if y + size > ih:
                        y = max(0, ih - size)
                        h = ih - y
                    save_name = '%s-%d-%d-%d-%d' % (name, x, y, w, h)
                    sub_anno = self.crop_bbox(anno, (x, y, x + w, y + h))
                    if sub_anno:
                        json.dump(sub_anno, open(os.path.join(out_dir_annos, save_name + '.json'), 'wt'), indent=2)
                    if sub_anno or save_empty:
                        save_path = os.path.join(out_dir_images, save_name + '.jpg')
                        sub_img = img[y: y + h, x: x + w]
                        imwrite(sub_img, save_path)

    def crop_batch(self, pairs, out_dir_images, out_dir_annos, save_empty=False):
        os.makedirs(out_dir_images, exist_ok=True)
        os.makedirs(out_dir_annos, exist_ok=True)
        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8))
        for image, anno in pairs:
            pool.apply_async(self.crop_with_anno, (image, anno, out_dir_images, out_dir_annos, save_empty))
        pool.close()
        pool.join()
