# -*- coding: utf-8 -*-
# File   : ops_det.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:44
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import random
import cv2 as cv
import numpy as np

from utils.misc import containerize
from utils.box.bbox_np import xy42xywha, xywha2xy4

from ..func import *


__all__ = ['RandomHFlip', 'RandomVFlip', 'Resize', 'ResizeJitter', 'ResizeLong', 'ResizeBase', 'Pad', 'RandomPad',
           'PadSize', 'PadSquare', 'PadBase', 'Rotate', 'RandomRotate', 'RandomRotate90', 'RandomCrop', 'BBoxFilter']


class RandomHFlip:
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            if anno:
                ih, iw = img.shape[:2]
                anno['bboxes'][:, :, 0] = iw - 1 - anno['bboxes'][:, :, 0]
            img = hflip(img)
        return img, anno


class RandomVFlip:
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            if anno:
                ih, iw = img.shape[:2]
                anno['bboxes'][:, :, 1] = ih - 1 - anno['bboxes'][:, :, 1]
            img = vflip(img)
        return img, anno


class Resize:
    def __init__(self, size, interpolate='BILINEAR'):
        self.size = containerize(size, 2)
        self.interpolate = interpolate

    def __call__(self, img, anno=None):
        if anno:
            ih, iw = img.shape[:2]
            rw, rh = self.size
            bboxes = anno['bboxes'] * [rw / iw, rh / ih]
            # Convert to rectangle, if not it should not affect much
            anno['bboxes'] = np.array([cv.boxPoints(cv.minAreaRect(bbox)) for bbox in bboxes.astype(np.float32)])
        img = resize(img, self.size, self.interpolate)
        return img, anno


class ResizeJitter:
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        rh, rw = [ih, iw] * np.random.uniform(*self.scale, 2)
        img, anno = Resize((int(rw), int(rw)))(img, anno)
        return img, anno


class ResizeLong:
    def __init__(self, length, interpolate='BILINEAR'):
        self.length = length
        self.interpolate = interpolate

    def __call__(self, img,  anno=None):
        ih, iw = img.shape[:2]
        if ih > iw:
            size = (int(iw / ih * self.length), self.length)
        else:
            size = (self.length, int(ih / iw * self.length))
        return Resize(size, self.interpolate)(img, anno)


class ResizeBase:
    def __init__(self, base, scale=1., interpolate='BILINEAR'):
        self.base = base
        self.scale = scale
        self.interpolate = interpolate

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        rh, rw = int(ih * self.scale), int(iw * self.scale)
        rh = (rh - rh % self.base + self.base) if rh % self.base else rh
        rw = (rw - rw % self.base + self.base) if rw % self.base else rw
        return Resize((rw, rh), self.interpolate)(img, anno)


class _Pad:
    def get_padding(self, img):
        raise NotImplementedError

    def __call__(self, img, anno=None):
        padding = self.get_padding(img)
        if anno:
            anno['bboxes'] += [padding[1][0], padding[0][0]]
        img = pad(img, padding)
        return img, anno


class Pad(_Pad):
    def __init__(self, padding):
        if isinstance(padding, (int, float)):
            padding = [[padding, padding], [padding, padding]]
        else:
            padding = [containerize(p, 2) for p in padding]
        self.padding = padding

    def get_padding(self, img):
        (ph1, ph2), (pw1, pw2) = self.padding
        ih, iw = img.shape[:2]
        ph1 = ph1 if isinstance(ph1, int) else int(ph1 * ih)
        ph2 = ph2 if isinstance(ph2, int) else int(ph2 * ih)
        pw1 = pw1 if isinstance(pw1, int) else int(pw1 * iw)
        pw2 = pw2 if isinstance(pw2, int) else int(pw2 * iw)
        padding = [[ph1, ph2], [pw1, pw2]]
        return padding


class RandomPad:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        if isinstance(self.padding, float):
            ph = pw = int(max(ih, iw) * np.random.uniform(0, self.padding))
        else:
            ph = pw = random.randint(0, self.padding)
        ph1 = random.randint(0, ph)
        pw1 = random.randint(0, pw)
        return Pad([[ph1, ph - ph1], [pw1, pw - pw1]])(img, anno)


class PadSize(_Pad):
    def __init__(self, size, check_size=False):
        self.size = containerize(size, 2)
        self.check_size = check_size

    def get_padding(self, img):
        pw, ph = self.size
        ih, iw = img.shape[:2]
        if self.check_size:
            assert ih <= ph and iw <= pw
        padding = (max(0, ph - ih) // 2), max(0, (pw - iw) // 2)
        padding = [[padding[0], max(0, ph - ih - padding[0])], [padding[1], max(0, pw - iw - padding[1])]]
        return padding


class PadSquare:
    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        pw = ph = max(ih, iw)
        return PadSize([pw, ph])(img, anno)


class PadBase:
    def __init__(self, base):
        self.base = base

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        ph = (ih - ih % self.base + self.base) if ih % self.base else ih
        pw = (iw - iw % self.base + self.base) if iw % self.base else iw
        return PadSize((pw, ph))(img, anno)


class Rotate:
    def __init__(self, angle, scale=1, expand=False, shift=False):
        self.angle = angle
        self.scale = scale
        self.expand = expand
        self.shift = shift

    def __call__(self, img, anno=None):
        nh, nw = ih, iw = img.shape[:2]
        center = ((iw - 1) / 2, (ih - 1) / 2)
        m = cv.getRotationMatrix2D(center, angle=-self.angle, scale=self.scale)
        if self.expand or self.shift:
            corner = np.array([[0, 0, 1], [iw - 1, 0, 1], [iw - 1, ih - 1, 1], [0, ih - 1, 1]], dtype=np.float32)
            corner = np.matmul(m, corner.T).T
            left, top = np.min(corner, axis=0)
            right, bottom = np.max(corner, axis=0)
            dx = (right - left - iw) / 2
            dy = (bottom - top - ih) / 2
            if self.expand:
                nw = int(np.ceil(right - left))
                nh = int(np.ceil(bottom - top))
                shiftX = dx
                shiftY = dy
            else:
                shiftX = np.random.uniform(-dx, dx) if dx > 0 else 0
                shiftY = np.random.uniform(-dy, dy) if dy > 0 else 0
            m[0, 2] += shiftX
            m[1, 2] += shiftY
        if anno:
            bound = (nw / 2, nh / 2), (nw, nh), 0
            bboxes, labels = [], []
            for bbox, label in zip(anno['bboxes'], anno['labels']):
                corner = np.matmul(m, np.c_[bbox, np.ones((4, 1))].T).T
                if not self.expand:
                    x, y, w, h, a = xy42xywha(corner)
                    inter_points = cv.rotatedRectangleIntersection(bound, ((x, y), (w, h), a))[1]
                    if inter_points is not None:
                        order_points = cv.convexHull(inter_points, returnPoints=True)
                        inter_area = cv.contourArea(order_points)
                        iou = inter_area / (w * h)
                        if iou >= 0.5:
                            corner = cv.boxPoints(cv.minAreaRect(order_points))
                        else:
                            continue
                bboxes.append(corner)
                labels.append(label)
            if bboxes:
                anno['bboxes'] = np.stack(bboxes)
                anno['labels'] = np.stack(labels)
            else:
                anno = None
        img = cv.warpAffine(img, m, (nw, nh))
        return img, anno


class RandomRotate:
    def __init__(self, angle=180, scale=1, expand=False, shift=False):
        self.angle = (-angle, angle) if isinstance(angle, (int, float)) else angle
        self.scale = containerize(scale, 2)
        self.expand = expand
        self.shift = shift

    def __call__(self, img, anno=None):
        angle = np.random.uniform(*self.angle)
        scale = np.random.uniform(*self.scale)
        return Rotate(angle, scale, self.expand, self.shift)(img, anno)


class RandomRotate90:
    def __init__(self, k=(0, 1, 2, 3)):  # CLOCKWISE
        self.k = k

    def __call__(self, img, anno=None):
        k = np.random.choice(self.k)
        ih, iw = img.shape[:2]
        if anno:
            if k == 1:
                anno['bboxes'][:, :, 1] = ih - 1 - anno['bboxes'][:, :, 1]
                anno['bboxes'] = anno['bboxes'][:, :, [1, 0]]
            if k == 2:
                anno['bboxes'] = ([iw - 1, ih - 1] - anno['bboxes'])
            if k == 3:
                anno['bboxes'][:, :, 0] = iw - 1 - anno['bboxes'][:, :, 0]
                anno['bboxes'] = anno['bboxes'][:, :, [1, 0]]
        img = rotate90(img, k)
        return img, anno


class RandomCrop:
    def __init__(self, size, max_aspect=1.0, iou_thresh=0.5, max_try=100, nonempty=True):
        self.size = size
        self.max_aspect = max_aspect
        self.iou_thresh = iou_thresh
        self.max_try = max_try
        self.nonempty = nonempty

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        polygons = []
        if anno:
            for bbox in anno['bboxes']:
                x, y, w, h, a = xy42xywha(bbox)
                polygons.append(((x, y), (w, h), a))
        for count in range(self.max_try):
            if isinstance(self.size, int):
                nh = nw = min(ih, iw, self.size)
            else:
                if self.max_aspect == 1:
                    nh = nw = random.randint(min(ih, iw, self.size[0]), min(ih, iw, self.size[1]))
                else:
                    nh = random.randint(min(ih, self.size[0]), min(ih, self.size[1]))
                    nw = random.randint(min(iw, self.size[0]), min(iw, self.size[1]))
                    if max(nh / nw, nw / nh) > self.max_aspect:
                        continue
            oh = random.randint(0, ih - nh)
            ow = random.randint(0, iw - nw)
            a = np.random.uniform(0, 360)
            src = xywha2xy4([ow + nw / 2, oh + nh / 2, nw, nh, a])
            dst = np.array([[0, 0], [nw, 0], [nw, nh]], dtype=np.float32)
            m = cv.getAffineTransform(src.astype(np.float32)[:3], dst)
            if anno:
                bound = (ow + nw / 2, oh + nh / 2), (nw, nh), a
                iou, intersections = [], []
                for polygon in polygons:
                    inter_points = cv.rotatedRectangleIntersection(bound, polygon)[1]
                    if inter_points is None:
                        iou.append(0)
                        intersections.append(None)
                    else:
                        order_points = cv.convexHull(inter_points, returnPoints=True)
                        inter_area = cv.contourArea(order_points)
                        iou.append(inter_area / (polygon[1][0] * polygon[1][1]))
                        intersections.append(cv.boxPoints(cv.minAreaRect(order_points)))
                iou = np.array(iou)
                if isinstance(self.iou_thresh, float):
                    mask = iou >= self.iou_thresh
                else:
                    mask = (iou > self.iou_thresh[0]) & (iou < self.iou_thresh[1])
                    if np.any(mask):
                        continue
                    mask = iou >= self.iou_thresh[1]
                if np.any(mask):
                    bboxes = np.array([inter for inter, m in zip(intersections, mask) if m])
                    bboxes = np.concatenate([bboxes, np.ones_like(bboxes[:, :, [0]])], axis=-1)
                    bboxes = np.matmul(m, bboxes.transpose([0, 2, 1])).transpose([0, 2, 1])
                    anno['bboxes'] = bboxes
                    anno['labels'] = anno['labels'][mask]
                else:
                    if self.nonempty:
                        continue
                    else:
                        anno = None
            img = cv.warpAffine(img, m, (nw, nh))
            break
        return img, anno


class BBoxFilter:
    def __init__(self, min_area):
        self.min_area = min_area

    def __call__(self, img, anno=None):
        if anno:
            wh = np.stack([xy42xywha(bbox)[2:4] for bbox in anno['bboxes']])
            area = wh[:, 0] * wh[:, 1]
            mask = area >= self.min_area
            if np.any(mask):
                anno['bboxes'] = anno['bboxes'][mask]
                anno['labels'] = anno['labels'][mask]
            else:
                anno.clear()
        return img, anno
