

import cv2 as cv
import numpy as np

from utils.misc import containerize


__all__ = ['hflip', 'vflip', 'rgb2gray', 'resize', 'rotate90', 'pad']


INTER_MODE = {'NEAREST': cv.INTER_NEAREST, 'BILINEAR': cv.INTER_LINEAR, 'BICUBIC': cv.INTER_CUBIC}


def hflip(img):
    return np.ascontiguousarray(np.fliplr(img))


def vflip(img):
    return np.ascontiguousarray(np.flipud(img))


def rgb2gray(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def resize(img, size, interpolate='BILINEAR'):
    w, h = containerize(size, 2)
    ih, iw = img.shape[:2]
    if ih != h or iw != w:
        img = cv.resize(img, (w, h), interpolation=INTER_MODE[interpolate])
    return img


def rotate90(img, k):  # CLOCKWISE k=0, 1, 2, 3
    if k % 4 != 0:
        img = np.ascontiguousarray(np.rot90(img, -k))
    return img


def pad(img, padding, mode='constant', **kwargs):
    if isinstance(padding, int):
        padding = [[padding, padding], [padding, padding]]
    else:
        padding = [containerize(p, 2) for p in padding]
    if img.ndim == 3 and len(padding) == 2:
        padding.append([0, 0])
    return np.pad(img, padding, mode, **kwargs)
