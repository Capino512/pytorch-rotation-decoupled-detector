# -*- coding: utf-8 -*-
# File   : prepare.py
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
import json
import cv2 as cv
import numpy as np

from utils.crop_image import Cropper


def txt2json(dir_txt, dir_json):
    os.makedirs(dir_json, exist_ok=True)
    for file in os.listdir(dir_txt):
        objs = []
        for i, line in enumerate(open(os.path.join(dir_txt, file)).readlines()):
            line = line.strip()
            line_split = line.split(' ')
            if len(line_split) == 10:
                obj = dict()
                coord = np.array(line_split[:8], dtype=np.float32).reshape([4, 2])
                bbox = cv.boxPoints(cv.minAreaRect(coord)).astype(np.int).tolist()
                obj['name'] = line_split[8].lower()
                obj['bbox'] = bbox
                objs.append(obj)
            else:
                print('<skip line> %s' % line)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, file.replace('txt', 'json')), 'wt'), indent=2)


def main(image_set):
    # (1)
    if image_set != 'test':
        dir_txt = os.path.join(dir_dataset, 'labelTxt', image_set)
        out_dir_json = os.path.join(dir_dataset, 'annotations', image_set)
        txt2json(dir_txt, out_dir_json)

    # (2)
    pairs = []
    for filename in os.listdir(os.path.join(dir_dataset, 'images', image_set)):
        anno = os.path.join(dir_dataset, 'annotations', image_set, filename.replace('png', 'json'))
        img = os.path.join(dir_dataset, 'images', image_set, filename)
        if not os.path.exists(anno):
            anno = None
        pairs.append([img, anno])

    out_dir_images = os.path.join(dir_dataset, 'images', f'{image_set}-crop')
    out_dir_annos = os.path.join(dir_dataset, 'annotations', f'{image_set}-crop')

    sizes = [512, 768, 1024, 1536]
    overlap = 0.25
    save_empty = image_set == 'test'

    cropper = Cropper(sizes, overlap)
    cropper.crop_batch(pairs, out_dir_images, out_dir_annos, save_empty)

    # (3)
    pairs = []
    for filename in os.listdir(out_dir_images):
        img = os.path.join('images', f'{image_set}-crop', filename)
        anno = None if image_set == 'test' else os.path.join('annotations', f'{image_set}-crop', filename.replace('jpg', 'json'))
        pairs.append([img, anno])
    out_dir = os.path.join(dir_dataset, 'image-sets')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(pairs, open(os.path.join(out_dir, f'{image_set}.json'), 'wt'), indent=2)


if __name__ == '__main__':

    # directory hierarchy

    # root/images/train/P0000.png
    # -----------/train/...
    # -----------/val/...
    # -----------/test/...

    # root/labelTxt/train/P0000.txt
    # -------------/train/...
    # -------------/val/...

    # (1) convert annotation files
    # (2) crop images
    # (3) generate image-set files

    dir_dataset = '<replace with your local path>'

    main('train')
    main('val')
    main('test')
