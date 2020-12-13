# -*- coding: utf-8 -*-
# File   : prepare.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:11
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


def txt2json(dir_txt, dir_json, category):
    os.makedirs(dir_json, exist_ok=True)
    for txt in os.listdir(dir_txt):
        objs = []
        name = os.path.splitext(txt)[0]
        for line in open(os.path.join(dir_txt, txt)).readlines():
            bbox = line.strip().split('\t')[:8]
            bbox = np.array(bbox, dtype=np.float32).reshape([4, 2])
            bbox = cv.boxPoints(cv.minAreaRect(bbox))
            bbox = bbox.tolist()
            obj = dict()
            obj['name'] = category
            obj['bbox'] = bbox
            objs.append(obj)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, name + '.json'), 'wt'), indent=2)


def main():
    # (1)
    dir_txt = os.path.join(dir_dataset, 'labelTxt', 'car')
    dir_anno_car = os.path.join(dir_dataset, 'annotations', 'car')
    txt2json(dir_txt, dir_anno_car, 'car')

    dir_txt = os.path.join(dir_dataset, 'labelTxt', 'plane')
    dir_anno_plane = os.path.join(dir_dataset, 'annotations', 'plane')
    txt2json(dir_txt, dir_anno_plane, 'plane')

    # (2)
    dir_img_car = os.path.join(dir_dataset, 'images', 'car')
    dir_img_plane = os.path.join(dir_dataset, 'images', 'plane')
    num_car = len(os.listdir(dir_img_car))
    num_plane = len(os.listdir(dir_img_plane))
    num_test = 400
    indexes_test = np.linspace(1, num_car + num_plane, num_test, endpoint=False, dtype=np.int).tolist()

    size = 768
    overlap = 0
    save_empty = False

    cropper = Cropper(size, overlap)

    pair_train, pair_test = [], []
    for category, dir_img, dir_anno in [['car', dir_img_car, dir_anno_car], ['plane', dir_img_plane, dir_anno_plane]]:
        pair_train_ = []
        for filename in os.listdir(dir_img):
            index = int(filename[1: -4])
            if index + (0 if category == 'car' else num_car) in indexes_test:
                img = os.path.join('images', category, filename)
                anno = os.path.join('annotations', category, filename.replace('png', 'json'))
                pair_test.append([img, anno])
            else:
                img = os.path.join(dir_img, filename)
                anno = os.path.join(dir_anno, filename.replace('png', 'json'))
                pair_train_.append([img, anno])

        out_dir_images = os.path.join(dir_dataset, 'images', f'{category}-crop')
        out_dir_annos = os.path.join(dir_dataset, 'annotations', f'{category}-crop')
        cropper.crop_batch(pair_train_, out_dir_images, out_dir_annos, save_empty)

        for filename in os.listdir(out_dir_images):
            img = os.path.join('images', f'{category}-crop', filename)
            anno = os.path.join('annotations', f'{category}-crop', filename.replace('jpg', 'json'))
            pair_train.append([img, anno])

    # (3)
    out_dir = os.path.join(dir_dataset, 'image-sets')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(pair_train, open(os.path.join(out_dir, 'train.json'), 'wt'), indent=2)
    json.dump(pair_test, open(os.path.join(out_dir, 'test.json'), 'wt'), indent=2)


if __name__ == '__main__':

    # directory hierarchy

    # dir_dataset/images/car/P0001.png
    # -----------/car/...
    # -----------/plane/...

    # dir_dataset/labelTxt/car/P0001.txt
    # -------------/car/...
    # -------------/plane/...

    # (1) convert annotation files
    # (2) crop images
    # (3) generate image-set files

    dir_dataset = '<replace with your local path>'

    main()
