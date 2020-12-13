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
import numpy as np

from bs4 import BeautifulSoup as bs
from utils.box.bbox_np import xywha2xy4


def xml2json(dir_xml, dir_json):
    os.makedirs(dir_json, exist_ok=True)
    for xml in os.listdir(dir_xml):
        objs = []
        name = os.path.splitext(xml)[0]
        for obj in bs(open(os.path.join(dir_xml, xml)), "html.parser").findAll('hrsc_object'):
            xywha = []
            xywha.append(float(obj.select_one('mbox_cx').text))
            xywha.append(float(obj.select_one('mbox_cy').text))
            xywha.append(float(obj.select_one('mbox_w').text))
            xywha.append(float(obj.select_one('mbox_h').text))
            xywha.append(np.rad2deg(float(obj.select_one('mbox_ang').text)))
            obj = dict()
            obj['name'] = 'ship'
            obj['bbox'] = xywha2xy4(xywha).tolist()
            objs.append(obj)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, name + '.json'), 'wt'), indent=2)


def main():
    # (1)
    dir_xml = os.path.join(dir_dataset, 'labelXml')
    out_dir_json = os.path.join(dir_dataset, 'annotations')
    xml2json(dir_xml, out_dir_json)

    # (2)
    out_dir = os.path.join(dir_dataset, 'image-sets')
    os.makedirs(out_dir, exist_ok=True)
    for image_set in ['trainval', 'test']:
        pairs = []
        with open(os.path.join(dir_dataset, 'ImageSets', f'{image_set}.txt')) as f:
            for line in f.readlines():
                line = line.strip()
                img = os.path.join('images', f'{line}.bmp')
                anno = os.path.join('annotations', f'{line}.json')
                if not os.path.exists(os.path.join(dir_dataset, anno)):
                    anno = None
                pairs.append([img, anno])
        json.dump(pairs, open(os.path.join(out_dir, f'{image_set}.json'), 'wt'), indent=2)


if __name__ == '__main__':

    # directory hierarchy

    # root/images/100000001.bmp
    # ----/images/...

    # root/labelXml/100000001.xml
    # ----/labelXml/...

    # (1) convert annotation files
    # (2) generate image-set files

    dir_dataset = '<replace with your local path>'

    main()
