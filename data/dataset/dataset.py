# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:44
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import os
import json
import torch
import numpy as np

from copy import deepcopy
from torch.utils.data import Dataset
from utils.misc import convert_path
from utils.image import imread
from utils.box.bbox_np import xy42xywha


class DetDataset(Dataset):
    def __init__(self, root, image_set, names, aug=None, color_space='RGB'):
        self.names = names
        self.aug = aug
        self.color_space = color_space
        self.label2name = dict((label, name) for label, name in enumerate(self.names))
        self.name2label = dict((name, label) for label, name in enumerate(self.names))
        self.dataset = self.load_dataset(root, image_set)

    @staticmethod
    def load_dataset(root, image_set):
        image_sets = [image_set] if isinstance(image_set, str) else image_set
        dataset = []
        for image_set in image_sets:
            for img, anno in json.load(open(os.path.join(root, 'image-sets', f'{image_set}.json'))):
                img = os.path.join(root, convert_path(img))
                anno = (os.path.join(root, convert_path(anno)) if anno else None)
                dataset.append([img, anno])
        return dataset

    @staticmethod
    def load_objs(path, name2label=None):
        objs = None
        if path:
            objs = json.load(open(path))
            bboxes = [obj['bbox'] for obj in objs]
            labels = [name2label[obj['name']] if name2label else obj['name'] for obj in objs]
            objs = {'bboxes': np.array(bboxes, dtype=np.float32), 'labels': np.array(labels)}
        return objs

    @staticmethod
    def convert_objs(objs):
        target = dict()
        if objs:
            # Limit the angle between -45° and 45° by set flag=2
            target['bboxes'] = torch.from_numpy(np.stack([xy42xywha(bbox, flag=2) for bbox in objs['bboxes']])).float()
            target['labels'] = torch.from_numpy(objs['labels']).long()
        return target

    def __getitem__(self, index):
        img_path, anno_path = self.dataset[index]
        img = imread(img_path, self.color_space)
        objs = self.load_objs(anno_path, self.name2label)
        info = {'img_path': img_path, 'anno_path': anno_path, 'shape': img.shape, 'objs': objs}
        if self.aug is not None:
            img, objs = self.aug(img, deepcopy(objs))
        return img, objs, info

    @staticmethod
    def collate(batch):
        images, targets, infos = [], [], []
        # Ensure data balance when parallelizing
        batch = sorted(batch, key=lambda x: len(x[1]['labels']) if x[1] else 0)
        for i, (img, objs, info) in enumerate(batch):
            images.append(torch.from_numpy(img).reshape(*img.shape[:2], -1).float())
            targets.append(DetDataset.convert_objs(objs))
            infos.append(info)
        return torch.stack(images).permute(0, 3, 1, 2), targets, infos

    def __len__(self):
        return len(self.dataset)
