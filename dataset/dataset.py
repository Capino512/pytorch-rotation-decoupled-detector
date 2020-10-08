

import os
import glob
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from utils.image import imread
from utils.box.bbox_np import xy42xywha


class ImageFolder:
    def __init__(self, folder, color_space='RGB'):
        self.color_space = color_space
        self.names = sorted(os.listdir(folder))
        self.dataset = self.get_dataset(folder)

    def get_dataset(self, folder):
        dataset = []
        for i, name in enumerate(self.names):
            paths = sorted(glob.glob(os.path.join(folder, name, '*')))
            dataset.extend([[p, i] for p in paths])
        return dataset

    def __getitem__(self, index):
        path, label = self.dataset[index]
        img = imread(path, self.color_space)
        return img, label, {'img_path': path, 'shape': img.shape}

    def __len__(self):
        return len(self.dataset)


class ClsDataset(Dataset):
    def __init__(self, folder, aug=None, color_space='RGB'):
        self.dataset = ImageFolder(folder, color_space)
        self.names = self.dataset.names
        self.aug = aug

    def __getitem__(self, index):
        img, label, info = self.dataset[index]
        if self.aug is not None:
            img, _ = self.aug(img)
        return img, label, info

    @staticmethod
    def collate(batch):
        images, labels, infos = [], [], []
        for i, (img, label, info) in enumerate(batch):
            images.append(torch.from_numpy(img).reshape(*img.shape[:2], -1).float())
            labels.append(label)
            infos.append(info)
        images = torch.stack(images).permute(0, 3, 1, 2)
        labels = torch.tensor(labels).long()
        return images, labels, infos

    def __len__(self):
        return len(self.dataset)


class BaseDataset:
    def __init__(self, flist, names, color_space='RGB'):
        self.names = names
        self.color_space = color_space
        self.label2name = dict((label, name) for label, name in enumerate(self.names))
        self.name2label = dict((name, label) for label, name in enumerate(self.names))
        self.dataset = sum([json.load(open(file)) for file in flist], [])

    def get_objs(self, path):
        objs = []
        if path is not None:
            for obj in json.load(open(path)):
                if obj['name'] in self.names:
                    obj['label'] = self.name2label[obj['name']]
                    objs.append(obj)
        return objs

    def __getitem__(self, index):
        img_path, anno_path = self.dataset[index]
        img = imread(img_path, self.color_space)
        objs = self.get_objs(anno_path)
        info = {'img_path': img_path, 'anno_path': anno_path, 'shape': img.shape}
        return img, objs, info

    def __len__(self):
        return len(self.dataset)


class DetDataset(Dataset):
    def __init__(self, flist, names, aug=None, color_space='RGB'):
        self.names = names
        self.dataset = BaseDataset(flist, names, color_space)
        self.aug = aug

    @staticmethod
    def merge_objs(objs):
        target = dict()
        if objs:
            target['bboxes'] = np.array([obj['bbox'] for obj in objs], dtype=np.float32)
            target['labels'] = np.array([obj['label'] for obj in objs], dtype=np.int64)
        return target

    @staticmethod
    def convert_objs(objs):
        target = dict()
        if objs:
            target['bboxes'] = torch.tensor([xy42xywha(bbox, angle45=True) for bbox in objs['bboxes']], dtype=torch.float)
            target['labels'] = torch.from_numpy(objs['labels']).long()
        return target

    def __getitem__(self, index):
        img, objs, info = self.dataset[index]
        objs = self.merge_objs(objs)
        if self.aug is not None:
            img, objs = self.aug(img, objs)
        return img, objs, info

    @staticmethod
    def collate(batch):
        images, targets, infos = [], [], []
        batch = sorted(batch, key=lambda x: len(x[1]['labels']) if x[1] else 0)
        for i, (img, objs, info) in enumerate(batch):
            images.append(torch.from_numpy(img).reshape(*img.shape[:2], -1).float())
            targets.append(DetDataset.convert_objs(objs))
            infos.append(info)
        return torch.stack(images).permute(0, 3, 1, 2), targets, infos

    def __len__(self):
        return len(self.dataset)
