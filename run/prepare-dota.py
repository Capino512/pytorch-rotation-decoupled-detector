

import os
import json
import cv2 as cv
import numpy as np
import multiprocessing

from copy import deepcopy
from shapely.geometry import Polygon
from utils.image import imread, imwrite


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
            json.dump(objs, open(os.path.join(dir_json, file.replace('txt', 'json')), 'wt'))


class Cropper:
    def __init__(self, out_dir_images, out_dir_annos, sizes, overlap):
        os.makedirs(out_dir_images, exist_ok=True)
        os.makedirs(out_dir_annos, exist_ok=True)
        self.out_dir_images = out_dir_images
        self.out_dir_annos = out_dir_annos
        self.sizes = sizes
        self.overlap = overlap

    def crop_bbox(self, objs, roi):
        sub_objs = []
        x, y = roi[:2]
        roi = Polygon.from_bounds(*roi)
        for obj in objs:
            p = Polygon(obj['bbox'])
            inter = p.intersection(roi)
            if inter.area / p.area > 0.5:
                sub_bbox = np.array(inter.minimum_rotated_rectangle.exterior.coords)[:4] - [x, y]
                obj = deepcopy(obj)
                obj['bbox'] = sub_bbox.tolist()
                sub_objs.append(obj)
        return sub_objs

    def crop_with_anno(self, path_img, path_anno, is_test_set):
        print('crop:', path_img, path_anno)
        img = imread(path_img)
        ih, iw = img.shape[:2]
        fname = os.path.splitext(os.path.basename(path_img))[0]
        if not is_test_set:
            anno = json.load(open(path_anno))
        for size in self.sizes:
            if min(iw, ih) < size * 0.75:
                continue
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
                    new_fname = '%s-%d-%d-%d' % (fname, size, x, y)
                    flag_save = is_test_set
                    if not is_test_set:
                        sub_anno = self.crop_bbox(anno, (x, y, x + w, y + h))
                        if sub_anno:
                            flag_save = True
                            json.dump(sub_anno, open(os.path.join(self.out_dir_annos, new_fname + '.json'), 'wt'))
                    if flag_save:
                        save_path = os.path.join(self.out_dir_images, new_fname + '.jpg')
                        sub_img = img[y:y + h, x:x + w]
                        imwrite(sub_img, save_path)

    def crop(self, pairs, is_test_set):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for image, anno in pairs:
            pool.apply_async(self.crop_with_anno, (image, anno, is_test_set))
        pool.close()
        pool.join()


def flist(dir_images, dir_annos, image_sets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train, val, test = [], [], []
    for fname in os.listdir(dir_images):
        index = fname.split('-')[0]
        if index in image_sets['train']:
            train.append([os.path.join(dir_images, fname), os.path.join(dir_annos, fname.replace('jpg', 'json'))])
        elif index in image_sets['val']:
            val.append([os.path.join(dir_images, fname), os.path.join(dir_annos, fname.replace('jpg', 'json'))])
        else:
            test.append([os.path.join(dir_images, fname), None])
    json.dump(train, open(os.path.join(out_dir, 'train.json'), 'wt'))
    json.dump(val, open(os.path.join(out_dir, 'val.json'), 'wt'))
    json.dump(test, open(os.path.join(out_dir, 'test.json'), 'wt'))


def main():
    root = r'<replace with your local path>'
    image_sets = '../demo/image-sets.json'

    txt2json(os.path.join(root, 'labelTxt-v1.0-obb'), os.path.join(root, 'labelJson-v1.0-obb'))

    image_sets = json.load(open(image_sets))
    out_dir_images = os.path.join(root, 'images-crop')
    out_dir_annos = os.path.join(root, 'labelJson-v1.0-obb-crop')
    sizes = [512, 768, 1024, 1536]
    overlap = 0.25

    croper = Cropper(out_dir_images, out_dir_annos, sizes, overlap)
    pairs = []
    for fname in image_sets['train']:
        anno = os.path.join(root, 'labelJson-v1.0-obb', fname + '.json')
        if os.path.exists(anno):
            pairs.append([os.path.join(root, 'images', fname + '.png'), anno])
    croper.crop(pairs, False)

    pairs = []
    for fname in image_sets['val']:
        anno = os.path.join(root, 'labelJson-v1.0-obb', fname + '.json')
        if os.path.exists(anno):
            pairs.append([os.path.join(root, 'images', fname + '.png'), anno])
    croper.crop(pairs, False)

    pairs = []
    for fname in image_sets['test']:
        pairs.append([os.path.join(root, 'images', fname + '.png'), None])
    croper.crop(pairs, True)

    flist(out_dir_images, out_dir_annos, image_sets, os.path.join(root, 'flist'))


if __name__ == '__main__':

    # root/images
    # ----/labelTxt-v1.0-obb

    # (1) p0001.txt -> p0001.json [{'name': 'xxx', 'bbox':[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...]
    # (2) crop images
    # (3) -> train.json [['p0000-xxx.jpg', 'p0000-xxx.json'], ...]
    #     -> val.json [['p0003-xxx.jpg', 'p0003-xxx.json'], ...]
    #     -> test.json [['p0006-xxx.jpg', None], ...]

    main()
