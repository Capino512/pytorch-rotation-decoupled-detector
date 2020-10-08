

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon
from utils.box.bbox_np import xy42xywha, xywha2xy4

from .func import *
from .compose import Compose


class ToFloat:
    def __call__(self, img, anno=None):
        img = img.astype(np.float32)
        return img, anno


class Normalize:
    def __init__(self, mean=0, std=255):
        self.mean = mean
        self.std = std

    def __call__(self, img, anno=None):
        img = (img - self.mean) / self.std
        return img, anno


class ConvertColor:
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img, anno=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)
        elif self.current == 'HSV' and self.transform == 'RGB':
            img = cv.cvtColor(img, cv.COLOR_HSV2RGB_FULL)
        else:
            raise NotImplementedError
        return img, anno


class RandomGray:  # RGB
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            img = rgb2gray(img)
        return img, anno


class RandomBrightness:  # RGB
    def __init__(self, delta=32):
        assert 0 <= delta <= 255
        self.delta = delta

    def __call__(self, img, anno=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            img = np.clip(img + delta, 0, 255)
        return img, anno


class RandomContrast:  # RGB
    def __init__(self, lower=0.5, upper=1.5):
        assert 0 < lower < upper
        self.lower = lower
        self.upper = upper

    def __call__(self, img, anno=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img = np.clip(alpha * img, 0, 255)
        return img, anno


class RandomLightingNoise:  # RGB
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            indexes = [0, 1, 2]
            np.random.shuffle(indexes)
            img = img[..., indexes]
        return img, anno


class RandomHue:  # HSV
    def __init__(self, delta=18.0):
        assert 0 <= delta <= 360
        self.delta = delta

    def __call__(self, img, anno=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            img[:, :, 0] = (img[:, :, 0] + delta) % 360
        return img, anno


class RandomSaturation:  # HSV
    def __init__(self, lower=0.5, upper=1.5):
        assert 0 < lower < upper
        self.lower = lower
        self.upper = upper

    def __call__(self, img, anno=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img[:, :, 1] = np.clip(alpha * img[:, :, 1], 0, 1)
        return img, anno


class PhotometricDistort:
    def __init__(self, prob_light_noise=0.2, prob_gray=0.2):
        self.prob_light_noise = prob_light_noise
        self.prob_gray = prob_gray
        self.pd = [
            RandomContrast(),
            ConvertColor(current='RGB', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.rand_gray = RandomGray()

    def __call__(self, img, anno=None):
        img, anno = self.rand_brightness(img, anno)
        distort = Compose(self.pd[:-1] if np.random.randint(2) else self.pd[1:])
        img, anno = distort(img, anno)
        if np.random.randint(2):
            if np.random.rand() < self.prob_light_noise:
                img, anno = self.rand_light_noise(img, anno)
        else:
            if np.random.rand() < self.prob_gray:
                img, anno = self.rand_gray(img, anno)
        return img, anno


###############################################################################


class RandomHFlip:
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            if anno:
                h, w = img.shape[:2]
                anno['bboxes'][:, :, 0] = w - 1 - anno['bboxes'][:, :, 0]
            img = hflip(img)
        return img, anno


class RandomVFlip:
    def __call__(self, img, anno=None):
        if np.random.randint(2):
            if anno:
                h, w = img.shape[:2]
                anno['bboxes'][:, :, 1] = h - 1 - anno['bboxes'][:, :, 1]
            img = vflip(img)
        return img, anno


class Resize:
    def __init__(self, size, interpolate='BILINEAR'):
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolate = interpolate

    def __call__(self, img, anno=None):
        if anno:
            h, w = img.shape[:2]
            rw, rh = self.size
            anno['bboxes'] *= [rw / w, rh / h]
        img = resize(img, self.size, self.interpolate)
        return img, anno


class ResizeJitter:
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img, anno=None):
        h, w = img.shape[:2]
        nh, nw = int(round(h * np.random.uniform(*self.scale))), int(round(w * np.random.uniform(*self.scale)))
        img, anno = Resize((nw, nh))(img, anno)
        return img, anno


class Pad:
    def __init__(self, size, check_size=True):
        self.check_size = check_size
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img, anno=None):
        w, h = self.size
        ih, iw = img.shape[:2]
        if self.check_size:
            assert ih <= h and iw <= w
        padding = (max(0, h - ih) // 2), max(0, (w - iw) // 2)
        padding = [[padding[0], max(0, h - ih - padding[0])], [padding[1], max(0, w - iw - padding[1])]]
        if anno:
            anno['bboxes'] += [padding[1][0], padding[0][0]]
        img = pad(img, padding)
        return img, anno


class PadSquare:
    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        w = h = max(ih, iw)
        if iw != w or ih != h:
            img, anno = Pad((w, h))(img, anno)
        return img, anno


class Rotate:
    def __init__(self, angle, scale=1, expand=False, shift=False):
        self.angle = angle
        self.scale = scale
        self.expand = expand
        self.shift = shift

    def __call__(self, img, anno=None):
        nh, nw = h, w = img.shape[:2]
        point = ((w - 1) / 2, (h - 1) / 2)
        m = cv.getRotationMatrix2D(point, angle=-self.angle, scale=self.scale)
        if self.expand or self.shift:
            corner = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float32)
            corner = np.matmul(m, corner.T).T
            x1, y1 = np.min(corner, axis=0)
            x2, y2 = np.max(corner, axis=0)
            dx = (x2 - x1 - w) / 2
            dy = (y2 - y1 - h) / 2
            if self.expand:
                nw = int(np.ceil(x2 - x1))
                nh = int(np.ceil(y2 - y1))
                shiftX = dx
                shiftY = dy
            else:
                shiftX = np.random.uniform(-dx, dx) if dx > 0 else 0
                shiftY = np.random.uniform(-dy, dy) if dy > 0 else 0
            m[0, 2] += shiftX
            m[1, 2] += shiftY
        bounds = Polygon.from_bounds(0, 0, nw, nh)
        if anno:
            bboxes, labels = [], []
            for bbox, label in zip(anno['bboxes'], anno['labels']):
                corner = np.matmul(m, np.c_[bbox, np.ones((4, 1))].T).T
                if not self.expand:
                    p = Polygon(corner)
                    inter = p.intersection(bounds)
                    if inter.area / p.area > 0.5:
                        corner = np.array(inter.minimum_rotated_rectangle.exterior.coords[:4])
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
        self.scale = (scale, scale) if isinstance(self.scale, (int, float)) else scale
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
        h, w = img.shape[:2]
        if anno:
            if k == 1:
                anno['bboxes'][:, :, 1] = h - 1 - anno['bboxes'][:, :, 1]
                anno['bboxes'] = anno['bboxes'][:, :, [1, 0]]
            if k == 2: anno['bboxes'] = [w - 1, h - 1] - anno['bboxes']
            if k == 3:
                anno['bboxes'][:, :, 0] = w - 1 - anno['bboxes'][:, :, 0]
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

    @staticmethod
    def get_length(start, stop):
        return np.random.randint(start, stop + 1)

    def __call__(self, img, anno=None):
        ih, iw = img.shape[:2]
        areas = None
        polygons = None
        for count in range(self.max_try):
            if isinstance(self.size, int):
                h = w = min(min(ih, iw), self.size)
            else:
                if self.max_aspect == 1:
                    h = w = self.get_length(min(self.size[0], min(ih, iw)), min(self.size[1], min(ih, iw)))
                else:
                    h = self.get_length(min(self.size[0], ih), min(ih, self.size[1]))
                    w = self.get_length(min(self.size[0], iw), min(iw, self.size[1]))
                    if max(h / w, w / h) > self.max_aspect:
                        continue
            oh = np.random.randint(0, ih - h + 1)
            ow = np.random.randint(0, iw - w + 1)
            a = np.random.uniform(0, 180)
            roi = xywha2xy4(ow + w / 2, oh + h / 2, w, h, a)
            m = cv.getAffineTransform(roi.astype(np.float32)[:3], np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32))
            bound = Polygon(roi)
            if anno:
                if polygons is None:
                    polygons = [Polygon(bbox) for bbox in anno['bboxes']]
                intersections = [bound.intersection(polygon) for polygon in polygons]
                if areas is None:
                    areas = [polygon.area for polygon in polygons]
                iou = np.array([inter.area / area for inter, area in zip(intersections, areas)])
                if isinstance(self.iou_thresh, float):
                    masks = iou >= self.iou_thresh
                else:
                    masks = (iou > self.iou_thresh[0]) & (iou < self.iou_thresh[1])
                    if np.any(masks):
                        continue
                    masks = iou >= self.iou_thresh[1]
                if np.any(masks):
                    bboxes = np.array([inter.minimum_rotated_rectangle.exterior.coords[:4] for inter, mask in zip(intersections, masks) if mask])
                    bboxes = np.concatenate([bboxes, np.ones_like(bboxes[:, :, [0]])], axis=-1)
                    bboxes = np.matmul(m, bboxes.transpose([0, 2, 1])).transpose([0, 2, 1])
                    anno['bboxes'] = bboxes
                    anno['labels'] = anno['labels'][masks]
                else:
                    if self.nonempty:
                        continue
                    else:
                        anno.clear()
            img = cv.warpAffine(img, m, (w, h))
            break
        return img, anno


class BboxFilter:
    def __init__(self, min_area):
        self.min_area = min_area

    def __call__(self, img, anno=None):
        if anno:
            wh = np.array([xy42xywha(bbox)[2:4] for bbox in anno['bboxes']])
            area = wh[:, 0] * wh[:, 1]
            mask = area >= self.min_area
            if np.any(mask):
                anno['bboxes'] = anno['bboxes'][mask]
                anno['labels'] = anno['labels'][mask]
            else:
                anno.clear()
        return img, anno
