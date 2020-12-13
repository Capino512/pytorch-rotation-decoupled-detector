

import cv2 as cv
import numpy as np

from ..func import *
from ..compose import Compose


__all__ = ['ToFloat', 'Normalize', 'ConvertColor', 'RandomGray', 'RandomBrightness', 'RandomContrast',
           'RandomLightingNoise', 'RandomHue', 'RandomSaturation', 'PhotometricDistort']


class ToFloat:
    def __call__(self, img, anno=None):
        img = img.astype(np.float32)
        return img, anno


class Normalize:
    def __init__(self, mean, std):
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
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
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
