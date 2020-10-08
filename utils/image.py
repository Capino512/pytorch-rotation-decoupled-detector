

import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def imread(path, mode='RGB'):
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)
    return np.array(img)


def imwrite(image, path):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)

