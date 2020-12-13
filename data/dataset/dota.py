

from config import CATEGORY_DOTA_V10 as NAMES

from .dataset import DetDataset


class DOTA(DetDataset):
    def __init__(self, root, image_sets, aug=None):
        super(DOTA, self).__init__(root, image_sets, NAMES, aug)
