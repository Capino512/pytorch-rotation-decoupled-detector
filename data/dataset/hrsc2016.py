

from config import CATEGORY_HRSC2016 as NAMES

from .dataset import DetDataset


class HRSC2016(DetDataset):
    def __init__(self, root, image_sets, aug=None):
        super(HRSC2016, self).__init__(root, image_sets, NAMES, aug)
