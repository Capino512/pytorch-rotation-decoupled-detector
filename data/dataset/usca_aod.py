

from config import CATEGORY_UCAS_AOD as NAMES

from .dataset import DetDataset


class UCAS_AOD(DetDataset):
    def __init__(self, root, image_sets, aug=None):
        super(UCAS_AOD, self).__init__(root, image_sets, NAMES, aug)
