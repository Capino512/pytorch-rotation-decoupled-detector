

import torch

from torch import nn
from xtorch import xnn

from ..utils.init import weight_init


class Classifier(xnn.ModulePipe):
    def __init__(self, backbone, num_classes, dropout=0):
        super(Classifier, self).__init__()
        self.backbone = backbone
        fc = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), xnn.Linear(num_classes)]
        if dropout > 0:
            fc.insert(2, nn.Dropout(dropout))
        self.fc = xnn.Sequential(*fc)

    def init(self):
        self.backbone.init()
        self.fc.apply(weight_init['normal'])

    def restore(self, path):
        weight = torch.load(path)
        self.load_state_dict(weight, strict=True)
