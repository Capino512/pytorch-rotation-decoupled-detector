

from .classifier import Classifier
from ..backbones import darknet


def darknet21(num_classes, dropout=0):
    return Classifier(darknet.darknet21(False), num_classes, dropout)


def darknet53(num_classes, dropout=0):
    return Classifier(darknet.darknet53(False), num_classes, dropout)
