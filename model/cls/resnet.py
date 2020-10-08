

from .classifier import Classifier
from ..backbones import resnet


def resnet18(num_classes, dropout=0):
    return Classifier(resnet.resnet18(False), num_classes, dropout)


def resnet34(num_classes, dropout=0):
    return Classifier(resnet.resnet34(False), num_classes, dropout)


def resnet50(num_classes, dropout=0):
    return Classifier(resnet.resnet50(False), num_classes, dropout)


def resnet101(num_classes, dropout=0):
    return Classifier(resnet.resnet101(False), num_classes, dropout)


def resnet152(num_classes, dropout=0):
    return Classifier(resnet.resnet152(False), num_classes, dropout)


def resnest50(num_classes, dropout=0):
    return Classifier(resnet.resnest50(False), num_classes, dropout)


def resnest101(num_classes, dropout=0):
    return Classifier(resnet.resnest101(False), num_classes, dropout)


def resnest200(num_classes, dropout=0):
    return Classifier(resnet.resnest200(False), num_classes, dropout)


def resnest269(num_classes, dropout=0):
    return Classifier(resnet.resnest269(False), num_classes, dropout)


def resnext50_32x4d(num_classes, dropout=0):
    return Classifier(resnet.resnext50_32x4d(False), num_classes, dropout)


def resnext101_32x8d(num_classes, dropout=0):
    return Classifier(resnet.resnext101_32x8d(False), num_classes, dropout)


def resnet18_d(num_classes, dropout=0):
    return Classifier(resnet.resnet18_d(False), num_classes, dropout)


def resnet34_d(num_classes, dropout=0):
    return Classifier(resnet.resnet34_d(False), num_classes, dropout)


def resnet50_d(num_classes, dropout=0):
    return Classifier(resnet.resnet50_d(False), num_classes, dropout)


def resnet101_d(num_classes, dropout=0):
    return Classifier(resnet.resnet101_d(False), num_classes, dropout)


def resnet152_d(num_classes, dropout=0):
    return Classifier(resnet.resnet152_d(False), num_classes, dropout)
