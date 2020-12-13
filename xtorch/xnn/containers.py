# -*- coding: utf-8 -*-
# File   : containers.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 12:07
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import torch

from torch import nn


__all__ = ['Module', 'ModuleAtom', 'ModulePipe', 'Sequential']


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def build_pipe(self, shape):
        return self(torch.randn(shape))

    build = __call__


class ModuleAtom(Module):
    def __init__(self, *args, **kwargs):
        super(ModuleAtom, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.module = None

    def _init_module(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.module is None:
            self._init_module(*args, **kwargs)
        return self.module(*args, **kwargs)


class ModulePipe(Module):
    def __init__(self):
        super(ModulePipe, self).__init__()

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class Sequential(nn.Sequential, Module):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
