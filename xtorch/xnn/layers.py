# -*- coding: utf-8 -*-
# File   : layers.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 12:07
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

from torch import nn

from .containers import ModuleAtom


__all__ = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
           'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d',
           'InstanceNorm3d', 'LayerNorm']


class Linear(ModuleAtom):
    def __init__(self, out_features, bias=True):
        super(Linear, self).__init__(out_features, bias=bias)

    def _init_module(self, x):
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.Linear(*self.args, **self.kwargs)


class ConvNd(ModuleAtom):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(ConvNd, self).__init__(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)

    def _init_params(self, x):
        if self.kwargs['groups'] < 0:
            assert x.shape[1] % self.kwargs['groups'] == 0
            self.kwargs['groups'] = x.shape[1] // -self.kwargs['groups']
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)


class Conv1d(ConvNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.Conv1d(*self.args, **self.kwargs)


class Conv2d(ConvNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.Conv2d(*self.args, **self.kwargs)


class Conv3d(ConvNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.Conv3d(*self.args, **self.kwargs)


class ConvTransposeNd(ModuleAtom):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(ConvTransposeNd, self).__init__(out_channels, kernel_size, stride=stride, padding=padding,
                                              output_padding=output_padding, dilation=dilation, groups=groups,
                                              bias=bias, padding_mode=padding_mode)

    def _init_params(self, x):
        if self.kwargs['groups'] < 0:
            assert x.shape[1] % self.kwargs['groups'] == 0
            self.kwargs['groups'] = x.shape[1] // -self.kwargs['groups']
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)


class ConvTranspose1d(ConvTransposeNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.ConvTranspose1d(*self.args, **self.kwargs)


class ConvTranspose2d(ConvTransposeNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.ConvTranspose2d(*self.args, **self.kwargs)


class ConvTranspose3d(ConvTransposeNd):
    def _init_module(self, x):
        self._init_params(x)
        self.module = nn.ConvTranspose3d(*self.args, **self.kwargs)


class BatchNormNd(ModuleAtom):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormNd, self).__init__(eps=eps, momentum=momentum, affine=affine,
                                          track_running_stats=track_running_stats)


class BatchNorm1d(BatchNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm1d(*self.args, **self.kwargs)


class BatchNorm2d(BatchNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm2d(*self.args, **self.kwargs)


class BatchNorm3d(BatchNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm3d(*self.args, **self.kwargs)


class GroupNorm(ModuleAtom):
    def __init__(self, num_groups, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, eps=eps, affine=affine)

    def _init_module(self, x):
        num_groups = self.args[0]
        if num_groups < 0:
            assert x.shape[1] % num_groups == 0
            num_groups = x.shape[1] // -num_groups
        self.args = (num_groups, x.shape[1])
        self.module = nn.GroupNorm(*self.args, **self.kwargs)


class InstanceNormNd(ModuleAtom):
    def __init__(self, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNormNd, self).__init__(eps=eps, momentum=momentum, affine=affine,
                                             track_running_stats=track_running_stats)


class InstanceNorm1d(InstanceNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm1d(*self.args, **self.kwargs)


class InstanceNorm2d(InstanceNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm2d(*self.args, **self.kwargs)


class InstanceNorm3d(InstanceNormNd):
    def _init_module(self, x):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm3d(*self.args, **self.kwargs)


class LayerNorm(ModuleAtom):
    def __init__(self, num_last_dimensions, *args, **kwargs):
        super(LayerNorm, self).__init__(num_last_dimensions, *args, **kwargs)

    def _init_module(self, x):
        self.args = (x.shape[-self.args[0]:],)
        self.module = nn.LayerNorm(*self.args, **self.kwargs)
