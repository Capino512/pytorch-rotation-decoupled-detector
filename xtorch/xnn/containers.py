

import torch

from torch import nn


__all__ = ['Module', 'ModuleAtom', 'ModulePipe', 'Sequential', 'ModuleList', 'ModuleDict']


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def build(self, *inputs, shape=None, **kwargs):
        if shape is not None:
            inputs = (torch.randn(shape),)
        return self(*inputs, **kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class ModuleAtom(Module):
    def __init__(self, *args, **kwargs):
        super(ModuleAtom, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.module = None

    def _init_module(self, *inputs, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        if self.module is None:
            self._init_module(*inputs, **kwargs)
        return self.module(*inputs, **kwargs)


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


class ModuleList(nn.ModuleList, Module):
    def __init__(self, modules=None):
        super(ModuleList, self).__init__(modules)


class ModuleDict(nn.ModuleDict, Module):
    def __init__(self, modules=None):
        super(ModuleDict, self).__init__(modules)
