

from torch import nn

from .containers import ModuleAtom


__all__ = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
           'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d',
           'InstanceNorm3d', 'LayerNorm']


class Linear(ModuleAtom):
    def __init__(self, out_features, bias=True):
        super(Linear, self).__init__(out_features, bias=bias)

    def _init_module(self, x, **kwargs):
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.Linear(*self.args, **self.kwargs)


class ConvNd(ModuleAtom):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ConvNd, self).__init__(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)

    def get_groups(self, channels):
        if self.kwargs['groups'] < 0:
            assert channels % self.kwargs['groups'] == 0
            self.kwargs['groups'] = channels // -self.kwargs['groups']


class Conv1d(ConvNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.Conv1d(*self.args, **self.kwargs)


class Conv2d(ConvNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.Conv2d(*self.args, **self.kwargs)


class Conv3d(ConvNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.Conv3d(*self.args, **self.kwargs)


class ConvTransposeNd(ModuleAtom):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ConvTransposeNd, self).__init__(out_channels, kernel_size, stride=stride, padding=padding,
                                              output_padding=output_padding, dilation=dilation, groups=groups, bias=bias,
                                              padding_mode=padding_mode)

    def get_groups(self, channels):
        if self.kwargs['groups'] < 0:
            assert channels % self.kwargs['groups'] == 0
            self.kwargs['groups'] = channels // -self.kwargs['groups']


class ConvTranspose1d(ConvTransposeNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.ConvTranspose1d(*self.args, **self.kwargs)


class ConvTranspose2d(ConvTransposeNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.ConvTranspose2d(*self.args, **self.kwargs)


class ConvTranspose3d(ConvTransposeNd):
    def _init_module(self, x, **kwargs):
        self.get_groups(x.shape[1])
        if self.args[0] is None:
            self.args = (x.shape[1], *self.args[1:])
        self.args = (x.shape[1], *self.args)
        self.module = nn.ConvTranspose3d(*self.args, **self.kwargs)


class BatchNormNd(ModuleAtom):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormNd, self).__init__(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class BatchNorm1d(BatchNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm1d(*self.args, **self.kwargs)


class BatchNorm2d(BatchNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm2d(*self.args, **self.kwargs)


class BatchNorm3d(BatchNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.BatchNorm3d(*self.args, **self.kwargs)


class GroupNorm(ModuleAtom):
    def __init__(self, num_groups=1, channels_per_group=0, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups=num_groups, channels_per_group=channels_per_group, eps=eps, affine=affine)

    def _init_module(self, x, **kwargs):
        if self.kwargs['num_groups'] < 0:
            assert x.shape[1] % self.kwargs['num_groups'] == 0
            self.kwargs['num_groups'] = x.shape[1] // -self.kwargs['num_groups']
        self.args = (self.kwargs.pop('num_groups'), x.shape[1])
        self.module = nn.GroupNorm(*self.args, **self.kwargs)


class InstanceNormNd(ModuleAtom):
    def __init__(self, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNormNd, self).__init__(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class InstanceNorm1d(InstanceNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm1d(*self.args, **self.kwargs)


class InstanceNorm2d(InstanceNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm2d(*self.args, **self.kwargs)


class InstanceNorm3d(InstanceNormNd):
    def _init_module(self, x, **kwargs):
        self.args = (x.shape[1], *self.args)
        self.module = nn.InstanceNorm3d(*self.args, **self.kwargs)


class LayerNorm(ModuleAtom):
    def __init__(self, ndim, *args, **kwargs):
        super(LayerNorm, self).__init__(ndim, *args, **kwargs)

    def _init_module(self, x, **kwargs):
        self.args = (x.shape[-self.args[0]:],)
        self.module = nn.LayerNorm(*self.args, **self.kwargs)
