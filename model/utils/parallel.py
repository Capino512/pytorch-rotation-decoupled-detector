

import torch

from torch import nn

from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name, SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallelDet(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, device_ids):
        super().__init__(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        splits = inputs[0].shape[0] // len(device_ids)
        data_splits = []
        for i, device in enumerate(device_ids):
            data_split = []
            for data in inputs:
                data = data[i::len(device_ids)]
                if isinstance(data, torch.Tensor):
                    data = data.to(f'cuda:{device}', non_blocking=True)
                data_split.append(data)
            data_splits.append(data_split)
        return data_splits, [kwargs] * len(device_ids)

    # (
    #  {}, {}, ...
    # )

    # (
    #  [[], None, ...], [[], None, ...]
    # )

    def gather(self, outputs, output_device):
        if self.training:
            outputs = super().gather(outputs, output_device)
            for key, val in list(outputs.items()):
                outputs[key] = val.mean()
        else:
            outputs = sum(map(list, zip(*outputs)), [])
        return outputs
