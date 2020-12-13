

import torch

from torch import nn


class CustomDetDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, device_ids):
        super().__init__(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        data_splits = []
        for i, device in enumerate(device_ids):
            data_split = []
            for data in inputs:
                data = data[i:: len(device_ids)]
                if isinstance(data, torch.Tensor):
                    data = data.to(f'cuda:{device}', non_blocking=True)
                data_split.append(data)
            data_splits.append(data_split)
        return data_splits, [kwargs] * len(device_ids)

    def gather(self, outputs, output_device):
        if self.training:
            # (
            #  {}, {}, ...
            # )
            outputs = super().gather(outputs, output_device)
            for key, val in list(outputs.items()):
                outputs[key] = val.mean()
        else:
            # (
            #  [[], [], ...], [[], [], ...]
            # )
            outputs = sum(map(list, zip(*outputs)), [])
        return outputs
