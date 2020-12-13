# -*- coding: utf-8 -*-
# File   : train.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:11
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import sys

sys.path.append('.')

import os
import tqdm
import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.aug.compose import Compose
from data.aug import ops
from data.dataset import UCAS_AOD

from model.rdd import RDD
from model.backbone import resnet

from utils.adjust_lr import adjust_lr_multi_step
from utils.parallel import convert_model, CustomDetDataParallel


def main():
    dir_weight = os.path.join(dir_save, 'weight')
    dir_log = os.path.join(dir_save, 'log')
    os.makedirs(dir_weight, exist_ok=True)
    writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    image_size = 768
    lr = 1e-3
    batch_size = 12
    num_workers = 4

    max_step = 30000
    lr_cfg = [[15000, lr], [24000, lr / 10], [max_step, lr / 50]]
    warm_up = [500, lr / 50, lr]
    save_interval = 1000

    aug = Compose([
        ops.ToFloat(),
        ops.PhotometricDistort(),
        ops.RandomHFlip(),
        ops.RandomVFlip(),
        ops.RandomRotate90(),
        ops.ResizeJitter([0.8, 1.2]),
        ops.PadSquare(),
        ops.Resize(image_size),
    ])
    dataset = UCAS_AOD(dir_dataset, ['train'], aug)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
                        collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64],
        'sizes': [3] * 4,
        'aspects': [[1, 2]] * 4,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 4,
    }

    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 1,
    }

    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    if current_step:
        model.restore(os.path.join(dir_weight, '%d.pth' % current_step))
    else:
        model.init()
    if len(device_ids) > 1:
        model = convert_model(model)
        model = CustomDetDataParallel(model, device_ids)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    training = True

    while training and current_step < max_step:
        tqdm_loader = tqdm.tqdm(loader)
        for images, targets, infos in tqdm_loader:
            current_step += 1
            adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            images = images.cuda() / 255
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key, val in list(losses.items()):
                losses[key] = val.item()
                writer.add_scalar(key, val, global_step=current_step)
            writer.flush()
            tqdm_loader.set_postfix(losses)
            tqdm_loader.set_description(f'<{current_step}/{max_step}>')

            if current_step % save_interval == 0:
                save_path = os.path.join(dir_weight, '%d.pth' % current_step)
                state_dict = model.state_dict() if len(device_ids) == 1 else model.module.state_dict()
                torch.save(state_dict, save_path)
                cache_file = os.path.join(dir_weight, '%d.pth' % (current_step - save_interval))
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            if current_step >= max_step:
                training = False
                writer.close()
                break


if __name__ == '__main__':

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    device_ids = [0, 1]
    torch.cuda.set_device(device_ids[0])
    backbone = resnet.resnet101

    dir_dataset = '<replace with your local path>'
    dir_save = '<replace with your local path>'

    main()
