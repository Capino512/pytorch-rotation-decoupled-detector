# -*- coding: utf-8 -*-
# File   : train.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2021/03/20 16:00
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.


def main(batch_size, rank, world_size):

    import os
    import tqdm
    import torch
    import tempfile

    from torch import optim
    from torch import distributed as dist
    from torch.nn import SyncBatchNorm
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from data.aug.compose import Compose
    from data.aug import ops
    from data.dataset import DOTA

    from model.rdd import RDD
    from model.backbone import resnet

    from utils.adjust_lr import adjust_lr_multi_step

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    backbone = resnet.resnet101

    dir_dataset = '<replace with your local path>'
    dir_save = '<replace with your local path>'

    dir_weight = os.path.join(dir_save, 'weight')
    dir_log = os.path.join(dir_save, 'log')
    os.makedirs(dir_weight, exist_ok=True)
    if rank == 0:
        writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    image_size = 768
    lr = 1e-3
    batch_size //= world_size
    num_workers = 4

    max_step = 250000
    lr_cfg = [[100000, lr], [200000, lr / 10], [max_step, lr / 50]]
    warm_up = [1000, lr / 50, lr]
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
        ops.BBoxFilter(24 * 24 * 0.4)
    ])
    dataset = DOTA(dir_dataset, ['train', 'val'], aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank)
    batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }

    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
    }
    device = torch.device(f'cuda:{rank}')
    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if current_step:
        model.module.load_state_dict(torch.load(os.path.join(dir_weight, '%d.pth' % current_step), map_location=device))
    else:
        checkpoint = os.path.join(tempfile.gettempdir(), "initial-weights.pth")
        if rank == 0:
            model.module.init()
            torch.save(model.module.state_dict(), checkpoint)
        dist.barrier()
        if rank > 0:
            model.module.load_state_dict(torch.load(checkpoint, map_location=device))
        dist.barrier()
        if rank == 0:
            os.remove(checkpoint)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    training = True
    while training and current_step < max_step:
        tqdm_loader = tqdm.tqdm(loader) if rank == 0 else loader
        for images, targets, infos in tqdm_loader:
            current_step += 1
            adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            images = images.cuda() / 255
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                for key, val in list(losses.items()):
                    losses[key] = val.item()
                    writer.add_scalar(key, val, global_step=current_step)
                writer.flush()
                tqdm_loader.set_postfix(losses)
                tqdm_loader.set_description(f'<{current_step}/{max_step}>')

                if current_step % save_interval == 0:
                    save_path = os.path.join(dir_weight, '%d.pth' % current_step)
                    state_dict = model.module.state_dict()
                    torch.save(state_dict, save_path)
                    cache_file = os.path.join(dir_weight, '%d.pth' % (current_step - save_interval))
                    if os.path.exists(cache_file):
                        os.remove(cache_file)

            if current_step >= max_step:
                training = False
                if rank == 0:
                    writer.close()
                break


if __name__ == "__main__":

    import os
    import sys
    import argparse
    import multiprocessing

    sys.path.append('.')
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--device_ids', default='0,1', type=str)
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    device_ids = list(range(len(args.device_ids.split(','))))

    processes = []
    for device_id in device_ids:
        p = multiprocessing.Process(target=main, args=(args.batch_size, device_id, len(device_ids)))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # python -m torch.distributed.launch run/dota/train-dist.py --batch_size=12 --device_ids=0,1
