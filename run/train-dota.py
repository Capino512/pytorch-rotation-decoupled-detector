

import os
import time
import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.aug.compose import Compose
from dataset.aug import ops
from dataset.dataset import DetDataset

from model.det.rdd import RDD
from model.backbones import resnet
from model.utils.adjust_lr import adjust_lr_multi_step
from model.utils.parallel import replace_w_sync_bn, CustomDataParallelDet

from utils.misc import format_time


def main():
    dir_weight = os.path.join(dir_save, 'weight')
    dir_log = os.path.join(dir_save, 'log')
    os.makedirs(dir_weight, exist_ok=True)
    writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    lr = 1e-3
    batch_size = 12
    num_workers = 4

    max_step = 250000
    lr_cfg = [[100000, lr], [200000, lr / 10], [max_step, lr / 50]]
    warm_up = [1000, lr / 50, lr]
    save_interval = 10000
    eta_interval = 10

    image_size = 768
    aug = Compose([ops.PhotometricDistort(),
                   ops.RandomHFlip(),
                   ops.RandomVFlip(),
                   ops.RandomRotate90(),
                   ops.ResizeJitter([0.8, 1.2]),
                   ops.PadSquare(),
                   ops.Resize(image_size),
                   ops.BboxFilter(24 * 24 * 0.4)
                   ])
    dataset = DetDataset([flist_train, flist_val], names, aug)
    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    anchor = {
        'img_size': [image_size],
        'stride': [8, 16, 32, 64, 128],
        'size': [24, 48, 96, 192, 384],
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
        'aspects': [[1, 2, 1 / 2, 4, 1 / 4, 8, 1 / 8]] * 5,
    }

    CFG = {'anchor': anchor,
           'num_classes': num_classes,
           'extra': 2,
           }

    model = RDD(backbone(fetch_feature=True), CFG)
    model.build(shape=[2, 3, image_size, image_size])
    if current_step:
        model.restore(os.path.join(dir_weight, '%d.pth' % current_step))
    else:
        model.init()
    if len(devices) > 1:
        model.apply(replace_w_sync_bn)
        model = CustomDataParallelDet(model, devices)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    training = True
    running_step = 0

    while training and current_step < max_step:
        for step, (images, targets, infos) in enumerate(loader_train):
            running_step += 1
            if running_step == 50:
                t = time.time()
            current_step += 1
            adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            images = images.cuda() / 255
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            text = []
            for name, val in losses.items():
                writer.add_scalar(name, val, global_step=current_step)
                text.append('%s: %.6f' % (name, val.item()))
            writer.flush()
            print('<%d of %d> %s' % (step + 1, len(loader_train), ' ,'.join(text)))

            if current_step % eta_interval == 0 and running_step > 50:
                eta = (time.time() - t) / (running_step - 50) * (max_step - current_step)
                print('<%d of %d> eta: %s' % (current_step, max_step, format_time(eta)))

            if current_step % save_interval == 0:
                print('save weight file...')
                save_path = os.path.join(dir_weight, '%d.pth' % current_step)
                state_dict = model.state_dict() if len(devices) == 1 else model.module.state_dict()
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

    names = ['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
             'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
             'storage-tank', 'swimming-pool', 'tennis-court']

    devices = [0, 1]
    backbone = resnet.resnest101
    dir_save = './output'

    flist_train = '<replace with your local path>'
    flist_val = '<replace with your local path>'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(device) for device in devices])
    devices = list(range(len(devices)))

    main()
