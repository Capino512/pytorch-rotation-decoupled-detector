

from torch import nn


def weight_init_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_( m.weight, 1, 0.02)
        nn.init.constant_( m.bias, 0)


def weight_init_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.uniform_(m.weight, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_( m.weight, 1)
        nn.init.constant_( m.bias, 0)


def weight_init_kaiming_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weight_init_kaiming_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weight_init_xavier_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weight_init_xavier_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


weight_init = {
    'normal': weight_init_normal,
    'uniform': weight_init_uniform,
    'kaiming_normal': weight_init_kaiming_normal,
    'kaiming_uniform': weight_init_kaiming_uniform,
    'xavier_normal': weight_init_xavier_normal,
    'xavier_uniform': weight_init_xavier_uniform,
}
