

import torch

from torch import nn
from xtorch import xnn

from .utils.modules import FPN, DetPredict
from .utils.priorbox import PriorBox
from .utils.loss import calc_loss
from .utils.detect import detect
from ..utils.init import weight_init


class RDD(xnn.Module):
    def __init__(self, backbone, cfg):
        super(RDD, self).__init__()

        cfg.setdefault('iou_thresh', [0.4, 0.5])
        cfg.setdefault('variance', [0.1, 0.2, 0.1])

        cfg.setdefault('conf_thresh', 0.1)
        cfg.setdefault('nms_thresh', 0.5)
        cfg.setdefault('top_n', None)

        cfg.setdefault('extra', 0)
        cfg.setdefault('fpn_plane', 256)
        cfg.setdefault('extra_plane', 512)

        self.backbone = backbone
        self.prior_box = PriorBox(cfg['anchor'])
        self.num_levels = len(cfg['anchor']['stride'])
        self.num_classes = cfg['num_classes']
        self.iou_thresh = cfg['iou_thresh']
        self.variance = cfg['variance']

        self.conf_thresh = cfg['conf_thresh']
        self.nms_thresh = cfg['nms_thresh']
        self.top_n = cfg['top_n']

        self.extra = cfg['extra']
        self.fpn_plane = cfg['fpn_plane']
        self.extra_plane = cfg['extra_plane']

        self.fpn = FPN(self.num_levels, self.fpn_plane)
        num_anchors = [len(anchors) for anchors in self.prior_box.base_anchors]
        self.predict = DetPredict(self.num_levels, self.fpn_plane, num_anchors, self.num_classes, 5)

        if self.extra > 0:
            self.extra_layers = xnn.ModuleList()
            for i in range(self.extra):
                self.extra_layers.append(xnn.Sequential(xnn.Conv2d(self.extra_plane, 3, 2, 1, bias=False),
                                                        xnn.BatchNorm2d(),
                                                        nn.ReLU(inplace=True)))

    def init(self):
        self.backbone.init()
        self.fpn.apply(weight_init['normal'])
        self.predict.apply(weight_init['normal'])
        if self.extra > 0:
            self.extra_layers.apply(weight_init['normal'])

    def restore(self, path):
        weight = torch.load(path)
        self.load_state_dict(weight, strict=True)

    def forward(self, inputs, targets=None):
        anchors = self.prior_box.get_anchors_grid_xywh(inputs.size(3), inputs.size(2)).to(inputs)

        features = list(self.backbone(inputs))
        features = features[-(self.num_levels - self.extra):]
        if self.extra > 0:
            for layer in self.extra_layers:
                features.append(layer(features[-1]))

        features = self.fpn(features)
        pred_cls, pred_loc = self.predict(features)

        if self.training:
            if targets is not None:
                return calc_loss(targets, pred_cls, pred_loc, anchors, self.num_classes, self.iou_thresh, self.variance)
        else:
            top_n = (inputs.size(2) // 32) * (inputs.size(3) // 32) if self.top_n is None else self.top_n
            return detect(pred_cls.detach_(), pred_loc.detach_(), anchors, self.variance, self.conf_thresh, self.nms_thresh, top_n)
