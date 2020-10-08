

import torch


class PriorBox:
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.cfg = cfg
        self.base_anchors = None
        self._buffer = {}
        self._get_anchors_grid_xywh()

    def get_anchors_grid_xywh(self, iw, ih):
        name = 'anchors_grid_xywh_%d_%d' % (iw, ih)
        anchors = self._buffer.get(name, None)
        if anchors is None:
            anchors, _ = self.__get_anchors_grid_xywh(self.cfg, iw, ih)
        return anchors

    def _get_anchors_grid_xywh(self):
        for img_size in self.cfg['img_size']:
            iw, ih = (img_size, img_size) if isinstance(img_size, int) else img_size
            anchors_grid_xywh, base_anchors = self.__get_anchors_grid_xywh(self.cfg, iw, ih)
            self._buffer['anchors_grid_xywh_%d_%d' % (iw, ih)] = anchors_grid_xywh
            if self.base_anchors is None:
                self.base_anchors = base_anchors

    @staticmethod
    def __get_anchors_grid_xywh(cfg, iw, ih):
        anchors, base_anchors = [], []
        for stride, size, scales, aspects, angles in zip(cfg['stride'], cfg['size'], cfg['scales'], cfg['aspects'], cfg['angles']):
            assert iw % stride == 0 and ih % stride == 0
            fmw, fmh = iw // stride, ih // stride
            _anchors = []
            for scale in scales:
                base_size = size / stride * scale
                for aspect in aspects:
                    _anchors.append([base_size * aspect ** 0.5, base_size / aspect ** 0.5])
            base_anchors.append(_anchors)
            _anchors = torch.tensor(_anchors, dtype=torch.float)
            offset = torch.stack(torch.meshgrid([torch.arange(fmw), torch.arange(fmh)]), dim=-1).permute(1, 0, 2).reshape(-1, 2).float()
            offset = offset[:, None].repeat(1, _anchors.shape[0], 1) + 0.5
            _anchors_grid = torch.cat([offset, _anchors[None].repeat(fmw * fmh, 1, 1)], dim=2)
            _anchors_grid[:, :, :4] *= stride
            _anchors_grid = _anchors_grid.reshape(-1, _anchors_grid.size(-1))
            anchors.append(_anchors_grid)
        anchors = torch.cat(anchors)
        return anchors, base_anchors
