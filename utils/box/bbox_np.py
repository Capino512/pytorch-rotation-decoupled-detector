

import numpy as np


def bbox_switch(bbox, in_type, out_type):  # 'xyxy', 'xywh'
    if in_type == 'xyxy' and out_type == 'xywh':
        bbox = np.concatenate([(bbox[..., 0:2] + bbox[..., 2:4]) / 2, bbox[..., 2:4] - bbox[..., 0:2]], axis=-1)
    elif in_type == 'xywh' and out_type == 'xyxy':
        bbox = np.concatenate([bbox[..., 0:2] - bbox[..., 2:4] / 2, bbox[..., 0:2] + bbox[..., 2:4] / 2], axis=-1)
    return bbox


def xywha2xy4(x, y, w, h, a):  # a represents the angle(degree), clockwise, a=0 along the X axis
    a = (a % 180 + 180) % 180
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def xy42xywha(bbox, angle45=False):  # bbox(4x2) represents a rectangle
    x, y = np.mean(bbox, axis=0)
    diff01 = bbox[0] - bbox[1]
    diff13 = bbox[1] - bbox[2]
    w = np.sqrt(np.square(diff01).sum())
    h = np.sqrt(np.square(diff13).sum())
    if w >= h:
        a = np.rad2deg(np.arctan2(diff01[1], diff01[0]))
    else:
        a = np.rad2deg(np.arctan2(diff13[1], diff13[0])) + 90
    a = (a + 180) % 180
    if angle45:
        if 45 <= a < 135:
            w, h = h, w
            a -= 90
        elif a >= 135:
            a -= 180
    return x, y, w, h, a
