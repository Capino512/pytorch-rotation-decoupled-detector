# Single-Stage Rotation-Decoupled Detector for Oriented Object

This is the repository of paper **Single-Stage Rotation-Decoupled Detector for Oriented Object**. [[Paper]](https://www.mdpi.com/2072-4292/12/19/3262) [[PDF]](https://www.mdpi.com/2072-4292/12/19/3262/pdf)

**Tips:**  The currently published code is untested, for preview only, and may be updated in the future.

<img src="demo/Graphical Abstract.png" alt="Graphical Abstract" style="zoom: 50%;" />



## Introduction

We optimized the anchor-based oriented object detection method by decoupling the matching of the oriented bounding box and the oriented anchor into the matching of the horizontal bounding box and the horizontal anchor.

## Performance

#### DOTA1.0 (Task1)

| backbone  | MS   | mAP   | PL    | BD    | BR    | GTF   | SV    | LV    | SH    | TC    | BC    | ST    | SBF   | RA    | HA    | SP    | HC    |
| --------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ResNet101 | ×    | 75.52 | 89.7  | 84.33 | 46.35 | 68.62 | 73.89 | 73.19 | 86.92 | 90.41 | 86.46 | 84.3  | 64.22 | 64.95 | 73.55 | 72.59 | 73.31 |
| ResNet101 | √    | 77.75 | 89.15 | 83.92 | 52.51 | 73.06 | 77.81 | 79    | 87.08 | 90.62 | 86.72 | 87.15 | 63.96 | 70.29 | 76.98 | 75.79 | 72.15 |

## Visualization

![Result](demo/Result.png)

## Run

### Requirements

```
numpy
pillow
cython
shapely
opnecv-python
pytorch>=1.2
torchvision>=0.4
tensorboard>=2.2
```

### Compile

```
# 'rbbox_batched_nms' will be used in the interface stage
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_gpu  # for Linux
python setup.py build_ext --inplace

# alternative
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_cpu  # for Windows and Linux
python setup.py build_ext --inplace
```

### Weights Download

Download pre-trained weight  file from [baiduyun](https://pan.baidu.com/s/1u9i3giU5Q-7XAF_rkyL8Bw ) (fetch code: 4m2c). Modify the `DIR_WEIGHT` defined in `config/__init__.py` to be the path where the weight file is placed.

### Train on DOTA

Download the [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset, and move files like:

```
$PATH_ROOT/images
----------/labelTxt-v1.0-obb
```

Processing data, including the following steps:

```
(1) p0001.txt -> p0001.json 
    p0001.json: [{'name': 'xxx', 'bbox':[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...]
(2) Crop images (with annotations)
(3) Generate file list
    -> train.json [['p0000-xxx.jpg', 'p0000-xxx.json'], ...]
    -> val.json [['p0003-xxx.jpg', 'p0003-xxx.json'], ...]
    -> test.json [['p0006-xxx.jpg', None], ...]
```

Complete these steps by running the the provided code:

```
REPO_ROOT$ python run/prepare-dota.py  # Need to modify the local path
```

Start training:

```
REPO_ROOT$ python run/train-dota.py  # Need to modify the local path
```

## Citation

```
@article{rdd,
    title={Single-Stage Rotation-Decoupled Detector for Oriented Object},
    author={Zhong, Bo and Ao, Kai},
    journal={Remote Sensing},
    year={2020}
}
```