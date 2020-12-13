# Single-Stage Rotation-Decoupled Detector for Oriented Object

This is the repository of paper **Single-Stage Rotation-Decoupled Detector for Oriented Object**. [[Paper]](https://www.mdpi.com/2072-4292/12/19/3262/htm) [[PDF]](https://www.mdpi.com/2072-4292/12/19/3262/pdf)

**Update:**  Updated the code for training on the DOTA, HRSC2016 and UCAS-AOD datasets. The test results on the HRSC2016 and UCAS-AOD datasets are very close to the results reported in our paper. Testing on the DOTA dataset is in progress.

<img src="demo/graphical-abstract.png" alt="Graphical Abstract" style="zoom: 50%;" />



## Introduction

We optimized the anchor-based oriented object detection method by decoupling the matching of the oriented bounding box and the oriented anchor into the matching of the horizontal bounding box and the horizontal anchor.

## Performance

### DOTA1.0 (Task1)

| backbone  | MS   | mAP   | PL    | BD    | BR    | GTF   | SV    | LV    | SH    | TC    | BC    | ST    | SBF   | RA    | HA    | SP    | HC    |
| --------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ResNet101 | ×    | 75.52 | 89.7  | 84.33 | 46.35 | 68.62 | 73.89 | 73.19 | 86.92 | 90.41 | 86.46 | 84.3  | 64.22 | 64.95 | 73.55 | 72.59 | 73.31 |
| ResNet101 | √    | 77.75 | 89.15 | 83.92 | 52.51 | 73.06 | 77.81 | 79    | 87.08 | 90.62 | 86.72 | 87.15 | 63.96 | 70.29 | 76.98 | 75.79 | 72.15 |

### HRSC2016

| backbone  | AP    |
| --------- | ----- |
| ResNet101 | 94.29 |
| ResNet152 | 94.61 |

### UCAS-AOD

| backbone  | plane | car   | mAP   |
| --------- | ----- | ----- | ----- |
| ResNet101 | 98.86 | 94.96 | 96.86 |
| ResNet152 | 98.85 | 95.18 | 97.01 |

## Visualization

![Result](demo/result.png)

## Run

### Requirements

```
tqdm
numpy
pillow
cython
beautifulsoup4
opnecv-python
pytorch>=1.2
torchvision>=0.4
tensorboard>=2.2
```

### Compile

```
# 'rbbox_batched_nms' will be used as post-processing in the interface stage
# use gpu, for Linux only
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_gpu
python setup.py build_ext --inplace

# alternative, use cpu, for Windows and Linux
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_cpu
python setup.py build_ext --inplace
```

###  Pre-training Weight

Download pretrained weight  files from [baiduyun](https://pan.baidu.com/s/1u9i3giU5Q-7XAF_rkyL8Bw ) (fetch code: 4m2c). Modify the `DIR_WEIGHT` defined in `config/__init__.py` to be the directory where the weight files are placed.

```
DIR_WEIGHT = /.../pre-training-weights
```

### Train on DOTA

#### Data Preprocessing

Download the [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset, and move files like:

```
$PATH_ROOT/images
----------/labelTxt-v1.0-obb
$PATH_ROOT/images/train/P0000.png
-----------------/train/...
-----------------/val/...
-----------------/test/...

$PATH_ROOT/labelTxt/train/P0000.txt
-------------------/train/...
-------------------/val/...
```

Modify `dir_dataset` and `dir_dataset` defined in `run/dota/prepare.py`,  `run/dota/train.py`, `run/dota/evaluate.py` to the local path.

```
dir_dataset = '/.../PATH_ROOT'  # The directory where the dataset is located
dir_save = '...'                # Output directory
```

Then run the provided code:

```
REPO_ROOT$ python run/dota/prepare.py
```

#### Start Training

```
REPO_ROOT$ python run/dota/train.py
```

#### Evaluate

```
REPO_ROOT$ python run/dota/evaluate.py
```

#### Checkpoint

upcoming...

### Train on HRSC2016

Similar to the steps on the DOTA dataset, the code is provided in `run/hrsc2016`.

#### Checkpoint

[baiduyun](https://pan.baidu.com/s/11gv3KZKMB4ZBkOaSygD3GA) (fetch code: 9vjf)

Run `run/hrsc2016/evaluate.py` with setting `checkpoint` to the path where the file is placed, you should get the following output:

```
AP
ship: 94.17826999090045
mAP: 94.17826999090045
```

### Train on UCAS-AOD

Similar to the steps on the DOTA dataset, the code is provided in `run/ucas-aod`.

#### Checkpoint

[baiduyun](https://pan.baidu.com/s/1sS5lc65F99lz7SmPMAw1uw) (fetch code: 1l2q)

Run `run/ucas-aod/evaluate.py` with setting `checkpoint` to the path where the file is placed, you should get the following output:

```
AP:
car: 95.17266104678446
plane: 98.56249539995756
mAP: 96.867578223371
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