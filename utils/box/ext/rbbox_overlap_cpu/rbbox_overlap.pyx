

# distutils: language = c++


import numpy as np
cimport numpy as np


assert sizeof(int) == sizeof(np.int32_t)


cdef extern from 'rbbox_overlap.h':
    cdef float RotateIoU(np.float64_t * region1, np.float64_t * region2)
    cdef void RotateIoU_1x1(np.float64_t * region1, np.float64_t * region2, int n, np.float64_t * ret)
    cdef void RotateIoU_nxn(np.float64_t * region1, np.float64_t * region2, int n1, int n2, np.float64_t * ret)
    cdef void RotateNMS(np.float64_t * bboxes, int n, float thresh, np.int32_t * keeps)


def rbbox_iou(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    return RotateIoU(&a[0], &b[0])


def rbbox_iou_1x1(np.ndarray[np.float64_t, ndim=2] a, np.ndarray[np.float64_t, ndim=2] b):
    cdef int n1 = a.shape[0]
    cdef int n2 = b.shape[0]
    assert n1 == n2
    cdef np.ndarray[np.float64_t, ndim=1] ret = np.zeros([n1], dtype=np.float64)
    RotateIoU_1x1(&a[0, 0], &b[0, 0], n1, &ret[0])
    return ret


def rbbox_iou_nxn(np.ndarray[np.float64_t, ndim=2] a, np.ndarray[np.float64_t, ndim=2] b):
    cdef int n1 = a.shape[0]
    cdef int n2 = b.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] ret = np.zeros([n1, n2], dtype=np.float64)
    RotateIoU_nxn(&a[0, 0], &b[0, 0], n1, n2, &ret[0, 0])
    return ret


def rbbox_nms(np.ndarray[np.float64_t, ndim=2] boxes, np.ndarray[np.float64_t, ndim=1] scores, float thresh):
    cdef int n = boxes.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] keeps = np.ones([n], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] indexes = np.argsort(scores).astype(np.int32)
    boxes = boxes[indexes]
    RotateNMS(&boxes[0, 0], n, thresh, &keeps[0])
    keeps = indexes[keeps.astype(np.bool)]
    if len(keeps) > 1:
        keeps = np.ascontiguousarray(keeps[::-1])
    return keeps


# python setup.py build_ext --inplace

# iou.cpp(2961): error C2664: 'void RotateNMS(float *,int,float,int *)': cannot convert argument 4 from '__pyx_t_5numpy_int32_t *' to 'int *'
#
# go to line(2961) in the generated file in iou.cpp
# Modify corresponding __pyx_t_5numpy_int32_t to int


