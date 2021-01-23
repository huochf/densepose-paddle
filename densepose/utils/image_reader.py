# -------------------------------------------------------------------
# Modified from densepose(https://github.com/facebookresearch/DensePose)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import cv2
import numpy as np

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
SCALES = 800
MAX_SIZE = 4000

def read_image(img_path):
    im_origin = cv2.imread(img_path)
    im_ori = im_origin.astype(np.float32, copy=True)
    im_ori_shape = im_ori.shape
    im = im_ori - PIXEL_MEANS
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(SCALES) / float(im_size_min)

    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = float(MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    h, w, c = im.shape
    im = im.transpose((2, 0, 1))
    im = im.reshape((1, c, h, w))

    new_h, new_w = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
    blob = np.zeros((1, 3, new_h, new_w), dtype=np.float32)
    blob[0, :, :h, :w] = im

    im_info = np.array([new_h, new_w, im_scale])[np.newaxis, :].astype(np.float32)
    return blob, im_info, im_origin, im_ori_shape
    


