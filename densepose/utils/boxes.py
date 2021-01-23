# -------------------------------------------------------------------
# Modified from densepose(https://github.com/facebookresearch/DensePose)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import numpy as np
import densepose.utils.cython_nms as cython_nms


def boxes_area(boxes):
    w = (boxes[:, 2] - boxes[:, 0] + 1)
    h = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = w * h 
    assert np.all(areas >= 0), "Negative areas founds"
    return areas


def clip_tiled_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes). 
    """
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4'.format(
            boxes.shape[1]
        )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "-1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "-1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def nms(dets, thresh):
    """
    Apply classic DPM-style greedy NMS.
    """
    if dets.shape[0] == 0:
        return []
    return cython_nms.nms(dets, thresh)

