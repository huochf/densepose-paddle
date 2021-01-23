# -------------------------------------------------------------------
# Modified from densepose(https://github.com/facebookresearch/DensePose)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import numpy as np
import cv2
import paddle.fluid.dygraph as dg

import densepose.utils.boxes as box_utils
from densepose.modeling.proposal_generator import map_rois_to_fpn_levels


def get_boxes(rois, cls_score, bbox_deltas, im_shape):
    pred_boxes = box_utils.bbox_transform(rois, bbox_deltas, weights=(10, 10, 5, 5))
    pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_shape)

    scores, boxes, cls_boxes = box_results_with_nms_and_limit(cls_score, pred_boxes)

    return scores, boxes, cls_boxes


def box_results_with_nms_and_limit(scores, boxes):
    num_classes = 2 # for body and non-body
    score_thresh = 0.05
    cls_boxes = [[] for _ in range(num_classes)]

    for j in range(1, num_classes): # skip j = 0 for background
        inds = np.where(scores[:, j] > score_thresh)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4: (j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        keep = box_utils.nms(dets_j, 0.5)
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def get_body_rois(im_rois, im_scale):
    """
    Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordiantes
        im_scale_factors (list): scale factors as returned by _get_image_blob

    
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
                        [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    rois_blob = rois_blob.astype(np.float32, copy=False)
    lvl_min = 2
    lvl_max = 5
    target_lvls = map_rois_to_fpn_levels(rois_blob[:,1:5])

    body_rois = []
    rois_idx_order = []
    for lvl in range(lvl_min, lvl_max + 1):
        idx_level = np.where(target_lvls == lvl)[0]
        rois_level = rois_blob[idx_level, :][:, 1:]
        batch_idx = np.zeros((len(rois_level), 1)) # assume only one test image in batch
        rois_level = np.hstack((batch_idx, rois_level))
        # rois_level = dg.to_variable(rois_level.astype('float32'))
        body_rois.append(rois_level)
        rois_idx_order.append(idx_level)
    
    return body_rois, rois_idx_order

def _project_im_rois(im_rois, scales):
    """
    Project image RoIs into the image pyramid.
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def get_body_part(boxes, ann_index, index_uv, u, v):
    # In case of 1
    if len(ann_index.shape) == 3:
        ann_index = np.expand_dims(ann_index, axis=0)
    if len(index_uv.shape) == 3:
        index_uv = np.expand_dims(index_uv, axis=0)
    if len(u.shape) == 3:
        u = np.expand_dims(u, axis=0)
    if len(v.shape) == 3:
        v = np.expand_dims(v, axis=0)

    K = 24 + 1
    outputs = []
    for ind, entry in enumerate(boxes):
        # Compute ref box width and height
        bx = max(entry[2] - entry[0], 1)
        by = max(entry[3] - entry[1], 1)

        # preds[ind] axes are CHW; bring p axes to WHC
        cur_ann_index = np.swapaxes(ann_index[ind], 0, 2)
        cur_index_uv = np.swapaxes(index_uv[ind], 0, 2)
        cur_u = np.swapaxes(u[ind], 0, 2)
        cur_v = np.swapaxes(v[ind], 0, 2)

        # Resize p from (HEATMAP_SIZE, HEATMAP_SIZE, c) to (int(bx), int(by), c)
        cur_ann_index = cv2.resize(cur_ann_index, (by, bx))
        cur_index_uv = cv2.resize(cur_index_uv, (by, bx))
        cur_u = cv2.resize(cur_u, (by, bx))
        cur_v = cv2.resize(cur_v, (by, bx))

        # Bring cur_preds axes back to CHW
        cur_ann_index = np.swapaxes(cur_ann_index, 0, 2)
        cur_index_uv = np.swapaxes(cur_index_uv, 0, 2)
        cur_u = np.swapaxes(cur_u, 0, 2)
        cur_v = np.swapaxes(cur_v, 0, 2)

        # Removed squeeze calls due to singleton dimension issues
        cur_ann_index = np.argmax(cur_ann_index, axis=0)
        cur_index_uv = np.argmax(cur_index_uv, axis=0)
        cur_index_uv = cur_index_uv * (cur_ann_index > 0).astype(np.float32)

        output = np.zeros([3, int(by), int(bx)], dtype=np.float32)
        output[0] = cur_index_uv

        for part_id in range(1, K):
            part_u = cur_u[part_id]
            part_v = cur_v[part_id]
            output[1, cur_index_uv == part_id] = part_u[cur_index_uv == part_id]
            output[2, cur_index_uv == part_id] = part_v[cur_index_uv == part_id]
        
        outputs.append(output)

    return outputs
