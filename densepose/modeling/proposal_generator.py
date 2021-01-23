# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# and densepose(https://github.com/facebookresearch/DensePose)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from .model_lib import conv1x1, conv3x3
from densepose.config import configurable
from densepose.utils.registry import Registry
import densepose.utils.boxes as box_utils

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
"""


PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
PROPOSAL_GENERATOR_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.
"""


def build_proposal_generator(cfg, ):
    return PROPOSAL_GENERATOR_REGISTRY.get(cfg.MODEL.RPN.NAME)(cfg)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(dg.Layer):

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
        super().__init__()

        self.conv = conv3x3(in_channels, in_channels)
        self.objectness_logits = conv1x1(in_channels, num_anchors)
        self.anchor_deltas = conv1x1(in_channels, num_anchors * box_dim)


    @classmethod
    def from_config(cls, cfg,):
        # Srandard RPN is shared across levels:
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_anchors = cfg.MODEL.RPN.NUM_ANCHORS
        box_dim = cfg.MODEL.RPN.BOX_DIM

        return {"in_channels": in_channels, "num_anchors": num_anchors, "box_dim": box_dim}


    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = L.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(dg.Layer):
    
    @configurable
    def __init__(self, 
                 head, 
                 generate_proposal_op,
                 rpn_max_level,
                 rpn_min_level,
                 anchor_size,
                 anchor_aspect_ratios):
        super().__init__()
        self.rpn_head = head
        self.generate_proposal_op = generate_proposal_op
        self.collect_and_distribute_op = CollectAndDistributeFpnRpnProposalsOp()
        self.k_max = rpn_max_level
        self.k_min = rpn_min_level
        self.anchor_size = anchor_size
        self.anchor_aspect_ratios = anchor_aspect_ratios


    @classmethod
    def from_config(cls, cfg, ):
        rpn_head_type = cfg.MODEL.RPN.HEAD
        ret = {
            "head": RPN_HEAD_REGISTRY.get(rpn_head_type)(cfg, ),
            "generate_proposal_op": GenerateProposalsOp(cfg),
            "rpn_max_level": cfg.MODEL.RPN.MAX_LEVEL,
            "rpn_min_level": cfg.MODEL.RPN.MIN_LEVEL,
            "anchor_size": cfg.MODEL.RPN.ANCHOR_START_SIZE,
            "anchor_aspect_ratios": cfg.MODEL.RPN.ASPECT_RATIOS,
        }
        return ret


    def forward(self, features, im_info):
        # features: p6 -> p2
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        rpn_rois = []
        rpn_roi_probs = [] # p2 -> p6
        for lvl in range(self.k_min, self.k_max + 1): # 2 -> 6
            lvl_anchors = generate_anchors(
                stride=2. ** lvl,
                sizes=[self.anchor_size[0] * 2. ** (lvl - self.k_min)],
                aspect_ratios=self.anchor_aspect_ratios
            )
            lvl_cls_logits = pred_objectness_logits[self.k_max - lvl]
            lvl_cls_logits = L.sigmoid(lvl_cls_logits).numpy()
            lvl_bbox_deltas = pred_anchor_deltas[self.k_max - lvl].numpy()

            lvl_rois, lvl_roi_probs, lvl_anchors = self.generate_proposal_op(lvl_cls_logits, 
                lvl_bbox_deltas, lvl_anchors, 1 / 2 ** lvl, im_info)
            rpn_rois.append(lvl_rois)
            rpn_roi_probs.append(lvl_roi_probs)
        
        rois = self.collect_and_distribute_op(rpn_rois, rpn_roi_probs)
        
        # list of ndarray(size: (n, 5)), p2 -> p5
        return rois


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 1)
):
    """
    Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )

    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window)
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


class GenerateProposalsOp(object):
    """
    Output object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors")
    """

    def __init__(self, cfg):
        self.pre_nms_topN = cfg.MODEL.RPN.PRE_NMS_TOP_N
        self.post_nms_topN = cfg.MODEL.RPN.POST_NMS_TOP_N
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.min_size = cfg.MODEL.RPN.MIN_SIZE


    def __call__(self, rpn_cls_probs, rpn_bbox_pred, anchors, spatial_scale, im_info):
        # 1. for each location i in a (H, W) grid:
        #        generate A anchor boxes centered on cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals

        # predicted probability of fg object for each RPN anchor
        scores = rpn_cls_probs
        # predicted anchors transformation
        bbox_deltas = rpn_bbox_pred
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        
        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        feat_stride = 1. / spatial_scale
        # Enumerate all shifted positions on the (H, W) grid
        shift_x = np.arange(0, width) * feat_stride
        shift_y = np.arange(0, height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        
        # Broadcast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #     - add A anchors of shape (1, A, 4) to
        #     - K shifts of shape (K, 1, 4) to get
        #     - all shifted anchors of shape (K, A, 4)
        #     - reshape to (K*A, 4) shifted anchors
        num_images = scores.shape[0]
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))

        rois = np.empty((0, 5), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :],
                scores[im_i, :, :, :]
            )
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32
            )
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)
        
        return rois, roi_probs, all_anchors


    def proposals_for_one_image(
        self, im_info, all_anchors, bbox_deltas, scores
    ):
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #     - bbox deltas will be (4 * A, H, W) format from conv output
        #     - transpose to (H, W, 4 * A)
        #     - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #       in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))

        # Same story for the scores:
        #     - scores are (A, H, W) format from conv output
        #     - transpose to (H, W, A)
        #     - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #       to match the order of anchors and bbox_deltas
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if self.pre_nms_topN <= 0 or self.pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(
                -scores.squeeze(), self.pre_nms_topN
            )[:self.pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = box_utils.bbox_transform(
            all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0)
        )

        # 2. clip proposals to image (may result in proposals with zero area)
        # that will be removed in the next step)
        proposals = box_utils.clip_tiled_boxes(proposals, im_info[:2])

        # 3.remove predicted boxes with either height or width < min_size
        keep = _filter_boxes(proposals, self.min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals ( -> RoIs top)
        if self.nms_thresh > 0:
            keep = box_utils.nms(np.hstack((proposals, scores)), self.nms_thresh)
            if self.post_nms_topN > 0:
                keep = keep[:self.post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
        return proposals, scores


def _filter_boxes(boxes, min_size, im_info):
    """
    Only keep boxes with both sides >= min_size and center within the image.
    """

    # Compute the width and height of the proposal boxes as measured in the original
    # image coordinate system (this is required to avoid "Negative Areas Found"
    # assertions in other parts of the code that measure).

    im_scale = im_info[2]
    ws_orig_scale = (boxes[:, 2] - boxes[:, 0]) / im_scale + 1
    hs_orig_scale = (boxes[:, 3] - boxes[:, 1]) / im_scale + 1
    # To avoid numerical issues we require the min_size to be at least 1 pixel in the 
    # original image
    min_size = np.maximum(min_size, 1)
    # Proposal center is computed relative to the scaled input image
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = np.where((ws_orig_scale >= min_size)
                    & (hs_orig_scale >= min_size)
                    & (x_ctr < im_info[1])
                    & (y_ctr < im_info[0]))[0]
    return keep


class CollectAndDistributeFpnRpnProposalsOp(object):
    def __call__(self, rpn_rois, rpn_roi_probs):
        # rpn_rois: list (p2 -> p6), (N, 5) (batch_id, x1, y1, x2, y2)
        # rpn_roi_probs: list (p2 -> p6), (N, 1)
        rois = collect(rpn_rois, rpn_roi_probs)
        rois = distribute(rois)

        # rois now is a list with length 4, for p2, p3, p4, p5
        # each item is np.ndarray with shape (n, 5)
        return rois


def collect(rpn_rois, rpn_roi_probs):
    post_nms_topN = 1000
    rois = np.concatenate(rpn_rois) # [N, 5], lose level information
    scores = np.concatenate(rpn_roi_probs).squeeze() # (N, )

    inds = np.argsort(-scores)[:post_nms_topN] # (N, )
    rois = rois[inds, :] # (1000, 5)
    return rois


def distribute(rois, ):
    lvls = map_rois_to_fpn_levels(rois[:, 1:5])

    # rois_idx_order = np.empty((0,))
    results = []
    for output_idx, lvl in enumerate(range(2, 6)):
        idx_lvl = np.where(lvls == lvl)[0]
        roi_level = rois[idx_lvl, :]
        results.append(roi_level)
    return results


def map_rois_to_fpn_levels(rois, k_min=2, k_max=5):
    # rois: (N, 4)
    s = np.sqrt(box_utils.boxes_area(rois))

    s0 = 224 # fpn_roi_canonical_scale
    lvl0 = 4 # roi_canonical_level

    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-5))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls

