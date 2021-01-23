# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import numpy as np
import paddle.fluid.dygraph as dg

from densepose.config import configurable
from .build import META_ARCH_REGISTRY
from .backbone import build_backbone
from .proposal_generator import build_proposal_generator
from .roi_heads import build_fast_rcnn_heads, build_body_uv_rcnn_heads

from densepose.utils import post_processor

__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(dg.Layer):
    """
    Generalized R-CNN. Any models that contains the following three components
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(self, *, 
                 backbone: dg.Layer,
                 proposal_generator: dg.Layer,
                 roi_fast_rcnn_heads: dg.Layer,
                 roi_body_uv_heads: dg.Layer
                 ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_fast_rcnn_heads = roi_fast_rcnn_heads
        self.roi_body_uv_heads = roi_body_uv_heads
    

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg,),
            "roi_fast_rcnn_heads": build_fast_rcnn_heads(cfg, ),
            "roi_body_uv_heads": build_body_uv_rcnn_heads(cfg, ),
        }


    def forward(self, image, im_info, im_ori_shape):
        features = self.backbone(image)
        rois = self.proposal_generator(features, im_info) 
        cls_prob, bbox_pred, lod_rcnn = self.roi_fast_rcnn_heads(features[1:], rois)

        all_rois = np.concatenate([o[:, 1:] / im_info[0][-1] for o in rois])
        scores, boxes, cls_boxes = post_processor.get_boxes(all_rois, cls_prob.numpy(), bbox_pred.numpy(), im_ori_shape)

        body_rois, rois_idx_order = post_processor.get_body_rois(boxes, im_info[0][-1])
        ann_index, index, u, v, lod_uv = self.roi_body_uv_heads(features[1:], body_rois)
        
        cls_bodys = post_processor.get_body_part(boxes, ann_index.numpy(), index.numpy(), u.numpy(), v.numpy())
        
        results = {'rois': rois, 'boxes': boxes, 'cls_boxes': cls_boxes[1], 'cls_bodys': cls_bodys}
        return results











