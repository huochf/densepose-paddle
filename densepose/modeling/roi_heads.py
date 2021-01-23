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

from densepose.modeling.model_lib import conv3x3, deconv4x4
from densepose.config import configurable
from densepose.utils.registry import Registry

ROI_FAST_RCNN_HEADS_REGISTRY = Registry("ROI_RCNN_HEADS")
ROI_FAST_RCNN_HEADS_REGISTRY.__doc__ = """
Registry for heads in RCNN branch for densepose model
"""

ROI_BODY_UV_HEADS_REGISTRY = Registry("ROI_BODY_UV_HEADS")
ROI_BODY_UV_HEADS_REGISTRY.__doc__ = """
Registry for heads in body UV branch for densepose model
"""


def build_fast_rcnn_heads(cfg, ):
    return ROI_FAST_RCNN_HEADS_REGISTRY.get(cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NAME)(cfg)


def build_body_uv_rcnn_heads(cfg, ):
    return ROI_BODY_UV_HEADS_REGISTRY.get(cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NAME)(cfg)


@ROI_FAST_RCNN_HEADS_REGISTRY.register()
class FastRCNNHead(dg.Layer):

    @configurable
    def __init__(self, mlp_head_dim, num_classes, num_bbox_reg_classes,
        roi_size, roi_spatial_scale, roi_sampling_ratio):
        super().__init__()
        in_dim = 256 # FPN output dimension
        self.mlp_head_dim = mlp_head_dim
        self.roi_size = roi_size
        self.roi_spatial_scale = roi_spatial_scale
        self.roi_sampling_ratio = roi_sampling_ratio

        self.fc6 = dg.Linear(in_dim * roi_size * roi_size, mlp_head_dim)
        self.fc7 = dg.Linear(mlp_head_dim, mlp_head_dim)
        self.cls_score = dg.Linear(mlp_head_dim, num_classes)
        self.bbox_pred = dg.Linear(mlp_head_dim, num_bbox_reg_classes * 4)
    

    @classmethod
    def from_config(cls, cfg,):
        ret = {
            "mlp_head_dim": cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.MLP_HEAD_DIM,
            "num_classes": cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NUM_CLASSES,
            "num_bbox_reg_classes": cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NUM_BBOX_REG_CLASSES,
            "roi_size": cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.ROI_XFORM_RESOLUTION,
            "roi_spatial_scale": 4, # default: p2 from FPN
            "roi_sampling_ratio": cfg.MODEL.ROI_HEADS.FAST_RCNN_HEAD.ROI_XFORM_SAMPLING_RATIO,

        }
        return ret


    def forward(self, features, rois,):
        # rois: list of ndarray (size: [n, 5]), p2 -> p5
        # features: p5 -> p2

        roi_features = []
        roi_lvl_batch_count = []
        for lvl in range(2, 6):
            feature_map = features[5 - lvl]
            spatial_scale = 1 / 2. ** lvl
            lvl_rois = rois[lvl - 2]
            if len(lvl_rois) == 0:
                continue
            b = feature_map.shape[0]
            batch_roi_count = []
            for i in range(b):
                batch_num = np.sum(lvl_rois[:, 0] == i)
                batch_roi_count.append(batch_num)
            
            roi_lvl_batch_count.append(batch_roi_count)

            lvl_rois = F.create_lod_tensor(lvl_rois[:, 1:], [batch_roi_count], place=F.CPUPlace())
            lvl_rois = dg.to_variable(lvl_rois)
            batch_roi_count = dg.to_variable(batch_roi_count).astype("int32")
            lvl_roi_features = L.roi_align(feature_map, 
                                           lvl_rois,
                                           pooled_height=self.roi_size,
                                           pooled_width=self.roi_size,
                                           spatial_scale=spatial_scale,
                                           sampling_ratio=self.roi_sampling_ratio,
                                           rois_num=batch_roi_count)
            c = lvl_roi_features.shape[1]
            lvl_roi_features = L.reshape(lvl_roi_features, (-1, self.roi_size * self.roi_size * c))
            roi_features.append(lvl_roi_features)
        
        roi_features = L.concat(roi_features, 0)
        lvl_lod = [0, ]
        batch_lod = [0, ]
        for i in range(len(roi_lvl_batch_count)):
            for j in range(len(roi_lvl_batch_count[i])):
                batch_lod.append(batch_lod[-1] + roi_lvl_batch_count[i][j])
            lvl_lod.append(lvl_lod[-1] + len(roi_lvl_batch_count[i]))
        # roi_features.set_lod([lvl_lod, batch_lod])

        fc6 = self.fc6(roi_features)
        fc6 = L.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = L.relu(fc7)

        cls_score = self.cls_score(fc7)
        cls_prob = L.softmax(cls_score)

        bbox_pred = self.bbox_pred(fc7)

        return cls_prob, bbox_pred, [lvl_lod, batch_lod]


@ROI_BODY_UV_HEADS_REGISTRY.register()
class BodyUVHead(dg.Layer):
    
    @configurable
    def __init__(self, num_stacked_convs, hidden_dim, num_patches,
        roi_size, roi_sampling_ratio):
        super().__init__()

        self.num_stacked_convs = num_stacked_convs
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        self.roi_size = roi_size
        self.roi_spatial_scale = 4
        self.roi_sampling_ratio = roi_sampling_ratio

        convs = []
        in_channels = 256 
        for i in range(num_stacked_convs):
            conv = conv3x3(in_channels, hidden_dim)
            self.add_sublayer("body_conv_fcn" + str(i + 1), conv)
            convs.append(conv)
            in_channels = hidden_dim
        self.convs = convs

        self.AnnIndex_lowres = deconv4x4(hidden_dim, 15)
        self.Index_UV_lowres = deconv4x4(hidden_dim, self.num_patches + 1)
        self.U_lowres = deconv4x4(hidden_dim, self.num_patches + 1)
        self.V_lowres = deconv4x4(hidden_dim, self.num_patches + 1)

        self.AnnIndex = deconv4x4(num_patches + 1, num_patches + 1)
        self.Index_UV = deconv4x4(num_patches + 1, num_patches + 1)
        self.U_estimated = deconv4x4(num_patches + 1, num_patches + 1)
        self.V_estimated = deconv4x4(num_patches + 1, num_patches + 1)


    @classmethod
    def from_config(cls, cfg,):
        ret = {
            "num_stacked_convs": cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NUM_STACKED_CONVS,
            "hidden_dim": cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.CONV_HEAD_DIM,
            "num_patches": cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NUM_PATCHES,
            "roi_size": cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.ROI_XFORM_RESOLUTION,
            "roi_sampling_ratio": cfg.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.ROI_XFORM_SAMPLING_RATIO,
        }
        return ret


    def forward(self, features, rois):
        # rois: list of ndarray (size: [n, 5]), p2 -> p5
        # features: p5 -> p2

        roi_features = []
        roi_lvl_batch_count = []
        for lvl in range(2, 6):
            feature_map = features[5 - lvl]
            spatial_scale = 1 / 2. ** lvl
            lvl_rois = rois[lvl - 2]
            if len(lvl_rois) == 0:
                continue
            b = feature_map.shape[0]
            batch_roi_count = []
            for i in range(b):
                batch_num = np.sum(lvl_rois[:, 0] == i)
                batch_roi_count.append(batch_num)
            
            roi_lvl_batch_count.append(batch_roi_count)

            lvl_rois = F.create_lod_tensor(lvl_rois[:, 1:].astype("float32"), [batch_roi_count], place=F.CPUPlace())
            lvl_rois = dg.to_variable(lvl_rois)
            batch_roi_count = dg.to_variable(batch_roi_count).astype("int32")
            lvl_roi_features = L.roi_align(feature_map, 
                                           lvl_rois,
                                           pooled_height=self.roi_size,
                                           pooled_width=self.roi_size,
                                           spatial_scale=spatial_scale,
                                           sampling_ratio=self.roi_sampling_ratio,
                                           rois_num=batch_roi_count)
            roi_features.append(lvl_roi_features)
        
        if roi_features == []:
            return None, None, None, None, None

        roi_features = L.concat(roi_features) # (n, c, 14, 14)

        for conv in self.convs:
            roi_features = conv(roi_features)
            roi_features = L.relu(roi_features)
        
        ann_index_lowres = self.AnnIndex_lowres(roi_features)
        index_UV_lowres = self.Index_UV_lowres(roi_features)
        u_lowres = self.U_lowres(roi_features)
        v_lowres = self.V_lowres(roi_features)

        b, _, h, w = ann_index_lowres.shape
        zero_out = dg.to_variable(np.zeros((b, 10, h, w)).astype("float32"))
        ann_index_lowres = L.concat([ann_index_lowres, zero_out], 1)
        ann_index = self.AnnIndex(ann_index_lowres)
        index = self.Index_UV(index_UV_lowres)
        u = self.U_estimated(u_lowres)
        v = self.V_estimated(v_lowres)
        
        lvl_lod = [0, ]
        batch_lod = [0, ]
        for i in range(len(roi_lvl_batch_count)):
            for j in range(len(roi_lvl_batch_count[i])):
                batch_lod.append(batch_lod[-1] + roi_lvl_batch_count[i][j])
            lvl_lod.append(lvl_lod[-1] + len(roi_lvl_batch_count[i]))
        # roi_features.set_lod([lvl_lod, batch_lod])

        return ann_index, index, u, v, [lvl_lod, batch_lod]

