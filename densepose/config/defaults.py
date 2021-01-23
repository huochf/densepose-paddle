# -------------------------------------------------------------------
# Copy from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"


_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "ResNet101_FPN"


_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.NORM = "AN"
_C.MODEL.RESNET.NUM_GROUPS = 32
_C.MODEL.RESNET.WIDTH_PER_GROUP = 8
_C.MODEL.RESNET.STEM_IN_CHANNELS = 3
_C.MODEL.RESNET.STEM_OUT_CHANNELS = 64
_C.MODEL.RESNET.STRIDE_IN_1X1 = False
_C.MODEL.RESNET.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']


_C.MODEL.FPN = CN()
_C.MODEL.FPN.IN_FEATURES =  ['res5', 'res4', 'res3', 'res2']
_C.MODEL.FPN.OUT_CHANNELS = 256


_C.MODEL.RPN = CN()
_C.MODEL.RPN.NAME = "RPN"
_C.MODEL.RPN.HEAD = "StandardRPNHead"
_C.MODEL.RPN.MAX_LEVEL = 6
_C.MODEL.RPN.MIN_LEVEL = 2
_C.MODEL.RPN.NUM_ANCHORS = 3
_C.MODEL.RPN.BOX_DIM = 4
_C.MODEL.RPN.ANCHOR_START_SIZE = [32]
_C.MODEL.RPN.ASPECT_RATIOS = [0.5, 1, 2]
_C.MODEL.RPN.PRE_NMS_TOP_N = 1000
_C.MODEL.RPN.POST_NMS_TOP_N = 1000
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 0


_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD = CN()
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NAME = "FastRCNNHead"
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.ROI_XFORM_RESOLUTION = 7
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.ROI_XFORM_SAMPLING_RATIO = 2
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NUM_CLASSES = 2
_C.MODEL.ROI_HEADS.FAST_RCNN_HEAD.NUM_BBOX_REG_CLASSES = 2

_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD = CN()
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NAME = "BodyUVHead"
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NUM_STACKED_CONVS = 8
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.CONV_HEAD_DIM = 512
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.NUM_PATCHES = 24
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.ROI_XFORM_RESOLUTION = 14
_C.MODEL.ROI_HEADS.BODY_UV_RCNN_HEAD.ROI_XFORM_SAMPLING_RATIO = 2


