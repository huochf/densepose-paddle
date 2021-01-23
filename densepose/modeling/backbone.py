# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------

from densepose.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, which extract feature maps from images
"""


def build_backbone(cfg, ):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone
