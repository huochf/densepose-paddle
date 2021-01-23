# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import paddle.fluid as F
from densepose.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `dg.Layer` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    print("Loading pretrained model from '/home/aistudio/densepose/pretrained_models/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pdparams'")
    state_dict, _ = F.load_dygraph('/home/aistudio/densepose/pretrained_models/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pdparams')
    model.load_dict(state_dict)
    return model
