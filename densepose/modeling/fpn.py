# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from .backbone import BACKBONE_REGISTRY
from .resnet import ResNet101

from .model_lib import (
    conv2d, 
    conv1x1, 
    conv3x3, 
)


class FPN(dg.Layer):
    """
    This module implements :paper:`FPN`
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, top_block=None):
        super().__init__()

        self.bottom_up = bottom_up
        self.in_features = in_features
        self.lateral_convs = []
        self.output_convs = []
        in_channels_per_feature = [2048, 1024, 512, 256]
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = conv1x1(in_channels, out_channels,)
            output_conv = conv3x3(out_channels, out_channels)
            self.add_sublayer("fpn_lateral{}".format(5 - idx), lateral_conv)
            self.add_sublayer("fpn_output{}".format(5 - idx), output_conv)

            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        self.top_block = top_block

    
    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features]

        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = L.image_resize(prev_features, scale=2, resample='NEAREST')
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features

            results.append(output_conv(prev_features))
        
        if self.top_block is not None:
            P6_feature = self.top_block(results[0])
            results.insert(0, P6_feature)
        
        return results


class LastLevelMaxPool(dg.Layer):

    def forward(self, x):
        return L.pool2d(x, pool_size=1, pool_type="max", pool_stride=2, pool_padding=0)


@BACKBONE_REGISTRY.register()
def ResNet101_FPN(cfg, ):
    bottom_up = ResNet101(cfg,)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up, in_features, out_channels, LastLevelMaxPool())

    return backbone
