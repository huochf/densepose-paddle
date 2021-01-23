# -------------------------------------------------------------------
# Modified from detectron2(https://github.com/facebookresearch/detectron2)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from .backbone import BACKBONE_REGISTRY

from .model_lib import (
    conv2d, 
    conv1x1, 
    conv3x3, 
    conv7x7, 
    get_norm,
    AffineNormalization,
)


class BasicBlock(dg.Layer):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="AN"):
        
        super().__init__()
        raise NotImplementedError()
    

    def forward(self, x):
        raise NotImplementedError()


class BottleneckBlock(dg.Layer):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`. It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(self, 
                 in_channels, 
                 bottleneck_channels,
                 out_channels,
                 stride=1, 
                 group=1,
                 norm="AN", 
                 stride_in_1x1=True):
        super().__init__()

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels
        (stride1x1, stride3x3) = (stride, 1) if stride_in_1x1 else (1, stride)

        if in_channels != out_channels:
            self.branch1 = conv1x1(in_channels, out_channels, stride, bias=False)
            self.branch1_norm = get_norm(norm, out_channels)
        else:
            self.branch1 = None
        
        self.branch2a = conv1x1(in_channels, bottleneck_channels, stride1x1, bias=False)
        self.branch2a_norm = get_norm(norm, bottleneck_channels)

        self.branch2b = conv3x3(bottleneck_channels, bottleneck_channels, stride3x3, group=group, bias=False)
        self.branch2b_norm = get_norm(norm, bottleneck_channels)

        self.branch2c = conv1x1(bottleneck_channels, out_channels, 1, bias=False)
        self.branch2c_norm = get_norm(norm, out_channels)


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2a_norm(out)
        out = L.relu(out)

        out = self.branch2b(out)
        out = self.branch2b_norm(out)
        out = L.relu(out)

        out = self.branch2c(out)
        out = self.branch2c_norm(out)

        if self.branch1 is not None:
            shortcut = self.branch1(x)
            shortcut = self.branch1_norm(shortcut)
        else:
            shortcut = x
        
        out += shortcut
        out = L.relu(out)
        return out


class BasicStem(dg.Layer):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=64, norm="AN"):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv7x7(in_channels, out_channels, stride=2, bias=False)
        self.conv1_norm = get_norm(norm, out_channels)
        self.pool = dg.Pool2D(pool_size=3, pool_type='max', 
                              pool_stride=2, pool_padding=1)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_norm(x)
        x = L.relu(x)
        x = self.pool(x)
        return x


class ResNet(dg.Layer):
    """
    Implement :paper:`ResNet`
    """

    def __init__(self, stem, stages, out_features):

        super().__init__()
        self.stem = stem

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            name = "res" + str(i + 2)
            stage = dg.Sequential(*blocks)
        
            self.add_sublayer(name, stage)
            self.stages_and_names.append((name, stage))
        
        self._out_features = out_features
    

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        
        for name, stage in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        
        return outputs


    @staticmethod
    def make_stage(block_class, num_blocks, 
                   in_channels, bottleneck_channels, out_channels, group,
                   norm, stride_in_1x1,):
        blocks = []
        stride = 2 if in_channels != 64 else 1
        for i in range(num_blocks):
            block = block_class(in_channels, 
                                bottleneck_channels, 
                                out_channels, 
                                stride,
                                group,
                                norm,
                                stride_in_1x1)
            blocks.append((str(i), block))
            in_channels = out_channels
            stride = 1
        return blocks


@BACKBONE_REGISTRY.register()
def ResNet101(cfg):
    stem = BasicStem(cfg.MODEL.RESNET.STEM_IN_CHANNELS,
                     cfg.MODEL.RESNET.STEM_OUT_CHANNELS,
                     cfg.MODEL.RESNET.NORM)
    in_channels = [64, 256, 512, 1024]
    hidden_dim = cfg.MODEL.RESNET.NUM_GROUPS * cfg.MODEL.RESNET.WIDTH_PER_GROUP
    bottleneck_channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]
    out_channels = [256, 512, 1024, 2048]
    stages = []
    for i, block_num in enumerate([3, 4, 23, 3]):
        stage = ResNet.make_stage(BottleneckBlock,
                                  block_num,
                                  in_channels[i],
                                  bottleneck_channels[i],
                                  out_channels[i],
                                  cfg.MODEL.RESNET.NUM_GROUPS,
                                  cfg.MODEL.RESNET.NORM,
                                  cfg.MODEL.RESNET.STRIDE_IN_1X1)
        stages.append(stage)
    
    return ResNet(stem, stages, cfg.MODEL.RESNET.OUT_FEATURES)
