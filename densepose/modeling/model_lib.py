import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg 


def conv2d(in_channels, out_channels, kernel_size, stride, padding, group=1, bias=None):
    return dg.Conv2D(num_channels=in_channels,
                     num_filters=out_channels, 
                     filter_size=kernel_size,
                     stride=stride, 
                     padding=padding, 
                     groups=group,
                     bias_attr=bias)


def conv1x1(in_channels, out_channels, stride=1, group=1, bias=None):
    return conv2d(in_channels, out_channels, 1, stride, 0, group, bias)


def conv3x3(in_channels, out_channels, stride=1, group=1, bias=None):
    return conv2d(in_channels, out_channels, 3, stride, 1, group, bias)


def conv7x7(in_channels, out_channels, stride=1, group=1, bias=None):
    return conv2d(in_channels, out_channels, 7, stride, 3, group, bias)


def deconv4x4(in_channels, out_channels, stride=2):
     return dg.Conv2DTranspose(num_channels=in_channels, 
        num_filters=out_channels, filter_size=4, padding=1, stride=stride,) 


class ReLU(dg.Layer):

    def forward(self, x):
        return L.relu(x)


def get_norm(norm_type, channels_num):
    if norm_type == "AN":
        return AffineNormalization(channels_num)
    elif norm_type == "BN":
        return dg.BatchNorm(channels_num)
    else:
        raise NotImplementedError()


class AffineNormalization(dg.Layer):

    def __init__(self, channels_num):

        super().__init__()
        self.channels_num = channels_num
        self.scale = self.create_parameter([channels_num])
        self.bias = self.create_parameter([channels_num])


    def forward(self, x):
        return L.affine_channel(x, self.scale, self.bias)





















