import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.nn import Conv3d, BatchNorm3d, ReLU, Conv2d, BatchNorm2d
from torch.nn import AdaptiveAvgPool2d, AdaptiveAvgPool3d
import numpy as np
from lib.blocks.BasicBlocks import *

class DeepLabv3_Head(nn.Module):
    """
    """
    def __init__(self, in_filters, n_classes, dims, last_layer=True):
        """If last_layer=True then it includes the final
           in_filters -> n_classes 1x1 convolution to produce the logits.
           If false, it won't, which is used by DeepLabv3Plus.
        """
        super(DeepLabv3_Head, self).__init__()

        if dims == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dims == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.project = nn.Sequential(
                Conv(in_filters*5, in_filters, 1),
                BN(in_filters),
                ReLU(),
                )

        self.last = None
        if last_layer:
            self.last = Conv(in_filters, n_classes, 1)

    def forward(self, x):
        x = self.project(x)
        if self.last != None:
            return self.last(x)
        return x

class DeepLabv3_ASPP_ImPooling(nn.Module):
    def __init__(self, in_filters, out_filters, dim):
        super(DeepLabv3_ASPP_ImPooling, self).__init__()

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
            AvgPool = AdaptiveAvgPool2d
            self.interpol_mode = "bilinear"
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d
            AvgPool = AdaptiveAvgPool3d
            self.interpol_mode = "trilinear"

        self.seq = nn.Sequential(
                AvgPool(1), # (Output = # of filters, avg across the HxW(xD)
                Conv(in_filters, out_filters, 1),
                #BN(out_filters), # This BN will throw an error if batch = 1
                ReLU(),
                )
        self.interpolate = Interpolate()

    def forward(self, x):
        out = self.seq(x)
        out = self.interpolate(out, x.shape[2:], mode=self.interpol_mode,
                align_corners=False)
        return out

class DeepLabv3_ASPP(nn.Module):
    """Head of DeepLabv3.
       This head combines: (a) 1x1, 3x3, 3x3, 3x3 and (b) global avg pooling.

    """

    def __init__(self, in_filters, out_filters, dilation_rates, dim):
        super(DeepLabv3_ASPP, self).__init__()

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.aspp = nn.ModuleList()
        self.aspp.append(nn.Sequential(
                Conv(in_filters, out_filters, 1),
                BN(out_filters),
                ReLU(),
                ))

        for rate in dilation_rates:
            self.aspp.append(nn.Sequential(
                Conv(in_filters, out_filters, 3, dilation=rate, padding=rate),
                BN(out_filters),
                ReLU(),
                ))

        self.pooling = DeepLabv3_ASPP_ImPooling(in_filters, out_filters, dim)

        self.cat = Cat()

    def forward(self, x):
        
        out = []
        for i in range(len(self.aspp)):
            out.append(self.aspp[i](x))
        out.append(self.pooling(x))

        out = self.cat(out, dim=1)
        return out


class DeepLabv3Plus_ResNetBlock(nn.Module):
    """
    """
    def __init__(self, in_filters, dims):
        """If last_layer=True then it includes the final
           in_filters -> n_classes 1x1 convolution to produce the logits.
           If false, it won't, which is used by DeepLabv3Plus.
        """
        super(DeepLabv3Plus_ResNetBlock, self).__init__()

        if dims == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dims == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.seq = nn.Sequential(
                ReLU(),
                BN(in_filters),
                Conv(in_filters, in_filters, 3, padding=1),
                ReLU(),
                BN(in_filters),
                Conv(in_filters, in_filters, 3, padding=1),
                )

    def forward(self, x):
        x = self.seq(x) + x
        return x

class Sep_Conv_DeepLabv3Plus(nn.Module):
    def __init__(self, filters, stride, dilation, skip_con, dim,
            return_skip=False):
        """filters: list of 4 filters for the 3 convolutions
           stride: stride of the last conv.
           dilation: dilation rate
           skip_con: type of skip connection (sum, none, conv)
           dim: dimensions
           return_skip: whether to return the second seq for the skip conenction
        """
        super(Sep_Conv_DeepLabv3Plus, self).__init__()
        self.return_skip = return_skip

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.seq1 = nn.Sequential(
                Depthwise_Sep_Conv(filters[0], filters[1],
                    Conv, dilation=dilation,
                    stride=1),
                BN(filters[1]),
                ReLU()
                )

        self.seq2 = nn.Sequential(
                Depthwise_Sep_Conv(filters[1], filters[2],
                    Conv, dilation=dilation,
                    stride=1),
                BN(filters[2]),
                ReLU()
                )

        self.seq3 = nn.Sequential(
                Depthwise_Sep_Conv(filters[2], filters[3],
                    Conv, dilation=dilation,
                    stride=stride),
                BN(filters[3]),
                ReLU()
                )

        # Skip connection (if any)
        skip_layers = []
        if skip_con == "sum":
            self.skip = nn.Sequential() # Identity
        elif skip_con == "conv":
            self.skip = nn.Sequential(
                    Conv(filters[0], filters[-1], 1, stride=stride),
                    BN(filters[-1])
                    )
        else:
            self.skip = None

        self.sum = Sum()

    def forward(self, x):
        x1 = self.seq1(x)
        x1 = self.seq2(x1) # This may be returned for the skip connection
        out = self.seq3(x1)

        if self.skip != None:
            out = self.sum([self.skip(x), out])

        if self.return_skip:
            return out, x1
        else:
            return out

class Depthwise_Sep_Conv(nn.Module):
    """Generic 'Sep Conv' used by Xception_DeepLabv3Plus (Fig. 4)
    """
    def __init__(self, in_filters, out_filters, Conv, dilation=1, stride=1):
        super(Depthwise_Sep_Conv, self).__init__()

        self.conv = Conv(in_filters, in_filters, 3, padding=dilation,
                groups=in_filters, dilation=dilation, stride=stride)
        self.pointwise = Conv(in_filters, out_filters, 1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

