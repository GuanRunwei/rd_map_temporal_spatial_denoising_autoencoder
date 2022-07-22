import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class depth_seperate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(depth_seperate_conv, self).__init__()
        self.deep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        output = self.deep_conv(x)
        output = self.point_conv(output)
        return output


class ds_basic_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ds_basic_conv, self).__init__()
        self.ds_conv = depth_seperate_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding)
        self.activation = nn.ReLU(inplace=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.ds_conv(x)
        output = self.activation(output)
        output = self.batchnorm(output)
        return output
