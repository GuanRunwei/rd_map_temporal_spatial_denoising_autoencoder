import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class depth_seperate_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(depth_seperate_deconv, self).__init__()
        self.deep_deconv = nn.ConvTranspose2d(
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
        output = self.deep_deconv(x)
        output = self.point_conv(output)
        return output


class ds_basic_deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ds_basic_deconv, self).__init__()
        self.ds_conv = depth_seperate_deconv(in_channels=in_channels, out_channels=out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.ds_conv(x)
        output = self.batchnorm(output)
        output = self.activation(output)
        return output


if __name__ == '__main__':
    input = torch.randn(8, 16, 64, 32)
    model = ds_basic_deconv(in_channels=16, out_channels=8)
    print(model(input).shape)
