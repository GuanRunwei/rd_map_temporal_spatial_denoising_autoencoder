import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nets.ds_deconv import ds_basic_deconv
from nets.ds_conv import ds_basic_conv


class de_mobile_backbone_stage1(nn.Module):
    def __init__(self, in_channels):
        super(de_mobile_backbone_stage1, self).__init__()
        self.in_channels = in_channels

        # ------------------------------------- 3分支->中间分支 ------------------------------------- #
        self.ds_basic_deconv = ds_basic_conv(in_channels=in_channels, out_channels=in_channels)
        self.batchnorm_center = nn.BatchNorm2d(in_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 3分支->左侧分支 ------------------------------------- #
        self.conv_left = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.batchnorm_left = nn.BatchNorm2d(in_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 3分支->右侧分支 ------------------------------------- #
        self.batchnorm_right = nn.BatchNorm2d(in_channels)
        # ------------------------------------------------------------------------------------------- #

        self.activation = nn.ReLU()

    def forward(self, x):
        x_left = x
        x_right = x
        x_center = x

        # --------------------------------- 中间分支 --------------------------------- #
        output_center = self.ds_basic_deconv(x_center)
        output_center = self.batchnorm_center(output_center)
        # print("中间分支:", output_center.shape)
        # ---------------------------------------------------------------------------- #

        # --------------------------------- 左侧分支 --------------------------------- #
        output_left = self.conv_left(x_left)
        output_left = self.batchnorm_left(output_left)
        # print("左侧分支:", output_left.shape)
        # ---------------------------------------------------------------------------- #

        # --------------------------------- 右侧分支 --------------------------------- #
        output_right = self.batchnorm_right(x_right)
        # print("右侧分支:", output_right.shape)
        # ---------------------------------------------------------------------------- #

        output = output_center + output_left + output_right

        output = self.activation(output)

        return output


class de_mobile_backbone_stage2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(de_mobile_backbone_stage2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ------------------------------------- 2分支->左侧分支 ------------------------------------- #
        # self.conv_left = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
        #                                     padding=1)
        self.conv_left = ds_basic_deconv(in_channels=in_channels, out_channels=out_channels)
        self.batchnorm_left = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        # ------------------------------------- 2分支->右侧分支 ------------------------------------- #
        self.conv_right = ds_basic_deconv(in_channels=in_channels, out_channels=out_channels)
        self.batchnorm_right = nn.BatchNorm2d(out_channels)
        # ------------------------------------------------------------------------------------------- #

        self.activation = nn.ReLU()

    def forward(self, x):
        x_left = x
        x_right = x

        output_left = self.conv_left(x_left)
        output_left = self.batchnorm_left(output_left)
        # print("deconv 左侧分支:", output_left.shape)

        output_right = self.conv_right(x_right)
        output_right = self.batchnorm_right(output_right)
        # print("deconv 右侧分支:", output_right.shape)

        output = output_left + output_right

        output = self.activation(output)

        return output


class de_mobile_backbone_two_stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(de_mobile_backbone_two_stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.de_conv_stage1 = de_mobile_backbone_stage1(in_channels=in_channels)
        self.de_conv_stage2 = de_mobile_backbone_stage2(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        output = self.de_conv_stage1(x)
        output = self.de_conv_stage2(output)
        return output


if __name__ == '__main__':
    input = torch.randn(8, 16, 64, 32)
    model = de_mobile_backbone_two_stage(in_channels=16, out_channels=8)
    print(model(input).shape)