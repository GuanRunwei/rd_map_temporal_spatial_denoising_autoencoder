import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary

from nets.mobile_backbone import mobile_backbone_two_stage


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.avg_pool(x)
        # print("average pool shape:", output.shape)
        output = self.conv(output.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        output = self.sigmoid(output)

        return x * output.expand_as(x)


class mobile_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mobile_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_init = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*3, kernel_size=1, padding=0,
                                   stride=1)
        self.dropout_init = nn.Dropout(p=0.01)

        self.encoder_first = mobile_backbone_two_stage(in_channels=in_channels*9, out_channels=32, kernel_size=3,
                                                       stride=2, padding=1)
        self.eca_first = eca_block(channel=32)
        self.dropout1 = nn.Dropout(p=0.01)
        # self.maxpool_first = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))

        self.encoder_second = mobile_backbone_two_stage(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                                                        padding=1)
        self.eca_second = eca_block(channel=64)
        self.dropout2 = nn.Dropout(p=0.01)
        # self.maxpool_second = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))

        self.encoder_third = mobile_backbone_two_stage(in_channels=64, out_channels=out_channels, kernel_size=3,
                                                       stride=2, padding=1)
        self.eca_third = eca_block(channel=out_channels)
        self.dropout3 = nn.Dropout(p=0.01)
        # self.maxpool_third = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x_t, x_t_1, x_t_2 = x

        x_t = self.conv_init(x_t)
        x_t_1 = self.conv_init(x_t_1)
        x_t_2 = self.conv_init(x_t_2)
        # print(x_t.shape)

        x_total = torch.cat([x_t, x_t_1, x_t_2], dim=1)
        x_total = self.dropout_init(x_total)

        x_total = self.encoder_first(x_total)
        x_total = self.eca_first(x_total)
        x_total = self.dropout1(x_total)
        # x_total = self.maxpool_first(x_total)
        x_first_out = x_total
        # print("x_first_out.shape:", x_first_out.shape)

        x_total = self.encoder_second(x_total)
        x_total = self.eca_second(x_total)
        x_total = self.dropout2(x_total)
        # x_total = self.maxpool_second(x_total)
        x_second_out = x_total
        # print("x_second_out.shape:", x_second_out.shape)

        x_total = self.encoder_third(x_total)
        x_total = self.eca_third(x_total)
        x_total = self.dropout3(x_total)
        # x_total = self.maxpool_third(x_total)
        # print("x_total.shape:", x_total.shape)

        return x_total, x_first_out, x_second_out







if __name__ == '__main__':
    input = torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64)
    model = mobile_encoder(in_channels=1, out_channels=128)
    output = model(input)
    param_num = sum([param.nelement() for param in model.parameters()])
    print(param_num)


