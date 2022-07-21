import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary


from nets.de_mobile_backbone import de_mobile_backbone_two_stage
from nets.mobile_encoder import eca_block
from nets.mobile_encoder import mobile_encoder


class nano_sta(nn.Module):
    def __init__(self, encoder_in_channels, encoder_out_channels, decoder_out_channels):
        super(nano_sta, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.encoder_out_channels = encoder_out_channels
        self.decoder_out_channels = decoder_out_channels

        # ----------------------- Encoder ------------------------- #
        self.nano_sta_encoder = mobile_encoder(in_channels=encoder_in_channels, out_channels=encoder_out_channels)
        # --------------------------------------------------------- #

        # ----------------------- Decoder ------------------------- #
        self.deconv1 = de_mobile_backbone_two_stage(in_channels=encoder_out_channels, out_channels=64)
        self.de_eca_first = eca_block(channel=64)
        self.dropout1 = nn.Dropout(p=0.05)

        self.deconv2 = de_mobile_backbone_two_stage(in_channels=64, out_channels=32)
        self.de_eca_second = eca_block(channel=32)
        self.dropout2 = nn.Dropout(p=0.05)

        self.deconv3 = de_mobile_backbone_two_stage(in_channels=32, out_channels=decoder_out_channels)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        encoder_out, encoder_first_out, encoder_second_out = self.nano_sta_encoder(x)

        output = self.deconv1(encoder_out)
        output = output + encoder_second_out
        output = self.de_eca_first(output)
        output = self.dropout1(output)

        output = self.deconv2(output)
        output = output + encoder_first_out
        output = self.de_eca_second(output)
        output = self.dropout2(output)

        output = self.deconv3(output)
        output = self.activation(output)

        return output


if __name__ == '__main__':
    input = torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64), torch.randn(4, 1, 128, 64)
    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1)
    output = nano_sta_model(input)
    param_num = sum([param.nelement() for param in nano_sta_model.parameters()])
    print("Nano-sta 参数数量：", param_num)
    print("Nano-sta 模型结构：", nano_sta_model)
    print("output shape:", output.shape)




