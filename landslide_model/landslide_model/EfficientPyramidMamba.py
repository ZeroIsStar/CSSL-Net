import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba_ssm import Mamba


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class MambaLayer(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
                    ConvBNReLU(in_chs, dim, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1)
                    ))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(in_chs, dim, kernel_size=1)
                    ))
        self.mamba = Mamba(
            d_model=dim*self.pool_len+in_chs,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand # Block expansion factor
        )

    def forward(self, x): # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(Block, self).__init__()
        self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
        self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=128, num_classes=6, last_feat_size=16):
        super().__init__()
        self.b3 = Block(in_chs=encoder_channels[-1], dim=decoder_channels, last_feat_size=last_feat_size)
        self.up_conv = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  Conv(decoder_channels // 2, num_classes, kernel_size=1))

    def forward(self, x0, x3):
        x3 = self.b3(x3)
        x3 = self.up_conv(x3)
        x = x3 + self.pre_conv(x0)
        x = self.head(x)
        return x



class EfficientPyramidMamba(nn.Module):
    def __init__(self,
                 backbone_name='swsl_resnet18',
                 pretrained=False,
                 in_channels = 14,
                 num_classes=2,
                 decoder_channels=128,
                 last_feat_size=16  # last_feat_size=input_img_size // 32
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 4), pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(in_channels,  64,kernel_size=(7, 7), stride=(2, 2),padding=(3, 3),bias=False)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x = self.decoder(x0, x3)

        return x


if __name__ == '__main__':
    from thop import profile
    import time
    x = torch.randn(1, 14, 128, 128).cuda()
    model = EfficientPyramidMamba().cuda()
    flops, params = profile(model, inputs=(x,))
    num_runs = 10
    total_time = 0
    # 多次推理，计算平均推理时间
    for _ in range(num_runs):
        start_time = time.time()
        results = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)
    # 计算平均推理时间
    avg_inference_time = total_time / num_runs
    # 计算FPS
    fps = 1 / avg_inference_time
    print(f"FPS: {fps:.2f} frames per second")
    print(f'FLOPs: {flops / 1e9}G')
    print(f'params: {params / 1e6}M')
    print(model(x).shape)
