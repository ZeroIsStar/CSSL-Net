# A multiscale feature fusion enhanced CNN with the multiscalechannel attention mechanism for eﬃcient landslide detection(MS2LandsNet) using medium-resolution remote sensing data
import torch
import torch.nn as nn


class DoubleConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class FFM(nn.Module):
    def __init__(self, ch_in):
        super(FFM, self).__init__()
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = nn.Conv2d(ch_in, ch_in, 1, 1, padding=0)
    def forward(self, x):
        x1, x2 = x
        o = torch.cat([x1, x2], 1)
        o = self.conv1x1(o)
        return o


class MS_CAM(nn.Module):
    def __init__(self, channels=64, inter_channels=4):
        super(MS_CAM, self).__init__()
        #         inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class MS2LandsNet(nn.Module):
    def __init__(self, n_classes=2, input_channel=4):
        super(MS2LandsNet, self).__init__()
        self.classes = n_classes
        self.input_channel = input_channel

        self.conv1 = DoubleConvPool(self.input_channel, 16)
        self.FFM1 = FFM(16 + self.input_channel)
        self.MS_CAM1 = MS_CAM(16 + self.input_channel, 16)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv2 = DoubleConvPool(16 + self.input_channel, 32)
        self.FFM2 = FFM(32 + self.input_channel)
        self.MS_CAM2 = MS_CAM(32 + self.input_channel, 32)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv3 = DoubleConvPool(32 + self.input_channel, 64)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.FFM3 = FFM(112 + self.input_channel)
        self.MS_CAM3 = MS_CAM(112 + self.input_channel, 64)
        self.conv4 = DoubleConv(112 + self.input_channel, 64)


        self.pooling = nn.AvgPool2d(2)
        self.outc = OutConv(64, self.classes)

    def forward(self, x):
        inputx = x
        x1 = self.conv1(x)
        x = self.pooling(x)

        x2 = x1, x
        x2 = self.conv2(self.MS_CAM1(self.FFM1(x2)))

        x = self.pooling(x)
        x3 = x2, x
        x3 = self.conv3(self.MS_CAM2(self.FFM2(x3)))

        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x3 = self.up3(x3)


        x3 = torch.cat([x1, x2, x3], dim=1)
        x5 = x3, inputx

        x5 = self.conv4(self.MS_CAM3(self.FFM3(x5)))
        x5 = self.outc(x5)

        return x5


if __name__ == "__main__":
    from thop import profile
    import time
    x = torch.randn(16, 14, 128, 128).cuda()
    model = MS2LandsNet(2,14).cuda()
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



