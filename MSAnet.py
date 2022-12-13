import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.attention import DANetHead, CrissCrossAttention

attention_blocks = [DANetHead, CrissCrossAttention]


class PPM(nn.Module):
    def __init__(self, phi, in_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        '''自加'''
        self.lastConv = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False)
        self.phi = phi
        if phi == 1:
            self.attention = attention_blocks[phi - 1](2048, 2048)
        elif phi == 2:
            self.attention = attention_blocks[phi - 1](2048)

    def forward(self, x):
        x_size = x.size()
        out = [x]

        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)

        # 把原图和四个池化图cat起来
        outs = torch.cat(out, 1)

        outs = self.lastConv(outs)
        '''自加'''
        if self.phi >= 1 and self.phi <= 3:
            outs = self.attention(outs)

        return outs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, phi, num_classes=21, pretrained=False, backbone='resnet50'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        # self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        '''注意力'''
        # self.phi = phi
        # if phi >= 1 and phi <= 3:
        #     self.feat5_attention = attention_blocks[phi - 1](2048,2048)
        self.ppm = PPM(phi, 2048, 512, [1, 2, 3, 6])

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # if self.phi >= 1 and self.phi <= 3:
        #     feat5 = self.feat5_attention(feat5)
        feat5 = self.ppm(feat5)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        # final = self.final(up1)

        return up1

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


# inputs = torch.rand(2, 1, 512, 512)
# a = Unet(phi=1)
#
# print(a(inputs).size())
