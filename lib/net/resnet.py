import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


class  RESNET(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=64, n_resblocks=16, kernel_size=3): ## TODO
        super(RESNET, self).__init__()
        self.head = conv(in_channels, 64, kernel_size)
        m_body = [ResBlock(n_feats, kernel_size) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

        self.up1 = up(64, 64, k_size=3, s_size=2)
        self.up2 = up(64, 64, k_size=5, s_size=3)
        self.up3 = up(64, 64, k_size=7, s_size=4)
        self.up4 = up(64, 64, k_size=9, s_size=5)

        self.tail = conv(64, out_channels, kernel_size)   ## 3x3 TODO

        ## Vavier initialized method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.01)
                # nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x2 += x1

        x3 = self.up1(x2)
        x4 = self.up2(x3)
        x5 = self.up3(x4)
        x6 = self.up4(x5)

        x7 = self.tail(x6)
        output = F.sigmoid(x7)  ##
        return output


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), stride=1, bias=True)

    def forward(self, x):
        output = self.conv(x)
        return output


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bn=False, res_scale=0.1):
        super(ResBlock, self).__init__()
        Layers = []
        for i in range(2):
            Layers.append(conv(n_feats, n_feats, kernel_size))
            if bn:
                Layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                Layers.append(nn.ReLU(inplace=True))

        self.resblock = nn.Sequential(*Layers)
        self.res_sacle = res_scale

    def forward(self, x):
        res = self.resblock(x).mul(self.res_sacle)
        res += x
        return res



class double_conv_skip_A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv_skip_A, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
								  # nn.BatchNorm2d(64),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
                                  # nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.add(x, y1)
        output = self.relu(y2)
        return output


class up(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=s_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=1)
        self.conv=double_conv_skip_A(in_channels, out_channels)

    def forward(self, y):
        y1 = self.up(y)
        output = self.conv(y1)
        return output


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    input = torch.rand(16, 1, 9, 16).cuda()    # bz = 1
    net = RESNET(1, 1)
    net = nn.DataParallel(net).cuda()

    #print(net)
    output = net(input)
    print(output.shape)
