import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class EDSR(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=64, n_resblocks=16, kernel_size=3):
        super(EDSR, self).__init__()
        scales = [2,3,4,5]
        self.head = conv(in_channels, 64, kernel_size)

        m_body = [ResBlock(n_feats, kernel_size) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

        m_upsample = [Upsample(n_feats, kernel_size, scale) for scale in range(scales[0], scales[-1]+1)]
        self.upsample = nn.Sequential(*m_upsample)

        self.tail = conv(64, out_channels, kernel_size)

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x2 += x1
        x3 = self.upsample(x2)
        x4 = self.tail(x3)
        output = F.sigmoid(x4)  ##
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


class Upsample(nn.Module):
    def __init__(self, n_feats, kernel_size, scale, bn=False, act=False):
        super(Upsample, self).__init__()
        Layers = []
        if scale == 2:
            Layers.append(conv(n_feats, 4 * n_feats, kernel_size))
            Layers.append(nn.PixelShuffle(2))
            if bn:
                Layers.append(nn.BatchNorm2d(n_feats))
            if act:
                Layers.append(nn.ReLU(inplace=True))

        elif scale == 3:
            Layers.append(conv(n_feats, 9 * n_feats, kernel_size))
            Layers.append(nn.PixelShuffle(3))
            if bn:
                Layers.append(nn.BatchNorm2d(n_feats))
            if act:
                Layers.append(nn.ReLU(inplace=True))

        elif scale == 4:
            for _ in range(int(math.log(scale, 2))):
                Layers.append(conv(n_feats, 4 * n_feats, kernel_size))
                Layers.append(nn.PixelShuffle(2))
                if bn:
                    Layers.append(nn.BatchNorm2d(n_feats))
                if act:
                    Layers.append(nn.ReLU(inplace=True))

        elif scale ==5:
            Layers.append(conv(n_feats, 25 * n_feats, kernel_size))
            Layers.append(nn.PixelShuffle(5))
            if bn:
                Layers.append(nn.BatchNorm2d(n_feats))
            if act:
                Layers.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        self.upblock = nn.Sequential(*Layers)

    def forward(self, x):
        output = self.upblock(x)
        return output


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    input = torch.rand(8, 1, 9, 16).cuda()    # bz = 1
    net = EDSR(1, 1).cuda()

    print(net)
    output = net(input)
    print(output.shape)
    ipdb.set_trace()