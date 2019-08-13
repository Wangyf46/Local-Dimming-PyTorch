import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class UNet_Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Downsampling, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 64, k_size=9, s_size=5)
        self.down2 = down(64, 64, k_size=7, s_size=4)
        self.down3 = down(64, 64, k_size=5, s_size=3)
        self.down4 = down(64, 64, k_size=3, s_size=2)

        self.outc = outconv(64, out_channels)

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

    def forward(self, x):          # torch.Size([1, 3, 1080, 1920])
        x1 = self.inc(x)           # torch.Size([1, 64, 1080, 1920])

        x2 = self.down1(x1)        # torch.Size([1, 64, 216, 384])
        x3 = self.down2(x2)        # torch.Size([1, 64, 54, 96])
        x4 = self.down3(x3)        # torch.Size([1, 64, 18, 32])
        x5 = self.down4(x4)        # torch.Size([1, 64, 9, 16])
        BL = self.outc(x5)    # torch.Size([1, 3, 1080, 1920])
        return BL


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x):
        output = self.conv(x)
        return output


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
								  nn.BatchNorm2d(out_channels),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
								  nn.BatchNorm2d(out_channels),
								  nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.conv(x)
        return output


class double_conv_skip_A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv_skip_A, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
								  nn.BatchNorm2d(64),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.add(x, y1)
        output = self.relu(y2)
        return output


class down(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=k_size//2, bias=True),   ## TODO padding
								    nn.BatchNorm2d(64),
								    double_conv_skip_A(64, out_channels))

    def forward(self, x):
        output = self.mpconv(x)
        return output


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.conv(x)
        output =  F.sigmoid(x1)   ##
        return(output)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    input = torch.rand(4, 3, 1080, 1920).cuda()
    net = UNet_Downsampling(3, 1).cuda()

    print(net)
    output = net(input)
    ipdb.set_trace()
    print(output.shape)
