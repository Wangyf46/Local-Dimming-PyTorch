import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


'''
    Debug:
          1.outconv
          2.ReLu, BatchNormalization
'''


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 64, k_size=9, s_size=5)
        self.down2 = down(64, 64, k_size=7, s_size=4)
        self.down3 = down(64, 64, k_size=5, s_size=3)
        self.down4 = down(64, 64, k_size=3, s_size=2)
        self.up1 = up_3x3(64, 64, k_size=3, s_size=2)
        self.up2 = up(128, 64, k_size=5, s_size=3)
        self.up3 = up(128, 64, k_size=7, s_size=4)
        self.up4 = up_9x9(128, 64, k_size=9, s_size=5)
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

        y1 = self.up1(x5)          # torch.Size([1, 64, 18, 32])
        y2 = self.up2(y1, x4)      # torch.Size([1, 64, 54, 96])
        y3 = self.up3(y2, x3)      # torch.Size([1, 64, 216, 384])
        y4 = self.up4(y3, x2)      # torch.Size([1, 64, 1080, 1920])

        output = self.outc(y4, x)    # torch.Size([1, 3, 1080, 1920])
        return output


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
								  # nn.BatchNorm2d(out_channels),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
								  # nn.BatchNorm2d(out_channels),
								  nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.conv(x)
        return output


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


class double_conv_skip_B(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv_skip_B, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
								  # nn.BatchNorm2d(64),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
                                  # nn.BatchNorm2d(64))
        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))
                                      # nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.conv_1x1(x)
        y3 = torch.add(y1, y2)
        output = self.relu(y3)
        return output


class down(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=k_size//2, bias=True),   ## TODO padding
								    # nn.BatchNorm2d(64),
								    double_conv_skip_A(64, out_channels))

    def forward(self, x):
        output = self.mpconv(x)
        return output


class up_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size, bilinear=True):
        super(up_3x3, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=s_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=1)
        self.conv=double_conv_skip_A(in_channels, out_channels)

    def forward(self, y):
        y1 = self.up(y)
        output = self.conv(y1)
        return output


class up(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=s_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=1)
        self.conv=double_conv_skip_B(in_channels, out_channels)

    def forward(self, y, x):
        out1 = torch.cat([y, x], dim=1)
        output = self.up(out1)
        output = self.conv(output)
        return output


class up_9x9(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, s_size, bilinear=True):
        super(up_9x9, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=s_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, 64, kernel_size=k_size, stride=s_size, padding=1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                  # nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
                                  # nn.BatchNorm2d(out_channels))

    def forward(self, y, x):
        y1 = torch.cat([y, x], dim=1)
        y2 = self.up(y1)
        output = self.conv(y2)
        return output


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv_skip = nn.Sequential(nn.Conv2d(3, in_channels, kernel_size=1, stride=1, padding=0))
                                       # nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, y4, x):
        out1 = self.conv_skip(x)
        out2 = torch.add(out1, y4)
        out3 = self.relu(out2)
        out4 = self.conv(out3)
        Icp =  F.sigmoid(out4)   ## TODO
        return(Icp)


if __name__ == '__main__':
    input = torch.rand(1, 3, 1080, 1920).cuda()  # bz = 1
    LDNN = UNet(3, 3).cuda()

    print(LDNN)
    output = LDNN(input)
    ipdb.set_trace()
    print(output.shape)
