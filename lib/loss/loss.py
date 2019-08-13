import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1] # todo
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    kernel /= kernel.sum() # todo
    return kernel


class SSIM_Loss(_Loss):
    def __init__(self, in_channels, size = 5, sigma = 1.5, size_average = True):
        super(SSIM_Loss, self).__init__(size_average)
        self.in_channels = in_channels
        self.size = size
        self.sigma = sigma
        self.size_average = size_average

        kernel = gaussian_kernel(self.size, self.sigma) # todo
        self.kernel_size = kernel.shape # 11*11
        weight = np.tile(kernel, (in_channels, 1, 1, 1))   # todo
        self.weight = Parameter(torch.from_numpy(weight).float(), requires_grad = False)   # todo

    def forward(self, img1, img2):
        mean1 = F.conv2d(img1, self.weight, padding = self.size, groups = self.in_channels)
        mean2 = F.conv2d(img2, self.weight, padding = self.size, groups = self.in_channels)
        mean1_sq = mean1 * mean1
        mean2_sq = mean2 * mean2
        mean_12 = mean1 * mean2

        sigma1_sq = F.conv2d(img1 * img1, self.weight, padding = self.size, groups = self.in_channels) - mean1_sq # padding = 0
        sigma2_sq = F.conv2d(img2 * img2, self.weight, padding = self.size, groups = self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(img1 * img2, self.weight, padding = self.size, groups = self.in_channels) - mean_12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mean_12 + C1) * (2 * sigma_12 + C2)) / ((mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out

class Loss(torch.nn.Module):
    def __init__(self, in_channels, size = 5, sigma = 1.5, size_average = True):
        super(Loss, self).__init__()
        self.ssim_loss = SSIM_Loss(in_channels, size, sigma, size_average)

    def forward(self, img1, img2):
        loss_c = self.ssim_loss(img1, img2)
        loss_e = torch.mean((img1 - img2) ** 2)  # TODO dim?? MSE
        loss = torch.add(torch.mul(loss_c, 0.001), loss_e)          #todo toch.mul ???
        return loss