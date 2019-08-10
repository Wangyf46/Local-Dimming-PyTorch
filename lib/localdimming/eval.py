import ipdb
import cv2
import numpy as np
from PIL import Image
from SSIM_PIL import compare_ssim
from scipy.signal import convolve2d

from lib.localdimming.transform import *


def get_PSNR(target, ref):
    MSE = np.mean((target - ref) ** 2)
    RMSE = np.sqrt(MSE)
    RMSE_norm = RMSE / 255.0       # RMSE normalization to [0.0, 1.0]
    PSNR = 20 * np.log10(255 / RMSE)
    return PSNR

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1] # todo
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    kernel /= kernel.sum() # todo
    return kernel


def filter(img, kernel, mode='same'):
    return convolve2d(img, np.rot90(kernel, 2), mode=mode)


def get_SSIM(target, ref):
    # numpy->PIL
    target = Image.fromarray(cv2.cvtColor(target.astype('uint8'), cv2.COLOR_BGR2RGB))
    ref = Image.fromarray(cv2.cvtColor(ref.astype('uint8'), cv2.COLOR_BGR2RGB))
    ssim = compare_ssim(target, ref)
    # target.show()
    # ref.show()
    return ssim


# def get_MI(target, ref):



## RGB-->XYX--->LAB
def get_CD(target, ref):
    B1, G1, R1 = cv2.split(target)
    R1_gamma, G1_gamma, B1_gamma = Gammas(R1/255.0, G1/255.0, B1/255.0)
    X1, Y1, Z1 = rgbToxyz(R1_gamma, G1_gamma, B1_gamma)
    L1, a1, b1 = xyzTolab(X1, Y1, Z1)

    B2, G2, R2 = cv2.split(ref)
    R2_gamma, G2_gamma, B2_gamma = Gammas(R2 / 255.0, G2 / 255.0, B2 / 255.0)
    X2, Y2, Z2 = rgbToxyz(R2_gamma, G2_gamma, B2_gamma)
    L2, a2, b2 = xyzTolab(X2, Y2, Z2)

    CD = np.mean(np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b1-b2)**2))
    return CD



def get_H(Y):
    Y = np.uint8(Y)
    H, W = Y.shape
    p, q = 0.1 * H * W, 0.9 * H * W
    Y_min, Y_max = np.min(Y), np.max(Y)
    total = 0
    for H_10 in range(Y_min, Y_max+1):
        num = np.sum(Y == H_10)
        total += num
        if total >= p:
            break
    for H_90 in range(Y_min, Y_max + 1):
        num = np.sum(Y == H_90)
        total += num
        if total >= q:
            break
    if H_10 == 0:
        H_10 = 1
    return H_10, H_90


def get_CR(Yin, LD, Ycp, Yout):
    ## Iin
    Hin_10, Hin_90 = get_H(Yin)
    CR_in = Hin_90 / (Hin_10)

    ## Icp
    LD = getLD_transform(LD)
    CR_ld = (np.max(LD) / np.min(LD)).astype('float64')
    Hcp_10, Hcp_90 = get_H(Ycp)
    CR_cp = Hcp_90 / (Hcp_10)
    CR_out= CR_ld * CR_cp

    ## Iout
    Hout_10, Hout_90 = get_H(Yout)
    CR_out1 = Hout_90 / (Hout_10)

    return CR_in, CR_cp, CR_ld, CR_out, CR_out1