import cv2
import numpy as np
import ipdb


def getBL_cdf(block):
    block = np.uint8(block)
    h, w = block.shape
    I_max, I_min = np.max(block), np.min(block)
    total, k = 0, 0.9
    threshold = k * h * w
    for BL in range(I_min, I_max+1):
        num = np.sum(block == BL)
        total += num
        if total >= threshold:
            break
    return BL

## TODO
def getBL_imf(block):
    n = 0.9
    I_avg = np.mean(block)
    I_max = np.max(block)
    zoneWeight = n * I_max + (1-n) * I_avg


def getBL_otsu(block):
    block = np.uint8(block)
    I_avg, I_min, I_max = np.mean(block), np.min(block), np.max(block)
    thresh, ret = cv2.threshold(block, 0, 1, cv2.THRESH_OTSU)  # fill(>)-1
    N0 = np.sum(ret == 0.0)
    N1 = np.sum(ret == 1.0)
    K = N1 / (N1 + N0)
    threshold = K * block.shape[0] * block.shape[1]
    if K == 1.0:
        BDG = I_max
    else:
        total = 0
        for BDG in range(I_min, I_max+1):
            num = np.sum(block == BDG)
            total += num
            if total >= threshold:
                break
    BDR = 0.5 + BDG / 510
    BL = BDR * I_max + (1-BDR) * I_avg
    return BL



def getBL_grayEntroy(block):
    block = np.uint8(block)
    I_avg, I_min, I_max = np.mean(block), np.min(block), np.max(block)
    h, w = block.shape
    var = 0.0
    for i in range(I_min, I_max+1):
        num = np.sum(block == i)
        prob = num / (h * w)
        if prob != 0.0:
            var += prob * np.log2(prob)
    H = -1.0 * var / np.log2(h * w)
    K = 1 - H
    BL = I_avg + K * (I_max - I_avg)
    return BL


def getBL_psnr(block):
    block = np.uint8(block)
    h, w = block.shape
    Rpsnr = 30
    Emse = 65.025
    Etse = Emse * h * w
    I_max = np.max(block)

    ## Ic init
    n = 0.7
    # n = 0.8
    # n = 0.9
    gamma = 2.2
    Ic = np.uint8((n ** (1.0 / gamma)) * I_max)

    Etse_m = 0
    while(1):
        if Etse_m <= Etse:
            for level in range(Ic, I_max+1):
                Etse_m += np.sum(block == level) * ((level-Ic)**2)
            Ic = Ic -1
        else:
            break
    BL = ((Ic / 255.0) ** gamma) * 255
    return BL



def getBL_gauss(block):
    h, w = block.shape
    I_avg = np.mean(block)
    delta = np.sum((block - I_avg) ** 2) / (h * w)
    BL = (delta ** 0.5) * 1.5 + I_avg
    return BL