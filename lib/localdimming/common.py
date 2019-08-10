import ipdb

from lib.localdimming.BL import *
from lib.localdimming.LD import *
from lib.localdimming.CP import *
from lib.localdimming.eval import *
from lib.localdimming.transform import *


def getBL(block, means):
    if means == 'max':
        BL = np.max(block)
    elif means == 'avg':
        BL = np.mean(block)
    elif means == 'square root':
        I_avg = np.mean(block)
        BL = 255 * np.sqrt(I_avg / 255.0)
    elif means == 'sd':
        I_avg = np.mean(block)
        sd = np.sqrt(np.mean((block - I_avg) ** 2))
        BL = I_avg + sd
    elif means == 'lut':
        I_max = np.max(block)
        I_avg = np.mean(block)
        diff = I_max - I_avg
        BL = I_avg + 0.50 * (diff + diff ** 2 / 255.0)
    elif means == 'cdf':
        BL = getBL_cdf(block)
    elif means == 'imf':
        BL = getBL_imf(block)           ## TODO
    elif means == 'gauss':
        BL = getBL_gauss(block)         ## TODO
    elif means == 'otsu':
        BL = getBL_otsu(block)
    elif means == 'gray entroy':
        BL = getBL_grayEntroy(block)
    elif means == 'psnr':
        BL = getBL_psnr(block)

    BL = np.where(BL < 0, 0.0, BL) * 1.0
    BL = np.where(BL > 255, 255.0, BL) * 1.0

    return BL


def getLD(BL, gray, means):
    if means == 'bma':
        LD = getLD_bma(BL, gray).astype('float32')
    elif means == 'interpolation':
        LD = getLD_interpolation(BL, gray)
    return LD


def getIcp(Iin, LD, means):
    if means == 'linear':
        Icp = getIcp_linear(Iin, LD)
    elif means == 'unlinear':
        Icp = getIcp_unlinear(Iin, LD)
    elif means == '2steps':
        Icp = getIcp_2steps(Iin, LD)
    elif means == 'log':
        Icp = getIcp_log(Iin, LD)
    return Icp


## TODO: rgb == yuv == Icp
def getIout(Icp, LD, means):
    if means == 'LINEAR':
        Iout = Icp * LD[:, :,np.newaxis] / 255.0
    elif means == 'UNLINEAR':
        gamma = 2.2
        Iout = (Icp / 255.0) ** gamma * LD[:, :,np.newaxis]
    return Iout


def get_eval(Iin, BL, LD, Icp, Iout):
    Bin, Gin, Rin = cv2.split(Iin)
    Yin, _, _ = rgbToyuv(Rin, Gin, Bin)

    Bcp, Gcp, Rcp = cv2.split(Icp)
    Ycp, _, _ = rgbToyuv(Rcp, Gcp, Bcp)

    Bout, Gout, Rout = cv2.split(Iout)
    Yout, _, _ = rgbToyuv(Rout, Gout, Bout)

    psnr = get_PSNR(Iin, Iout)
    ssim = get_SSIM(Iin, Iout)
    # mi = get_MI(Iin, Yout)                  # TODO
    cd = get_CD(Iin, Iout)
    cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_CR(Yin, LD, Ycp, Yout)
    psnr, ssim, cd, cr_in, cr_cp, cr_ld, cr_out, cr_out1 = round(psnr, 2), round(ssim, 2), round(cd, 2),\
                                                           round(cr_in, 2), \
                                                           round(cr_cp, 2), round(cr_ld, 2), round(cr_out, 2), \
                                                           round(cr_out1, 2)
    return psnr, ssim, cd, cr_in, cr_cp, cr_ld, cr_out, cr_out1


def vis_show(Iin, BL, LD, Icp, Iout):
    cv2.namedWindow("Iin", 0)
    cv2.resizeWindow("Iin", 640, 480)
    cv2.imshow('Iin', np.uint8(Iin))

    # cv2.imshow('BL', np.uint8(BL))

    # cv2.imshow('LD', np.uint8(LD))

    cv2.namedWindow("Icp", 0)
    cv2.resizeWindow("Icp", 640, 480)
    cv2.imshow('Icp', np.uint8(Icp))

    cv2.namedWindow("Iout", 0)
    cv2.resizeWindow("Iout", 640, 480)
    cv2.imshow('Iout', np.uint8(Iout))

    cv2.waitKey(0)




