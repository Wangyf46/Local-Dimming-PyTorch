import os
import cv2
import numpy as np
import ipdb

from config import cfg
from DIV2K import DIV2K
from lib.localdimming.common  import *


if __name__ == '__main__':
    Iin_dir = '/home/zengq/U-net_BL/Data_zoo/hdr/images/validation/'
    BL_dir = '/home/zengq/U-net_BL/Data_zoo/hdr/annotations/validation/'
    name_list = os.listdir(Iin_dir)
    psnrs, ssims, cds, cr_ins, cr_cps, cr_lds, cr_outs, cr_out1s = 0, 0, 0, 0, 0, 0, 0, 0
    for fname in name_list:
        Iin_file = os.path.join(Iin_dir, fname)
        Iin = cv2.imread(Iin_file).astype('float64')    ## BGR-HWC-[0.0, 255.0]

        gray = np.zeros((Iin.shape[0], Iin.shape[1]), dtype='float64')
        for i in range(Iin.shape[0]):
            for j in range(Iin.shape[1]):
                gray[i, j] = max(Iin[i, j][0], Iin[i, j][1], Iin[i, j][2])
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)

        '''
        ## From BLnet-zq
        BL_file = os.path.join(BL_dir, fname)
        BL = cv2.imread(BL_file, 0).astype('float64')
        '''
        ## get blocks and corresponding backlight-luminance(BL)
        m = 1080 / 36
        n = 1920 / 66
        BL = np.zeros([36, 66])  # [0.0, 255.0]
        for i in range(36):
            x1 = int(m * i)
            x2 = int(m * (i + 1))
            for j in range(66):
                y1 = int(n * j)
                y2 = int(n * (j + 1))
                block = gray[x1:x2, y1:y2]
                BL[i][j] = getBL(block, means='lut')     ## TODO: LUT
        LD = getLD(BL, gray, 'bma')
        Icp = getIcp(Iin, LD, 'unlinear')
        Iout = getIout(Icp, LD, 'LINEAR')

        if cfg.EVAL:
            psnr, ssim, cd, cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_eval(Iin, BL, LD, Icp, Iout)
            psnrs += psnr
            ssims += ssim
            cds += cd
            cr_ins += cr_in
            cr_cps += cr_cp
            cr_lds += cr_ld
            cr_outs += cr_out
            cr_out1s += cr_out1

            print_str = 'Index: [{0}]  '.format(fname)
            print_str += 'PSNR: {0}  '.format(psnr)
            print_str += 'SSIM: {0}  '.format(ssim)
            print_str += 'CD: {0}\t'.format(cd)
            print_str += 'CR_in: {0}\t'.format(cr_in)
            print_str += 'CR_cp: {0}\t'.format(cr_cp)
            print_str += 'CR_ld: {0}\t'.format(cr_ld)
            print_str += 'CR_out: {0}\t'.format(cr_out)
            print_str += 'CR_out1: {0}\t'.format(cr_out1)
            print(print_str)
            print(print_str, file=cfg.RECORD_FILE)

        if cfg.VIS:
            vis_show(Iin, BL, LD, Icp, Iout)

        if cfg.SAVE:
            LD_name = os.path.join(cfg.LD_SAVE_DIR, fname)
            Icp_name = os.path.join(cfg.Icp_SAVE_DIR, fname)
            Iout_name = os.path.join(cfg.Iout_SAVE_DIR, fname)
            cv2.imwrite(LD_name, LD)
            cv2.imwrite(Icp_name, Icp)
            cv2.imwrite(Iout_name, Iout)


    num = len(name_list)
    print(round(psnrs / num, 2), round(ssims / num, 2),
          round(cds / num, 2),
          round(cr_ins / num, 2),
          round(cr_cps / num, 2), round(cr_lds / num, 2),
          round(cr_outs / num, 2),
          round(cr_out1s / num, 2))

    print(round(psnrs / num, 2), round(ssims / num, 2),
          round(cds / num, 2),
          round(cr_ins / num, 2),
          round(cr_cps / num, 2), round(cr_lds / num, 2),
          round(cr_outs / num, 2),
          round(cr_out1s / num, 2),
          file=cfg.RECORD_FILE)




