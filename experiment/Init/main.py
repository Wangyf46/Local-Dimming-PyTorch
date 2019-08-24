import os
import numpy as np
import ipdb

from config import cfg
from DIV2K import DIV2K
from lib.localdimming.common  import *


if __name__ == '__main__':
    print(cfg.EXP)
    listDataset = DIV2K(cfg)
    psnrs, ssims, cds, cr_ins, cr_cps, cr_lds, cr_outs, cr_out1s = 0, 0, 0, 0, 0, 0, 0, 0
    for blob in listDataset:
        ## gray: maximum grayscale processing
        gray = np.zeros(cfg.DATA_SIZE, dtype='float64')
        for i in range(cfg.DATA_SIZE[0]):
            for j in range(cfg.DATA_SIZE[1]):
                gray[i, j] = max(blob['Iin'][i, j][0], blob['Iin'][i, j][1], blob['Iin'][i, j][2])

        ## get blocks and corresponding backlight-luminance(BL)
        m = cfg.DATA_SIZE[0] / cfg.DATE_BLOCK[0]
        n = cfg.DATA_SIZE[1] / cfg.DATE_BLOCK[1]
        BL = np.zeros(cfg.DATE_BLOCK, dtype='float64')  # [0.0, 255.0]
        for i in range(cfg.DATE_BLOCK[0]):
            x1 = int(m * i)
            x2 = int(m * (i + 1))
            for j in range(cfg.DATE_BLOCK[1]):
                y1 = int(n * j)
                y2 = int(n * (j + 1))
                block = gray[x1:x2, y1:y2]
                BL[i][j] = getBL(block, means=cfg.BL)
        LD = getLD(BL, gray, cfg.LD)  # [0.0, 255.0]
        if cfg.LD_TRANSFORMA:
            LD = getLD_transform(LD)
        Icp = getIcp(blob['Iin'], LD, cfg.CP)  # [0.0, 255.0]
        Iout = getIout(Icp, LD, cfg.DISPLAY)   # [0.0, 255.0]

        if cfg.EVAL:
            psnr, ssim, cd, cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_eval(blob['Iin'], BL, LD, Icp, Iout)
            psnrs += psnr
            ssims += ssim
            cds += cd
            cr_ins += cr_in
            cr_cps += cr_cp
            cr_lds += cr_ld
            cr_outs += cr_out
            cr_out1s += cr_out1

            print_str = 'Index: [{0}]  '.format(blob['fname'])
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
            vis_show(blob['Iin'], BL, LD, Icp, Iout)

        if cfg.SAVE:
            LD_name = os.path.join(cfg.LD_SAVE_DIR, blob['fname'])
            Icp_name = os.path.join(cfg.Icp_SAVE_DIR, blob['fname'])
            Iout_name = os.path.join(cfg.Iout_SAVE_DIR, blob['fname'])
            cv2.imwrite(LD_name, LD)
            cv2.imwrite(Icp_name, Icp)
            cv2.imwrite(Iout_name, Iout)

    num = listDataset.get_num_samples()
    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2))

    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2),
          file=cfg.RECORD_FILE)




    '''
    num = len(listDataset)
    for idx in range(len(listDataset)):
        listDataset[idx]

    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2))

    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2),
          file=cfg.RECORD_FILE)
    '''

