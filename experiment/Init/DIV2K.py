import os
import ipdb
import torch
from torch.utils.data import Dataset
from random import shuffle

from lib.localdimming.common  import *


class DIV2K(Dataset):
    def __init__(self, cfg):
        self.img_dir = os.path.join(cfg.DATA_PATH, 'DIV2K_valid_HR_aug')
        self.name_list = os.listdir(self.img_dir)
        shuffle(self.name_list)
        self.cfg = cfg
        self.psnrs = 0
        self.ssims = 0
        self.cds = 0
        self.cr_ins = 0
        self.cr_cps = 0
        self.cr_lds = 0
        self.cr_outs = 0
        self.cr_out1s = 0

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file =  os.path.join(self.img_dir, name)

        ## Iin(BGR): float32-[0.0-255.0]
        Iin = cv2.imread(img_file).astype('float32')                 #

        ## gray: maximum grayscale processing
        gray = np.zeros(self.cfg.DATA_SIZE, dtype='float32')
        for i in range(self.cfg.DATA_SIZE[0]):
            for j in range(self.cfg.DATA_SIZE[1]):
                gray[i, j] = max(Iin[i, j][0], Iin[i, j][1], Iin[i, j][2])

        ## get blocks and corresponding backlight-luminance(BL)
        m = self.cfg.DATA_SIZE[0] / self.cfg.DATE_BLOCK[0]
        n = self.cfg.DATA_SIZE[1] / self.cfg.DATE_BLOCK[1]
        BL = np.zeros(self.cfg.DATE_BLOCK, dtype='float32')
        for i in range(self.cfg.DATE_BLOCK[0]):
            x1 = int(m * i)
            x2 = int(m * (i + 1))
            for j in range(self.cfg.DATE_BLOCK[1]):
                y1 = int(n * j)
                y2 = int(n * (j + 1))
                block = gray[x1:x2, y1:y2]
                BL[i][j] = getBL(block, means=self.cfg.BL)                # numpy-float32-[0.0-255.0]

        LD = getLD(BL, gray, self.cfg.LD)                                 # numpy-float32-[0.0-255.0]
        if self.cfg.LD_TRANSFORMA:
            LD = getLD_transform(LD)
        Icp = getIcp(Iin, LD, self.cfg.CP)                                # numpy-float32-[0.0-255.0]
        Iout = getIout(Icp, LD, self.cfg.DISPLAY)                         # numpy-float32-[0.0-255.0]

        if self.cfg.EVAL:
            psnr, ssim, cd, cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_eval(Iin, BL, LD, Icp, Iout)
            self.psnrs += psnr
            self.ssims += ssim
            self.cds += cd
            self.cr_ins += cr_in
            self.cr_cps += cr_cp
            self.cr_lds += cr_ld
            self.cr_outs += cr_out
            self.cr_out1s += cr_out1

            print_str = 'Index: [{0}]  '.format(name)
            print_str += 'PSNR: {0}  '.format(psnr)
            print_str += 'SSIM: {0}  '.format(ssim)
            print_str += 'CR_in: {0}\t'.format(cr_in)
            print_str += 'CR_cp: {0}\t'.format(cr_cp)
            print_str += 'CR_ld: {0}\t'.format(cr_ld)
            print_str += 'CR_out: {0}\t'.format(cr_out)
            print_str += 'CR_out1: {0}\t'.format(cr_out1)
            print(print_str)
            print(print_str, file=self.cfg.RECORD_FILE)

        if self.cfg.VIS:
            vis_show(Iin, BL, LD, Icp, Iout)

        if self.cfg.SAVE:
            LD_name = os.path.join(self.cfg.LD_SAVE_DIR, name)
            Icp_name = os.path.join(self.cfg.Icp_SAVE_DIR, name)
            Iout_name = os.path.join(self.cfg.Iout_SAVE_DIR, name)
            cv2.imwrite(LD_name, LD)
            cv2.imwrite(Icp_name, Icp)
            cv2.imwrite(Iout_name, Iout)

        return name





