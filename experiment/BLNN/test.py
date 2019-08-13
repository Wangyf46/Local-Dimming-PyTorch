import os
import time
import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader


from config import cfg
from DIV2K import DIV2K
from lib.loss.loss import *
from lib.utils import *
from lib.localdimming.common import *



def test_net(cfg, net):

    listDataset = DIV2K(cfg)

    test_loader = DataLoader(listDataset,
                              batch_size=cfg.TEST_BZ,
                              pin_memory=True)

    net.eval()
    psnrs, ssims, cds, cr_ins, cr_cps, cr_lds, cr_outs, cr_out1s = 0, 0, 0, 0, 0, 0, 0, 0
    num = len(listDataset)
    for i_batch, (Iin_transform, name) in enumerate(test_loader):
        Iin = Iin_transform.cuda()                                              # torch.float32, [0.0-1.0]
        BL = net(Iin)                                                           # torch.float32, [0.0-1.0]

        Iin_cpu = Iin_transform.squeeze(0).numpy().transpose((1, 2, 0)) * 255
        BL_cpu = BL.squeeze(0).squeeze(0) * 255
        LD = getLD(BL_cpu, np.zeros((cfg.DATA_SIZE)), cfg.LD)
        Icp = getIcp(Iin_cpu, LD, cfg.CP)  # HWC
        Iout = getIout(Icp, LD, cfg.DISPLAY)                                             # HWC

        if cfg.SAVE:
            Icp_name = os.path.join(cfg.Icp_SAVE_DIR, name[0])
            Iout_name = os.path.join(cfg.Iout_SAVE_DIR, name[0])

            cv2.imwrite(Icp_name, np.uint8(Icp))
            cv2.imwrite(Iout_name, np.uint8(Iout))

            if cfg.EVAL:
                psnr = get_PSNR(Iin_cpu, Iout)
                ssim = get_SSIM(Iin_cpu, Iout)
                cd = get_CD(Iin_cpu, Iout)
                cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_CR(Iin_cpu, LD, Icp, Iout)

                psnrs += psnr
                ssims += ssim
                cds += cd
                cr_ins += cr_in
                cr_cps += cr_cp
                cr_lds += cr_ld
                cr_outs += cr_out
                cr_out1s += cr_out1

                print_str = 'Index: [{0}]\t'.format(name[0])
                print_str += 'PSNR: {0}\t'.format(psnr)
                print_str += 'SSIM: {0}\t'.format(ssim)
                print_str += 'CD: {0}\t'.format(cd)
                print_str += 'CR_in: {0}\t'.format(cr_in)
                print_str += 'CR_cp: {0}\t'.format(cr_cp)
                print_str += 'CR_ld: {0}\t'.format(cr_ld)
                print_str += 'CR_out: {0}\t'.format(cr_out)
                print_str += 'CR_out1: {0}\t'.format(cr_out1)

                ## 打印到文件
                log_print(print_str, cfg.TEST_RECORD_FILE, color="blue", attrs=["bold"])

        print(psnrs / num, ssims / num, cds / num, cr_ins / num, cr_cps / num, cr_lds / num, cr_outs / num,
              cr_out1s / num)
        print(psnrs / num, ssims / num, cds / num, cr_ins / num, cr_cps / num, cr_lds / num, cr_outs / num,
              cr_out1s / num,
              file=cfg.TEST_RECORD_FILE)


if __name__ == '__main__':

    from lib.net.unet_down import UNet_Downsampling
    net = UNet_Downsampling(3, 1)

    if cfg.TEST_CKPT:
        net.load_state_dict(torch.load(cfg.TEST_CKPT))
        print('Model loaded from {}'.format(cfg.TEST_CKPT))
    if cfg.GPU:
        net.cuda()

    test_net(cfg, net)