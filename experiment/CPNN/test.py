import os
import time
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg
from DIV2K import DIV2K
from lib.net.unet import UNet
from lib.utils import *
from lib.localdimming.eval import *

def test_net(cfg, net):

    listDataset = DIV2K(cfg)

    test_loader = DataLoader(listDataset,
                              batch_size=cfg.TEST_BZ,
                              pin_memory=True)

    net.eval()
    psnrs, ssims, cds, cr_ins, cr_cps, cr_lds, cr_outs, cr_out1s = 0, 0, 0, 0, 0, 0, 0, 0
    num = len(listDataset)
    for i_batch, (Iin, LD, name) in enumerate(test_loader):

        Iin_gpu = Iin.cuda()                                # torch.float32, [0.0-1.0]
        LD_gpu = LD.cuda()                                  # torch.float32-[0.0-255.0]
        Icp_gpu = net(Iin_gpu)                              # torch.float32-[0.0-1.0]

        Iout_gpu = Icp_gpu * LD_gpu.unsqueeze(1) / 255.0    # torch.float32-[0.0-1.0]

        Iin_cpu = Iin.squeeze(0).numpy().transpose((1, 2, 0)) * 255
        LD_cpu = LD.squeeze(0).numpy()
        Icp_cpu = Icp_gpu.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255
        Iout_cpu = Iout_gpu.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255


        if cfg.SAVE:
            Icp_name = os.path.join(cfg.Icp_SAVE_DIR, name[0])
            Iout_name = os.path.join(cfg.Iout_SAVE_DIR, name[0])
            cv2.imwrite(Icp_name, np.uint8(Icp_cpu))
            cv2.imwrite(Iout_name, np.uint8(Iout_cpu))

        if cfg.EVAL:
            psnr = get_PSNR(Iin_cpu, Iout_cpu)
            ssim = get_SSIM(Iin_cpu, Iout_cpu)
            cd = get_CD(Iin_cpu, Iout_cpu)
            cr_in, cr_cp, cr_ld, cr_out, cr_out1 = get_CR(Iin_cpu, LD_cpu, Icp_cpu, Iout_cpu)

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

    print(round(psnrs / num, 2), round(ssims / num, 2), round(cds / num, 2),
          round(cr_ins / num, 2), round(cr_cps / num, 2), round(cr_lds / num, 2), round(cr_outs / num, 2),
          round(cr_out1s / num, 2))
    print(round(psnrs / num, 2), round(ssims / num, 2), round(cds / num, 2),
          round(cr_ins / num, 2), round(cr_cps / num, 2), round(cr_lds / num, 2), round(cr_outs / num, 2),
          round(cr_out1s / num, 2),
          file=cfg.TEST_RECORD_FILE)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

    net = UNet(3, 3)

    if cfg.TEST_CKPT:
        net.load_state_dict(torch.load(cfg.TEST_CKPT))
        print('Model loaded from {}'.format(cfg.TEST_CKPT))
        print(torch.load(cfg.TEST_CKPT))
        ipdb.set_trace()
    if cfg.GPU:
        net.cuda()



    test_net(cfg, net)
