import os
import ipdb
import torch
from torch.utils.data import DataLoader

from config import cfg
from DIV2K import DIV2K
from lib.utils import *
from lib.localdimming.eval import *


def test_net(cfg, net):
    listDataset = DIV2K(cfg)

    test_loader = DataLoader(listDataset,
                              batch_size=cfg.TEST_BZ,
                              pin_memory=True)

    net.eval()
    num = len(listDataset)
    psnrs, ssims, cds, cr_ins, cr_cps, cr_lds, cr_outs, cr_out1s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i_batch, (Iin_transform, Icp_transform, BL_transform, name) in enumerate(test_loader):
        BL = BL_transform.unsqueeze(1).cuda()
        LD = net(BL)
        Icp = Icp_transform.cuda()
        Iout = Icp * LD / 255.0

        Iin_cpu = Iin_transform.squeeze(0).numpy().transpose((1, 2, 0))
        BL_cpu = BL_transform.squeeze(0)
        LD_cpu = LD.detach().squeeze(0).squeeze(0).cpu().numpy() * 255
        Icp_cpu = Icp_transform.squeeze(0).numpy().transpose((1, 2, 0))
        Iout_cpu = Iout.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255

        if cfg.SAVE:
            LD_name = os.path.join(cfg.LD_SAVE_DIR, name[0])
            Iout_name = os.path.join(cfg.Iout_SAVE_DIR, name[0])
            cv2.imwrite(LD_name, np.uint8(LD_cpu))
            cv2.imwrite(Iout_name, np.uint8(Iout_cpu))

        if cfg.EVAL:
            psnr = get_PSNR(Iin_cpu, Iout_cpu)
            ssim = get_SSIM(Iin_cpu, Iout_cpu)                                      # 0.0-1.0
            cd = get_CD(Iin_cpu, Iout_cpu)                                          # 0.0
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

    print(round(psnrs/num, 2), round(ssims/num, 2),  round(cds/num, 2),
          round(cr_ins/num, 2), round(cr_cps/num, 2), round(cr_lds/num, 2), round(cr_outs/num, 2), round(cr_out1s/num, 2))
    print(round(psnrs/num, 2), round(ssims/num, 2),  round(cds/num, 2),
          round(cr_ins/num, 2), round(cr_cps/num, 2), round(cr_lds/num, 2), round(cr_outs/num, 2), round(cr_out1s/num, 2),
          file=cfg.TEST_RECORD_FILE)


if __name__ == '__main__':
    # from lib.net.unet_up import UNet_Upsampling
    # net = UNet_Upsampling(1, 1)

    from lib.net.edsr import EDSR
    net = EDSR(1, 1)

    if cfg.TEST_CKPT:
        net.load_state_dict(torch.load(cfg.TEST_CKPT))
        print('Model loaded from {}'.format(cfg.TEST_CKPT))
    if cfg.GPU:
        net.cuda()

    test_net(cfg, net)