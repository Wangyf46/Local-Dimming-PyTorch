import os
import sys
import time
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#### 

from config import cfg
from DIV2K import DIV2K
from lib.net.unet import UNet
from lib.loss.loss import *
from lib.utils import *


def train_net(cfg, net):
    if cfg.TRAIN_TBLOG:
        tblogger = SummaryWriter(cfg.TRAIN_LOG_DIR)

    listDataset = DIV2K(cfg)      ## TODO

    train_loader = DataLoader(listDataset,
                              batch_size=cfg.TRAIN_BZ,
                              shuffle=True,
                              pin_memory=True)

    optimizer = optim.Adam(net.parameters(),
                           lr=cfg.TRAIN_LR,
                           betas=(0.9, 0.999),
                           eps=1e-08)

    if cfg.LOSS == 'mse':
        criterion = nn.MSELoss(size_average=True)
    elif cfg.LOSS == 'ssim+mse':
        criterion = Loss(3).cuda()              ## TODO

    itr = 0
    max_itr = cfg.TRAIN_EPOCHS * len(train_loader)
    print(itr, max_itr, len(train_loader))
    net.train()
    for epoch in range(cfg.TRAIN_EPOCHS):
        print('Starting epoch {}/{}.'.format(epoch + 1, cfg.TRAIN_EPOCHS))

        data_time = AverageMeter()    ##TODO
        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        for i_batch, (Iin, LD, name) in enumerate(train_loader):
            data_time.update(time.time() - end)                     # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, epoch, cfg.TRAIN_LR)
            Iin = Iin.cuda()                                        # torch.float32, [0.0-1.0]
            LD = LD.cuda()                                          # torch.float32-[0.0-255.0]
            Icp = net(Iin)                                          # torch.float32-[0.0-1.0]            TODO

            Iout = Icp * LD.unsqueeze(1) / 255.0                # torch.float32-[0.0-1.0]            TODO

            loss = criterion(Iout, Iin)

            losses.update(loss.item(), cfg.TRAIN_BZ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            print_str = 'Epoch: [{0}/{1}]\t'.format(epoch, cfg.TRAIN_EPOCHS)
            print_str += 'Batch: [{0}]/{1}\t'.format(i_batch + 1, listDataset.__len__() // cfg.TRAIN_BZ)
            print_str += 'LR: {0}\t'.format(now_lr)
            print_str += 'Data time {data_time.cur:.3f}({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_str += 'Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            print_str += 'Loss {loss.cur:.4f}({loss.avg:.4f})\t'.format(loss=losses)
            log_print(print_str, cfg.TRAIN_RECORD_FILE, color="green", attrs=["bold"])

            ## torch.float32-CHW-[0.0-1.0]
            tblogger.add_scalar('loss', losses.avg, itr)
            tblogger.add_scalar('lr', now_lr, itr)

            end = time.time()
            itr += 1
        save_path = os.path.join(cfg.TRAIN_CKPT_DIR, '%s_itr%d.pth' % (epoch, itr))
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

    ## 固定随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net = UNet(3, 3)
    # net = nn.DataParallel(net)

    if cfg.PERIOD == 'train':
        if not os.path.isdir(cfg.TRAIN_LOG_DIR):
            os.makedirs((cfg.TRAIN_LOG_DIR))
        if not os.path.isdir(cfg.TRAIN_CKPT_DIR):
            os.makedirs((cfg.TRAIN_CKPT_DIR))
        cfg.TRAIN_RECORD_FILE = open(os.path.join(cfg.EXP, cfg.DATE, 'log', 'record.txt'), 'w')

    if cfg.TRAIN_CKPT:
        net.load_state_dict(torch.load(cfg.TRAIN_CKPT))
        print('Model loaded from {}'.format(cfg.TRAIN_CKPT))
    if cfg.GPU:
        net.cuda()
    try:
        train_net(cfg, net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), cfg.EXP + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)