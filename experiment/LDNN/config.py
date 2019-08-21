import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        self.GPU = True
        self.GPU_ID = '2'
        self.PERIOD = 'train'

        self.BL = 'lut'
        self.LD = 'unet_up'
        self.CP = 'unet'
        self.DISPLAY = 'LINEAR'  ## TODO
        # self.LOSS = 'mse'
        self.LOSS = 'ssim+mse'
        self.EXP = self.BL + '-' + self.LD + '-' + self.CP + '-' + self.DISPLAY + '-' + self.LOSS

        self.LD_TRANSFORMA = False
        self.VIS = False
        self.SAVE = True
        self.EVAL = True

        self.DATA_NAME = 'DIV2K'
        self.DATA_PATH = '/home/wangyf/datasets/DIV2K'
        self.DATA_SIZE = [1080, 1920]
        self.DATE_BLOCK = [9, 16]
        self.DATA_AUG = True

        self.TRAIN_LR = 0.0001
        self.TRAIN_BZ = 2
        self.TRAIN_EPOCHS = 50
        self.TRAIN_TBLOG = True
        self.TRAIN_CKPT = ''
        self.TRAIN_LOG_DIR = os.path.join(self.EXP, self.DATE, 'log')
        self.TRAIN_CKPT_DIR = os.path.join(self.EXP, self.DATE, 'checkpoints')

        self.TEST_BZ = 1
        # self.TEST_CKPT ='~/checkpoints/unet_up/last_mse.pth',                     # mse
        # self.TEST_CKPT = '~/checkpoints/unet_up/last_ssim+mse.pth'                # ssim
        self.TEST_CKPT = '~/checkpoints/edsr/last_ssim+mse.pth'                   # ssim+mse


        self.LD_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'LD')
        self.Icp_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'Icp')
        self.Iout_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'Iout')

        self.__check()
        self.__add_path('/home/wangyf/codes/Local-Dimming-PyTorch')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')
        if self.PERIOD == 'train':
            if not os.path.isdir(self.TRAIN_LOG_DIR):
                os.makedirs((self.TRAIN_LOG_DIR))
            if not os.path.isdir(self.TRAIN_CKPT_DIR):
                os.makedirs((self.TRAIN_CKPT_DIR))
            self.TRAIN_RECORD_FILE = open(os.path.join(self.EXP, self.DATE, 'log', 'record.txt'), 'w')
        else:
            if not os.path.isdir(self.LD_SAVE_DIR):
                os.makedirs((self.LD_SAVE_DIR))
            if not os.path.isdir(self.Icp_SAVE_DIR):
                os.makedirs((self.Icp_SAVE_DIR))
            if not os.path.isdir(self.Iout_SAVE_DIR):
                os.makedirs((self.Iout_SAVE_DIR))
            self.TEST_RECORD_FILE = open(os.path.join(self.EXP, self.DATE, 'output', 'record.txt'), 'w')

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = Configuration()


