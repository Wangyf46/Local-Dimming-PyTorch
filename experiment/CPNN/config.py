import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        self.GPU = True
        self.GPU_ID = '1'
        self.PERIOD = 'test'

        self.BL = 'lut'
        self.LD = 'bma'
        self.CP = 'unet'         ## model name
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
        # self.TEST_CKPT = '~/checkpoints/last_mse.pth'     # mse
        self.TEST_CKPT = 'checkpoints/last_ssim+mse.pth'  # mse+ssim

        self.LD_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'LD')
        self.Icp_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'Icp')
        self.Iout_SAVE_DIR = os.path.join(self.EXP, 'output', self.DATE, 'Iout')

        self.__check()
        self.__add_path('/home/wangyf/codes/Local-Dimming-PyTorch/')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = Configuration()


