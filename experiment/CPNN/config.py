import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.DATA_NAME = 'DIV2K-aug'
        self.DATA_PATH = '/data/workspace/DIV2K-aug/'

        self.DATA_SIZE = [1080, 1920]
        self.DATE_BLOCK = [9, 16]
        self.DATA_AUG = True

        ## Local-Dimming
        self.BL = 'lut'
        self.LD = 'bma'
        self.CP = 'unet'         ## model name
        self.DISPLAY = 'LINEAR'  ## TODO
        # self.LOSS = 'mse'
        self.LOSS = 'ssim+mse'
        self.LD_TRANSFORMA = False
        self.VIS = False
        self.SAVE = True
        self.EVAL = True

        ## TODO-11.15:
        self.MODEL = 'Unet'
        self.TRAIN_SET = self.DATA_PATH + 'lut-bma/train-LDs/'
        self.TEST_SET = self.DATA_PATH + 'lut-bma/valid-LDs/'
        self.TEST_CKPT = ''
        self.RESULT = ''

        ## TODO-old:
        # self.TEST_CKPT = '~/checkpoints/last_mse.pth'     # mse
        # self.TEST_CKPT = 'checkpoints/last_ssim+mse.pth'  # mse+ssim

        self.DATE = time.strftime("%Y%m%d%_H%M%S_ubuntu", time.localtime())
        self.WORK_DIR = os.path.join('/data/workspace/DIV2K-aug/workout/', self.MODEL, self.DATE)
        self.TRAIN_LR = 0.0001
        self.TRAIN_BZ = 32
        self.TRAIN_EPOCHS = 100
        self.TRAIN_CKPT = ''
        self.TRAIN_LOG_DIR = os.path.join(self.WORK_DIR, 'tf_logs')
        self.TEST_BZ = 1

        self.__check()
        self.__add_path('/home/wyf/codes/dsp/Local-Dimming-PyTorch/')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = Configuration()


