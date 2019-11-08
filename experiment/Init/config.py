import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        self.GPU_ID = '0'

        self.BL = 'lut'
        self.LD = 'bma'
        self.CP = 'unlinear'
        self.DISPLAY = 'LINEAR'       ## TODO
        # self.EXP = self.BL + '-' + self.LD + '-' +self.CP + '-' + self.DISPLAY
        self.EXP = 'Zoo_36x66' + '-' + self.BL  + '-' + self.LD + '-' + self.CP + '-' + self.DISPLAY

        self.LD_TRANSFORMA = False
        self.VIS = False
        self.SAVE =True
        self.EVAL = True

        self.BATCH_SIZE = 1

        self.DATA_NAME = 'DIV2K'
        self.DATA_PATH = '/home/wangyf/datasets/DIV2K'
        self.DATA_SIZE = [1080, 1920]
        self.DATE_BLOCK = [9, 16]
        self.DATA_AUG = True

        self.LD_SAVE_DIR = os.path.join('output', self.EXP, self.DATE, 'LD')
        self.Icp_SAVE_DIR = os.path.join('output', self.EXP, self.DATE, 'Icp')
        self.Iout_SAVE_DIR = os.path.join('output', self.EXP, self.DATE, 'Iout')

        self.__check()
        self.__add_path('/home/wangyf/codes/Local-Dimming-PyTorch')

        self.RECORD_FILE = open(os.path.join('output', self.EXP, self.DATE, 'record.txt'), 'w')


    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')
        if not os.path.isdir(self.LD_SAVE_DIR):
            os.makedirs((self.LD_SAVE_DIR))
        if not os.path.isdir(self.Icp_SAVE_DIR):
            os.makedirs((self.Icp_SAVE_DIR))
        if not os.path.isdir(self.Iout_SAVE_DIR):
            os.makedirs((self.Iout_SAVE_DIR))

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = Configuration()


