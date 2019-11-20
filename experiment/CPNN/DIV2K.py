import os
import pdb
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


from lib.localdimming.common  import *


class DIV2K(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.PERIOD == 'train':
            self.img_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_train_HR_aug')
            self.LD_dri = self.cfg.TRAIN_SET
            self.name_list = os.listdir(self.img_dir)
        else:
            self.img_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_valid_HR_aug')
            self.LD_dri = self.cfg.TEST_SET
            self.name_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        '''
         numpy image: H X W X C
         PIL image: C X H X W
         torch image: C X H X W
         img_PIL = transforms.ToPILImage()(img).convert('RGB')
        '''
        name = self.name_list[idx]
        Iin_path = os.path.join(self.img_dir, name)
        Iin = cv2.imread(Iin_path).astype('float32')

        LD_path = os.path.join(self.LD_dri, name)
        LD = cv2.imread(LD_path).astype('float32')

        Iin_transform = torch.from_numpy(Iin.transpose((2, 0, 1)) / 255.0)  # torch.float32-[0.0-1.0]
        LD_transform = torch.from_numpy(LD)                               # torch.float32-[0.0-255.0]
        return Iin_transform, LD_transform, name



        # name = self.name_list[idx]
        # img_file =  os.path.join(self.img_dir, name)
        #
        # ## Iin(BGR): float32-[0.0-255.0]
        # Iin = cv2.imread(img_file).astype('float32')
        #
        # ## gray: maximum grayscale processing
        # gray = np.zeros(self.cfg.DATA_SIZE, dtype='float32')
        # for i in range(self.cfg.DATA_SIZE[0]):
        #     for j in range(self.cfg.DATA_SIZE[1]):
        #         gray[i, j] = max(Iin[i, j][0], Iin[i, j][1], Iin[i, j][2])
        #
        # ## get blocks and corresponding backlight-luminance(BL)
        # m = self.cfg.DATA_SIZE[0] / self.cfg.DATE_BLOCK[0]
        # n = self.cfg.DATA_SIZE[1] / self.cfg.DATE_BLOCK[1]
        # BL = np.zeros(self.cfg.DATE_BLOCK, dtype='float32')
        # for i in range(self.cfg.DATE_BLOCK[0]):
        #     x1 = int(m * i)
        #     x2 = int(m * (i + 1))
        #     for j in range(self.cfg.DATE_BLOCK[1]):
        #         y1 = int(n * j)
        #         y2 = int(n * (j + 1))
        #         block = gray[x1:x2, y1:y2]
        #         BL[i][j] = getBL(block, means=self.cfg.BL)            # numpy-float32-[0.0-255.0]
        #
        # LD = getLD(BL, gray, self.cfg.LD)                             # numpy-float32-[0.0-255.0]
        # if self.cfg.LD_TRANSFORMA:
        #     LD = getLD_transform(LD)
        #
        # img_transform = torch.from_numpy(Iin.transpose((2,0,1)) / 255.0) # torch.float32-[0.0-1.0]
        # LD_transform = torch.from_numpy(LD)                              # torch.float32-[0.0-255.0]
        # return  img_transform, LD_transform, name


