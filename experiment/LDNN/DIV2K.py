import os
import cv2
import ipdb
import torch
from torch.utils.data import Dataset

from lib.localdimming.common  import *


class DIV2K(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.PERIOD == 'train':
            self.Iin_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_train_HR_aug')
            self.Icp_dir = os.path.join('/home/wangyf/codes/LDNN/output/', '2019-08-06-12-54/Icp_train')    ## TODO
            self.name_list = os.listdir(self.Iin_dir)

        else:
            self.Iin_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_valid_HR_aug')
            self.Icp_dir = os.path.join('/home/wangyf/codes/LDNN/output/', '2019-08-04-15-26/Icp')          ## TODO
            self.name_list = os.listdir(self.Iin_dir)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        Iin_file =  os.path.join(self.Iin_dir, name)
        Icp_file = os.path.join(self.Icp_dir, name)

        Iin = cv2.imread(Iin_file).astype('float32')                               # numpy-float32-[0.0-255.0]
        Icp = cv2.imread(Icp_file).astype('float32')                               # numpy-float32-[0.0-255.0]

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
                BL[i][j] = getBL(block, means=self.cfg.BL)  # numpy-float32-[0.0-255.0]

        Iin_transform = torch.from_numpy(Iin.transpose((2, 0, 1)))   # torch.float32-[0.0-255.0]
        Icp_transform = torch.from_numpy(Icp.transpose((2, 0, 1)))   # torch.float32-[0.0-255.0]
        BL_transform = torch.from_numpy(BL / 255.0)                                     # torch.float32-[0.0-1.0]

        return Iin_transform, Icp_transform, BL_transform, name


