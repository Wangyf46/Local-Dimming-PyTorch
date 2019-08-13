import os
import ipdb
import cv2
import torch
from torch.utils.data import Dataset

from lib.localdimming.common  import *


class DIV2K(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.PERIOD == 'train':
            self.img_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_train_HR_aug')
            self.name_list = os.listdir(self.img_dir)
        else:
            self.img_dir = os.path.join(self.cfg.DATA_PATH, 'DIV2K_valid_HR_aug')
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
        img_file =  os.path.join(self.img_dir, name)
        Iin = cv2.imread(img_file).astype('float32')                               # numpy-float32-[0.0-255.0]

        Iin_transform = torch.from_numpy(Iin.transpose((2,0,1)) / 255.0)           # torch.float32-[0.0-1.0]
        return  Iin_transform, name


