import os
from random import shuffle

from lib.localdimming.common  import *


class DIV2K():
    def __init__(self, cfg):
        self.img_dir = os.path.join(cfg.DATA_PATH, 'DIV2K_valid_HR_aug')
        self.name_list = os.listdir(self.img_dir)
        # shuffle(self.name_list)
        self.num_samples = len(self.name_list)
        self.id_list = range(0, self.num_samples)
        self.blob_list = {}
        print('Pre-loading the data. This may take a minutes')
        idx = 0
        for fname in self.name_list:
            Iin_file = os.path.join(self.img_dir, fname)
            Iin = cv2.imread(Iin_file).astype('float64')    ## BGR-HWC-[0.0, 255.0]
            blob = {}
            blob['Iin'] = Iin
            blob['fname'] = fname
            self.blob_list[idx] = blob
            idx += 1
            if idx % 100 == 0:
                print('Loaded', idx, '/', self.num_samples, 'files')
        print('Completed Loading ', idx, 'files')

    def get_num_samples(self):
        return self.num_samples

    def __iter__(self):
        for idx in self.id_list:
            blob = self.blob_list[idx]
            blob['idx'] = idx
            yield blob


