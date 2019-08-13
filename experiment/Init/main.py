import os
import cv2
import numpy as np
import ipdb
from torch.utils.data import DataLoader

from config import cfg
from DIV2K import DIV2K


if __name__ == '__main__':
    print(cfg.EXP)
    listDataset = DIV2K(cfg)
    num = len(listDataset)
    for idx in range(len(listDataset)):
        listDataset[idx]

    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2))

    print(round(listDataset.psnrs / num, 2), round(listDataset.ssims / num, 2),
          round(listDataset.cds / num, 2),
          round(listDataset.cr_ins / num, 2),
          round(listDataset.cr_cps / num, 2), round(listDataset.cr_lds / num, 2),
          round(listDataset.cr_outs / num, 2),
          round(listDataset.cr_out1s / num, 2),
          file=cfg.RECORD_FILE)

