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

    for idx in range(len(listDataset)):
        listDataset[idx]

    print(listDataset.psnrs / len(listDataset), listDataset.ssims / len(listDataset),
          listDataset.cds / len(listDataset),
          listDataset.cr_ins / len(listDataset),
          listDataset.cr_cps / len(listDataset), listDataset.cr_lds / len(listDataset),
          listDataset.cr_outs / len(listDataset),
          listDataset.cr_out1s / len(listDataset))

    print(listDataset.psnrs / len(listDataset), listDataset.ssims / len(listDataset),
          listDataset.cds / len(listDataset),
          listDataset.cr_ins / len(listDataset),
          listDataset.cr_cps / len(listDataset), listDataset.cr_lds / len(listDataset),
          listDataset.cr_outs / len(listDataset),
          listDataset.cr_out1s / len(listDataset),
          file=cfg.RECORD_FILE)

