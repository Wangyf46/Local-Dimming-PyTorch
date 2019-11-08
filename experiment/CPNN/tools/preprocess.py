#!/usr/bin/python

import os
import pdb
import cv2
import shutil
import argparse
import numpy as np

from config import cfg
from lib.localdimming.common  import *

def parse_args():
	parser = argparse.ArgumentParser('Local Dimming Algorithm include BLs and LDs')
	parser.add_argument('srcpath', help='include all rgb image files.')
	parser.add_argument('LDpath', help='save LD files.')
	return parser.parse_args()


def main(args):
	name_list = os.listdir(args.srcpath)
	for idx in range(len(name_list)):
		fname = os.path.join(args.srcpath, name_list[idx])

		## Iin(BGR): float32-[0.0-255.0]
		Iin = cv2.imread(fname).astype(np.float32)

		## gray: maximum grayscale processing
		gray = np.zeros(cfg.DATA_SIZE, dtype='float32')
		for i in range(cfg.DATA_SIZE[0]):
			for j in range(cfg.DATA_SIZE[1]):
				gray[i, j] = max(Iin[i, j][0], Iin[i, j][1], Iin[i, j][2])

		## get blocks and corresponding backlight-luminance(BL)
		m = cfg.DATA_SIZE[0] / cfg.DATE_BLOCK[0]
		n = cfg.DATA_SIZE[1] / cfg.DATE_BLOCK[1]
		BL = np.zeros(cfg.DATE_BLOCK, dtype='float32')
		for i in range(cfg.DATE_BLOCK[0]):
			x1 = int(m * i)
			x2 = int(m * (i + 1))
			for j in range(cfg.DATE_BLOCK[1]):
				y1 = int(n * j)
				y2 = int(n * (j + 1))
				block = gray[x1:x2, y1:y2]
				BL[i][j] = getBL(block, means=cfg.BL)  # numpy-float32-[0.0-255.0]

		LD = getLD(BL, gray, cfg.LD)  # numpy-float32-[0.0-255.0]

		if cfg.LD_TRANSFORMA:
			LD = getLD_transform(LD)
		LD_dir = args.LDpath + fname
		cv2.imwrite(LD_dir, LD)



if __name__ == '__main__':
	args = parse_args()
	main(args)