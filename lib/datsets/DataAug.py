import os
import sys
import cv2
import ipdb
import random
import numpy as np
from PIL import Image


if __name__ == '__main__':
    ROOT_DIR = '/home/wangyf/datasets'
    img_dir = os.path.join(ROOT_DIR, 'DIV2K_valid_HR')
    name_list = os.listdir(img_dir)

    new_dir = '/home/wangyf/datasets/DIV2K_valid_HR_aug/'
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    for idx in range(len(name_list)):
        name = name_list[idx]
        img_file = os.path.join(img_dir, name)
        img_RGB = Image.open(img_file)
        img_1080x1920 = img_RGB.resize((1920, 1080), Image.ANTIALIAS)
        img_1080x1920.save(os.path.join(new_dir, name), 'PNG')
        # img_1080x1920_h = img_1080x1920.transpose(Image.FLIP_LEFT_RIGHT)
        # img_1080x1920_h.save(os.path.join(new_dir, name.split('.')[0] + '-h.png'), 'PNG')
        # img_1080x1920_v = img_1080x1920.transpose(Image.FLIP_TOP_BOTTOM)
        # img_1080x1920_v.save(os.path.join(new_dir, name.split('.')[0] + '-v.png'), 'PNG')
        print(name, idx)
