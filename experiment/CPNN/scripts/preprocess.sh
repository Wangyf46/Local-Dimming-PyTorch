#!/usr/bin/env bash

## TODO:
srcpath='/home/wangyf/datasets/DIV2K/DIV2K_valid_HR_aug/'
LDpath='/home/wangyf/codes/Local-Dimming-PyTorch/CPNN/DIV2K/lut-bma-vaild/LDs/'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../tools/prepocess.py $srcpath $LDpath



