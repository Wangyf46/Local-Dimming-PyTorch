#!/usr/bin/env bash

## TODO:
srcpath='/data/workspace/DIV2K-aug/DIV2K_train_HR/'
LDpath='/data/workspace/DIV2K-aug/lut-bma/train-LDs/'

python preprocess.py $srcpath $LDpath
