#!/bin/bash

cuda=6
model="gcnn"
# workdir=/home/songk/workdirs/scnn/"$model"_aug_flip
workdir="/home/songk/workdirs/scnn/"$model

CUDA_VISIBLE_DEVICES=$cuda python -m scnn.train \
    --config=configs/default.py \
    --config.model=$model \
    --workdir=$workdir \
    --use_wandb=False
    # --eval=True \
    # --step=300