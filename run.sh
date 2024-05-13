#!/bin/bash

cuda=7
model="cnn"
dataset="cifar10"
aug=False
# workdir=/home/songk/workdirs/scnn/"$model"_aug_$dataset
# workdir="/home/songk/workdirs/scnn/"$model"_"$dataset
workdir="/home/songk/workdirs/scnn/cnn_noaug_cifar10"

CUDA_VISIBLE_DEVICES=$cuda python -m scnn.train \
    --config=configs/default.py \
    --config.model=$model \
    --config.dataset=$dataset \
    --config.augment_data=$aug \
    --config.num_train_steps=200 \
    --workdir=$workdir \
    --use_wandb=False

# python -m scnn.train \
#     --config=configs/default.py \
#     --config.model=$model \
#     --workdir=$workdir \
#     --use_wandb=False \
#     --eval=True \
#     --step=500