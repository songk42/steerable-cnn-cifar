#!/bin/bash

cuda=5
model="autoencoder"
dataset="cifar10"
# dataset="caltech101"
aug=False
add_noise=True
workdir="/home/songk/workdirs/scnn/"$model"_"$dataset
if [ $aug = "True" ]; then
    workdir="/home/songk/workdirs/scnn/"$model"_aug_"$dataset
fi

CUDA_VISIBLE_DEVICES=$cuda python -m scnn.train \
    --config=configs/default.py \
    --config.model=$model \
    --config.dataset=$dataset \
    --config.augment_data=$aug \
    --config.add_noise=$add_noise \
    --config.eval_every_steps=20 \
    --config.num_train_steps=100 \
    --config.log_every_steps=5 \
    --workdir=$workdir \
    --use_wandb=False

# python -m scnn.train \
#     --config=configs/default.py \
#     --config.model=$model \
#     --workdir=$workdir \
#     --use_wandb=False \
#     --eval=True \
#     --step=500