#!/bin/bash

workdir="/home/songk/workdirs/scnn/data_aug"
python -m scnn.train --config=configs/default.py --workdir=$workdir --use_wandb=False