#!/usr/bin/env bash

python ../main.py --dataset_mode=multi_obj \
               --scan_folder=../data/PartScan \
               --train_list=../data/train_list.json \
               --val_list=../data/val_list.json \
               --lr=1e-4 \
               --epochs=1100 \
               --lr_drop=1000 \
               --resume=../checkpoint1099.pth \
               --only_parts_train=True
