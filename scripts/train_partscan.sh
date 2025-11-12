#!/usr/bin/env bash

python main.py --dataset_mode=multi_obj \
               --scan_folder=/PartScan \
               --train_list=data/train_list_new.json \
               --val_list=data/val_list_new.json \
               --lr=1e-4 \
               --epochs=1100 \
               --lr_drop=1000 \
               --resume=./checkpoint1099.pth \
               --only_parts_train=True
