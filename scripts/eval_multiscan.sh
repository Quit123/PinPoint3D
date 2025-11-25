#!/usr/bin/env bash
python ../evaluate.py --scan_folder=../data/MultiScan_new \
               --val_list=../data/val_list_multiscan.json \
               --output_dir=../results/Multiscan_val \
               --checkpoint=../weights/checkpoint1099.pth