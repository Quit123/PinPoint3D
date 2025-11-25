#!/usr/bin/env bash
python ../evaluate.py --scan_folder=../data/PartScan \
               --val_list=../data/val_list.json \
               --output_dir=../results/PartScan \
               --checkpoint=../weights/checkpoint1099.pth
