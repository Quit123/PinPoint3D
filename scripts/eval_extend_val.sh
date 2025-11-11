#!/usr/bin/env bash
python evaluate_part.py --scan_folder=/home/yehangjian/scans_final \
               --val_list=data/val_list_extend.json \
               --output_dir=results/ScanNet_Part \
               --checkpoint=/home/yehangjian/interSeg3D-Studio-test/src/backend/pinpoint3d/weights/checkpoint1099.pth