# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
import MinkowskiEngine as ME
from utils.seg import mean_iou_scene, extend_clicks, get_simulated_clicks
import utils.misc as utils

from evaluation.evaluator_MO import EvaluatorMO
import os

# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

# from visualize import visualize_low_cases

def save_execution_log(args, results_dict, output_dir, execution_time):
    """Save execution results and parameters to a log file"""
    
    # Create log file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"execution_log_{timestamp}.json")
    
    # Prepare log data
    log_data = {
        "timestamp": timestamp,
        "execution_time_seconds": execution_time,
        "parameters": {
            "scan_folder": args.scan_folder,
            "val_list": args.val_list,
            "dialations": args.dialations,
            "conv1_kernel_size": args.conv1_kernel_size,
            "bn_momentum": args.bn_momentum,
            "voxel_size": args.voxel_size,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "num_heads": args.num_heads,
            "num_decoders": args.num_decoders,
            "num_bg_queries": args.num_bg_queries,
            "dropout": args.dropout,
            "pre_norm": args.pre_norm,
            "normalize_pos_enc": args.normalize_pos_enc,
            "positional_encoding_type": args.positional_encoding_type,
            "gauss_scale": args.gauss_scale,
            "hlevels": args.hlevels,
            "shared_decoder": args.shared_decoder,
            "aux": args.aux,
            "val_batch_size": args.val_batch_size,
            "device": args.device,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "num_workers": args.num_workers,
            "checkpoint": args.checkpoint,
            "max_num_clicks": args.max_num_clicks,
            "model_type": args.model_type,
            "use_fp16": args.use_fp16,
            "use_amp": args.use_amp
        },
        "results": results_dict,
        "model_info": {
            "checkpoint_path": args.checkpoint,
            "model_type": args.model_type
        }
    }
    
    # Save to JSON file
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"Execution log saved to: {log_file}")
    
    # Also save a summary text file
    summary_file = os.path.join(output_dir, f"execution_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Execution Summary - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        f.write("Key Parameters:\n")
        f.write(f"  Model Type: {args.model_type}\n")
        f.write(f"  Voxel Size: {args.voxel_size}\n")
        f.write(f"  Max Clicks: {args.max_num_clicks}\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Precision: {'FP16' if args.use_fp16 else 'FP32'}\n")
        f.write(f"  AMP: {'Enabled' if args.use_amp else 'Disabled'}\n\n")
        f.write("Results:\n")
        for key, value in results_dict.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Execution summary saved to: {summary_file}")
    
    return log_file, summary_file

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)

    # dataset
    parser.add_argument('--dataset_mode', default='part')
    parser.add_argument('--scan_folder', default='/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/preprocess_correct', type=str)
    parser.add_argument('--val_list', default='/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/val_scannet_randomEachScene_max10_close.json', type=str)
    parser.add_argument('--train_list', default='', type=str)
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool)

    # evaluation
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint1099.pth', help='resume from checkpoint')

    parser.add_argument('--max_num_clicks', default=10, help='maximum number of clicks per part on average', type=int)
 

     # model
    parser.add_argument('--model_type', default='pinpoint3D', type=str, choices=['agile3d', 'pinpoint3D'], 
                       help='Model type: agile3d (original) or pinpoint3D (two-level masks)')
    
    # Precision control
    parser.add_argument('--use_fp16', action='store_true', default=False,
                       help='Use FP16 (half precision) to reduce memory usage - NOTE: may not work with MinkowskiEngine')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (AMP) for better memory efficiency - safer option')
    
    
    return parser



def Evaluate(model, data_loader, args, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    instance_counter = 0
    # 创建结果文件记录评估数据
    results_file = os.path.join(args.output_dir, 'val_results_multi.csv')
    f = open(results_file, 'w')

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, num_obj, obj_label = batched_inputs
        coords = coords.to(device)
        raw_coords = raw_coords.to(device)
        
        # Convert features to half precision if requested (but keep compatible with MinkowskiEngine)
        if args.use_fp16:
            # Only convert features if not using MinkowskiEngine layers
            # For now, keep features in FP32 to avoid compatibility issues
            print("Keeping features in FP32 for MinkowskiEngine compatibility")
            # feats = feats.half()  # Commented out to avoid MinkowskiEngine issues
        
        labels = [l.to(device) for l in labels]
        labels_full = [l.to(device) for l in labels_full]
        selected_object_ids_new = {}

        data = ME.SparseTensor(
                                coordinates=coords,
                                features=feats,
                                device=device
                                )

        ###### interactive evaluation ######
        batch_idx = coords[:,0]
        batch_size = batch_idx.max()+1


        click_time_idx = copy.deepcopy(click_idx)

        click_idx = list(click_idx)
        click_time_idx = list(click_time_idx)

        current_num_clicks = 0

        # pre-compute backbone features only once
        # Use autocast for mixed precision, but be careful with MinkowskiEngine
        if args.use_amp and not args.use_fp16:
            # Only use autocast when not forcing FP16 to avoid MinkowskiEngine issues
            with autocast():
                pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)
        else:
            # Use regular forward pass for compatibility
            pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        # Calculate part_num for each scene and store in a list
        num_part = []
        for idx in range(batch_idx.max()+1):
            part_num = 0
            for obj_id, click_dict in click_idx[idx].items():
                if obj_id == '0':
                    continue
                for click_id, click_indices in click_dict.items():
                    part_num += 1
            num_part.append(part_num)
        
        # --- 新增：超过阈值直接跳过该样本 ---
        PART_CAP = 15  # 你要的阈值
        max_part_num = max(num_part) if len(num_part) > 0 else 0
        if max_part_num > PART_CAP:
            print(f"{scene_name[0]} skipped: part_num={max_part_num} > {PART_CAP}")
            # 可选：写一个占位行到结果文件，方便后处理
            try:
                line = f"{instance_counter} {scene_name[0]} {max_part_num} SKIPPED\n"
                f.write(line)
            except Exception:
                pass
            instance_counter += len(num_obj)
            continue
        ########################################
        max_part_num = max(num_part)
        # for part_num in num_part:

        # Use the first scene's part_num for max_num_clicks calculation
        max_num_clicks = max_part_num * args.max_num_clicks
        # index = 0

        for idx in range(batch_idx.max()+1):
            selected_object_ids_new[idx] = 1

        # while current_num_clicks <= max_num_clicks[len(max_num_clicks)-1]:
        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(device) for l in labels]
            else:

                # Use autocast for mixed precision, but be careful with MinkowskiEngine
                if args.use_amp and not args.use_fp16:
                    # Only use autocast when not forcing FP16 to avoid MinkowskiEngine issues
                    with autocast():
                        outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                                     click_idx=click_idx, click_time_idx=click_time_idx, target_object_id=selected_object_ids_new)
                        # outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                        #                              click_idx=click_idx, click_time_idx=click_time_idx)
                else:
                    # Use regular forward pass for compatibility
                    outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                                 click_idx=click_idx, click_time_idx=click_time_idx, target_object_id=selected_object_ids_new)
                if outputs == "ignore this test":
                    print("bad test")
                    break
                pred_logits = outputs['part_predictions_mask']
                pred = [p.argmax(-1) for p in pred_logits]

            updated_pred = []
            
            for idx in range(batch_idx.max()+1):
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]

                sample_mask = sample_mask.to(feats.device)  # Move sample_mask to the same device as feats
                sample_feats = feats[sample_mask]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for object_id, part_dict in click_idx[idx].items():
                        if object_id == '0':
                            for cids in part_dict:  # 这里假设 cids 是列表
                                sample_pred[cids] = 0
                            continue
                        for part_id, cids in part_dict.items():
                            sample_pred[cids] = int(part_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_map[idx]]

                sample_labels_full = labels_full[idx]
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)
                
                line = str(instance_counter+idx) + ' ' + scene_name[idx] + ' ' + str(num_part[idx]) + ' ' + str(current_num_clicks/num_part[idx]) + ' ' + str(
                sample_iou.cpu().numpy()) + '\n'
                f.write(line)

                print(scene_name[idx], 'Part: ', num_part[idx], 'average num clicks: ', current_num_clicks/num_part[idx],'num clicks: ',current_num_clicks, 'IOU: ', sample_iou.item())
                new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, labels_shield_qv=None, current_num_clicks=current_num_clicks, training=False)
                # print("new_clicks:",new_clicks)
                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

            # if current_num_clicks != 0:
            #     metric_logger.update(mIoU=mean_iou(updated_pred, labels))
            #     metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
            #                         **loss_dict_reduced_scaled,
            #                         **loss_dict_reduced_unscaled)

            if current_num_clicks == 0:
                new_clicks_num = num_part[idx]
                pass
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        instance_counter += len(num_obj)

    f.close()
    evaluator = EvaluatorMO(args.val_list, results_file, [0.5,0.65,0.8,0.85,0.9], args.max_num_clicks)
    results_dict = evaluator.eval_results()

    average_results_path = os.path.join(args.output_dir, 'val_results_multi.csv')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(results_dict)
    
    return stats

    
def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Additional seed settings for complete reproducibility
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set numpy and random seeds
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Using seed: {seed} for reproducibility")

    # build model
    model = build_model(args)
    model.to(device)
    
    # Set model to eval mode and disable randomness
    model.eval()
    
    # Convert model to half precision if requested
    if args.use_fp16:
        print("Converting model to FP16 (half precision) - excluding MinkowskiEngine layers")
        # Only convert non-MinkowskiEngine layers to FP16
        for name, module in model.named_modules():
            # Skip MinkowskiEngine layers
            if 'conv' in name and hasattr(module, 'conv'):
                print(f"Keeping {name} in FP32 for MinkowskiEngine compatibility")
                continue
            # Convert other layers to FP16
            if hasattr(module, 'weight') and module.weight.dtype == torch.float32:
                module = module.half()
        
        # Don't convert the entire model to avoid MinkowskiEngine issues
        print("Model partially converted to FP16 (MinkowskiEngine layers remain FP32)")
    

    # build dataset and dataloader
    dataset_val, collation_fn_val = build_dataset(split='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collation_fn_val, num_workers=args.num_workers,  # Set to 0 for reproducibility
                                 pin_memory=False)  # Set to False for reproducibility

    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    
    # Record execution start time
    start_time = time.time()
    
    # Run evaluation
    results_dict = Evaluate(model, data_loader_val, args, device)
    
    # Record execution end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    # Save execution log
    save_execution_log(args, results_dict, args.output_dir, execution_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script on interactive multi-object segmentation ', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
