#!/usr/bin/env python3
"""Lightweight sanity check for the PinPoint3D GPU stack.

This script builds the PinPoint3D model, sends a synthetic sparse point cloud
through the backbone and mask head, and verifies that everything runs on the
GPU without throwing runtime errors. Use it after installing the environment
or whenever you upgrade CUDA/PyTorch/MinkowskiEngine.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import MinkowskiEngine as ME

from models import build_model


def build_default_args() -> SimpleNamespace:
    """Return the minimal argument namespace expected by ``build_model``."""

    return SimpleNamespace(
        model_type="pinpoint3D",
        hidden_dim=128,
        num_heads=8,
        dim_feedforward=1024,
        shared_decoder=False,
        num_decoders=3,
        num_bg_queries=10,
        dropout=0.0,
        pre_norm=False,
        positional_encoding_type="fourier",
        normalize_pos_enc=True,
        hlevels=[4],
        voxel_size=0.05,
        gauss_scale=1.0,
        aux=True,
        bn_momentum=0.02,
        conv1_kernel_size=5,
        dialations=[1, 1, 1, 1],
    )


def generate_dummy_scene(extent: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dense lattice of integer coordinates and random features."""

    grid = torch.stack(torch.meshgrid(
        torch.arange(extent, dtype=torch.int32),
        torch.arange(extent, dtype=torch.int32),
        torch.arange(extent * 2, dtype=torch.int32),
        indexing="ij",
    ), dim=-1).reshape(-1, 3)

    coords = torch.cat([
        torch.zeros((grid.size(0), 1), dtype=torch.int32),
        grid,
    ], dim=1)

    feats = torch.randn(grid.size(0), 3, dtype=torch.float32)
    raw_coords = torch.randn(grid.size(0), 3, dtype=torch.float32)
    return coords, feats, raw_coords


def run(device: str = "cuda") -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    torch_device = torch.device(device)

    args = build_default_args()
    model = build_model(args).to(torch_device).eval()

    coords, feats, raw_coords = generate_dummy_scene()
    coords = coords.to(torch_device)
    feats = feats.to(torch_device)
    raw_coords = raw_coords.to(torch_device)

    sparse = ME.SparseTensor(coordinates=coords, features=feats, device=torch_device)
    pcd_features, aux, coordinates, pos_encodings = model.forward_backbone(
        sparse, raw_coordinates=raw_coords
    )

    click_idx = [{
        "0": [],
        "1": {"1": list(range(10))},
    }]
    click_time_idx = [{
        "0": [],
        "1": {"1": list(range(10))},
    }]
    target_object_id = {0: 1}

    outputs = model.forward_mask(
        pcd_features,
        aux,
        coordinates,
        pos_encodings,
        click_idx=click_idx,
        click_time_idx=click_time_idx,
        target_object_id=target_object_id,
    )

    mask = outputs["part_predictions_mask"][0]
    print(f"PinPoint3D dummy forward OK – mask shape: {tuple(mask.shape)} on {mask.device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU smoke test for PinPoint3D")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run the check on (cuda, cuda:0, cpu). Default: cuda",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args.device)
