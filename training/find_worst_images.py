#!/usr/bin/env python3
"""Find the worst-performing image in each validation dataset."""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

WORKTREE = Path("/home/johann/autoresearch/20260331-unet_dual_head")
sys.path.insert(0, str(WORKTREE / "training" / "autoresearch_v3"))
sys.path.insert(0, str(WORKTREE / "training"))

import candidate
from fixed_harness import CornerDataset, corner_distances, move_to_device

BEST_CHECKPOINT = "/home/johann/autoresearch/20260331-unet_dual_head/training/autoresearch_v3/runs/20260331-unet_dual_head/artifacts/20260404T225955Z_92353d6/best.pt"
ANNOTATIONS = "/home/johann/autoresearch/20260331-unet_dual_head/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN/training/data"

VAL_SPLITS = [
    ("chessred2k", "val"),
    ("chess_dataset_recovered", "val"),
    ("synthetic", "val"),
]


def main():
    device = torch.device("cpu")
    defaults = candidate.get_defaults()

    # Load model
    model = candidate.build_model(input_channels=2)  # gray_edges
    ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    candidate.load_checkpoint(model, ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    for group, split in VAL_SPLITS:
        print(f"\n{'='*60}")
        print(f"  {group}:{split}")
        print(f"{'='*60}")

        ds = CornerDataset(
            annotations_path=ANNOTATIONS,
            images_root=IMAGES_ROOT,
            group=group,
            split=split,
            candidate=candidate,
            candidate_defaults=defaults,
            input_mode="gray_edges",
            img_size=384,
            augment=False,
            seed=1337,
        )

        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        all_dists = []
        all_meta = []

        with torch.no_grad():
            for images, targets, meta in loader:
                images = images.to(device)
                targets = move_to_device(targets, device)
                outputs = model(images)
                pred_coords = candidate.decode_coords(outputs)
                dists = corner_distances(pred_coords, targets["coords"])
                # dists shape: (batch, 4) - per corner distances
                mean_dists = dists.mean(dim=1)  # per-image mean

                for i in range(len(mean_dists)):
                    all_dists.append(float(mean_dists[i]))
                    all_meta.append({
                        "image_id": int(meta["image_id"][i]),
                        "path": meta["path"][i],
                        "per_corner": [float(d) for d in dists[i]],
                    })

        # Sort by worst
        ranked = sorted(zip(all_dists, all_meta), key=lambda x: -x[0])

        print(f"  Total images: {len(ranked)}")
        print(f"  Mean dist: {np.mean(all_dists):.6f}")
        print(f"  Median dist: {np.median(all_dists):.6f}")
        print(f"\n  Top 5 worst:")
        for i, (dist, meta) in enumerate(ranked[:5]):
            print(f"    {i+1}. dist={dist:.6f}  id={meta['image_id']}  path={meta['path']}")
            print(f"       per_corner: {[f'{d:.4f}' for d in meta['per_corner']]}")


if __name__ == "__main__":
    main()
