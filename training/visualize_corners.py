#!/usr/bin/env python3
"""Visualize predicted vs ground-truth corners on random images."""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

WORKTREE = Path("/home/johann/autoresearch/20260331-unet_dual_head")
sys.path.insert(0, str(WORKTREE / "training" / "autoresearch_v3"))
sys.path.insert(0, str(WORKTREE / "training"))

import candidate
from fixed_harness import CornerDataset, corner_distances, move_to_device

BEST_CHECKPOINT = "/home/johann/autoresearch/20260331-unet_dual_head/training/autoresearch_v3/runs/20260331-unet_dual_head/artifacts/20260406T122113Z_7e15e8e/best.pt"
ANNOTATIONS = "/home/johann/autoresearch/20260331-unet_dual_head/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN/training/data"

GROUP = "synthetic"
SPLIT = "val"
N_SAMPLES = 5
SEED = 42


def draw_corners(orig, gt, pred, dist):
    """Draw GT and predicted corners on image. Returns annotated copy."""
    h, w = orig.shape[:2]
    img = orig.copy()

    gt_px = gt.copy()
    gt_px[:, 0] *= w
    gt_px[:, 1] *= h
    pred_px = pred.copy()
    pred_px[:, 0] *= w
    pred_px[:, 1] *= h

    corner_names = ["TL", "TR", "BR", "BL"]
    gt_color = (0, 255, 0)
    pred_color = (0, 0, 255)

    gt_order = [0, 1, 2, 3]
    cv2.polylines(img, [gt_px[gt_order].astype(np.int32)], True, gt_color, 2)
    cv2.polylines(img, [pred_px[gt_order].astype(np.int32)], True, pred_color, 2)

    for i, name in enumerate(corner_names):
        gx, gy = int(gt_px[i, 0]), int(gt_px[i, 1])
        px, py = int(pred_px[i, 0]), int(pred_px[i, 1])
        cv2.circle(img, (gx, gy), 8, gt_color, -1)
        cv2.circle(img, (px, py), 8, pred_color, -1)
        cv2.line(img, (gx, gy), (px, py), (255, 0, 255), 2)
        cv2.putText(img, name, (gx + 12, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)

    cv2.putText(img, f"Green=GT  Red=Pred  dist={dist:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    device = torch.device("cpu")
    defaults = candidate.get_defaults()

    model = candidate.build_model(input_channels=2)
    ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    candidate.load_checkpoint(model, ckpt["model_state_dict"])
    model.eval()

    ds = CornerDataset(
        annotations_path=ANNOTATIONS,
        images_root=IMAGES_ROOT,
        group=GROUP,
        split=SPLIT,
        candidate=candidate,
        candidate_defaults=defaults,
        input_mode="gray_edges",
        img_size=384,
        augment=False,
        seed=1337,
    )

    random.seed(SEED)
    indices = random.sample(range(len(ds)), N_SAMPLES)

    panels = []
    for idx in indices:
        image, targets, meta = ds[idx]

        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
            pred_coords = candidate.decode_coords(outputs)

        pred = pred_coords[0].numpy().reshape(4, 2)
        gt = targets["coords"].numpy().reshape(4, 2)
        dists = np.linalg.norm(pred - gt, axis=1)
        mean_dist = dists.mean()

        img_path = Path(IMAGES_ROOT) / meta["path"]
        orig = cv2.imread(str(img_path))
        panel = draw_corners(orig, gt, pred, mean_dist)

        # Resize all panels to same height for grid
        target_h = 512
        scale = target_h / panel.shape[0]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), target_h))
        panels.append(panel)

        print(f"  {meta['path']}: dist={mean_dist:.6f}")

    # Arrange in grid: top row 3, bottom row 2 (centered)
    max_w = max(p.shape[1] for p in panels)
    def pad_to(img, w):
        if img.shape[1] < w:
            pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img

    row1 = np.hstack([pad_to(p, max_w) for p in panels[:3]])
    # Center the bottom row
    row2_imgs = np.hstack([pad_to(p, max_w) for p in panels[3:]])
    left_pad = (row1.shape[1] - row2_imgs.shape[1]) // 2
    if left_pad > 0:
        row2 = np.zeros((row2_imgs.shape[0], row1.shape[1], 3), dtype=np.uint8)
        row2[:, left_pad:left_pad + row2_imgs.shape[1]] = row2_imgs
    else:
        row2 = row2_imgs

    grid = np.vstack([row1, row2])

    out_path = "/home/johann/ImagetoFEN/training/viz_synthetic_5.png"
    cv2.imwrite(out_path, grid)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
