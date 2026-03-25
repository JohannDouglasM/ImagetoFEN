#!/usr/bin/env python3
"""
ResNet coordinate-regression candidate template.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

DEFAULTS = {
    "candidate_name": "gray_edges_resnet18_coords",
    "img_size": 384,
    "input_mode": "gray_edges",
    "batch_size": 16,
    "lr": 0.0001,
    "weight_decay": 0.001,
    "eval_interval_s": 180.0,
    "train_splits": "chessred2k:train,user:train,chess_dataset_recovered:train",
    "val_splits": "chessred2k:val,chess_dataset_recovered:val",
    "report_splits": "chessred2k:val,chess_dataset_recovered:val",
    "max_no_improve_evals": 3,
    "resume_candidates": [
        "autoresearch_gray_edges_mixed_models/best_corner_hybrid.pt",
        "ablation/gray_edges_10ep/last_corner_hybrid.pt",
        "models/best_corner_hybrid.pt",
    ],
}

INPUT_MODE_TO_CHANNELS = {
    "hybrid": 3,
    "gray_edges": 2,
    "gray": 1,
}

HEATMAP_SIGMA = 8


def get_defaults():
    return dict(DEFAULTS)


def needs_square_centers(input_mode):
    return input_mode == "hybrid"


def find_empty_squares_for_image(image_bgr):
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    min_side = min(h, w) * 0.015
    max_side = min(h, w) * 0.12

    best_pts = []
    best_score = 0.0

    for thresh in [70, 80, 90, 100, 110, 120]:
        for flag in [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]:
            _, binary = cv2.threshold(l_ch, thresh, 255, flag)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                side = np.sqrt(area)
                if side < min_side or side > max_side:
                    continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if max(rw, rh) < 1:
                    continue
                if min(rw, rh) / max(rw, rh) < 0.45:
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0
                if solidity < 0.85:
                    continue

                bx, by, bw_r, bh_r = cv2.boundingRect(cnt)
                roi = l_ch[by:by + bh_r, bx:bx + bw_r]
                if roi.size == 0:
                    continue

                roi_mask = np.zeros((bh_r, bw_r), dtype=np.uint8)
                cnt_shifted = cnt.copy()
                cnt_shifted[:, :, 0] -= bx
                cnt_shifted[:, :, 1] -= by
                cv2.drawContours(roi_mask, [cnt_shifted], 0, 255, -1)

                pixels = roi[roi_mask > 0]
                if len(pixels) == 0 or np.std(pixels) > 18:
                    continue

                pts.append((rect[0][0], rect[0][1]))
                areas.append(area)

            if len(pts) < 4:
                continue

            areas_arr = np.array(areas)
            med_area = np.median(areas_arr)
            keep = (areas_arr > med_area * 0.4) & (areas_arr < med_area * 2.5)
            pts_filt = [p for p, keep_flag in zip(pts, keep) if keep_flag]
            if len(pts_filt) < 4:
                continue

            areas_filt = areas_arr[keep]
            size_cv = np.std(areas_filt) / np.mean(areas_filt)
            consistency = max(0.0, 1.0 - size_cv)
            score = len(pts_filt) * (0.5 + 0.5 * consistency)
            if score > best_score:
                best_score = score
                best_pts = pts_filt

    return best_pts


def make_heatmap(square_centers, orig_w, orig_h, out_size):
    heatmap = np.zeros((out_size, out_size), dtype=np.float32)
    if not square_centers:
        return heatmap

    sx = out_size / orig_w
    sy = out_size / orig_h

    for cx, cy in square_centers:
        x = cx * sx
        y = cy * sy
        x0 = max(0, int(x - 3 * HEATMAP_SIGMA))
        x1 = min(out_size, int(x + 3 * HEATMAP_SIGMA) + 1)
        y0 = max(0, int(y - 3 * HEATMAP_SIGMA))
        y1 = min(out_size, int(y + 3 * HEATMAP_SIGMA) + 1)

        for iy in range(y0, y1):
            for ix in range(x0, x1):
                d2 = (ix - x) ** 2 + (iy - y) ** 2
                heatmap[iy, ix] = max(heatmap[iy, ix], np.exp(-d2 / (2 * HEATMAP_SIGMA ** 2)))

    return heatmap


def make_input_tensor(image_bgr, square_centers=None, input_mode="gray_edges", out_size=384):
    if input_mode not in INPUT_MODE_TO_CHANNELS:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (out_size, out_size))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 80, 200).astype(np.float32) / 255.0

    channels = [gray]
    if input_mode in {"hybrid", "gray_edges"}:
        channels.append(edges)
    if input_mode == "hybrid":
        channels.append(make_heatmap(square_centers or [], w, h, out_size))

    return np.stack(channels, axis=0)


def augment_image(image_bgr, rng):
    beta = float(rng.uniform(-40.0, 40.0))
    alpha = float(rng.uniform(0.6, 1.4))
    return np.clip(alpha * image_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def convert_conv1_weights(conv_weight, input_channels):
    if input_channels == conv_weight.shape[1]:
        return conv_weight
    mean_weight = conv_weight.mean(dim=1, keepdim=True)
    return mean_weight.repeat(1, input_channels, 1, 1) * (3.0 / float(input_channels))


def build_model(input_channels):
    model = models.resnet18(weights=None)
    cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if cached.exists():
        state = torch.load(cached, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    if input_channels != 3:
        original = model.conv1
        new_conv = nn.Conv2d(
            input_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(convert_conv1_weights(original.weight.data, input_channels))
            if original.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(original.bias.data)
        model.conv1 = new_conv

    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n_features, 8),
        nn.Sigmoid(),
    )
    return model


def make_targets(corners, *, orig_w, orig_h, img_size, defaults):
    return {
        "coords": torch.tensor([
            corners["top_left"][0] / orig_w,
            corners["top_left"][1] / orig_h,
            corners["top_right"][0] / orig_w,
            corners["top_right"][1] / orig_h,
            corners["bottom_right"][0] / orig_w,
            corners["bottom_right"][1] / orig_h,
            corners["bottom_left"][0] / orig_w,
            corners["bottom_left"][1] / orig_h,
        ], dtype=torch.float32)
    }


def compute_loss(outputs, targets):
    loss = nn.functional.smooth_l1_loss(outputs, targets["coords"])
    return loss, {"coord_loss": float(loss.detach().item())}


def decode_coords(outputs):
    return outputs


def create_optimizer(model, *, lr, weight_decay, resumed):
    effective_lr = lr * 0.1 if resumed else lr
    return optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)


def create_scheduler(optimizer, *, total_train_steps):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_train_steps))


def load_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    patched_state = {}
    skipped_mismatch = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        patched_value = value
        if key == "conv1.weight" and value.shape != target.shape:
            patched_value = convert_conv1_weights(value, target.shape[1])
        if patched_value.shape != target.shape:
            skipped_mismatch.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        patched_state[key] = patched_value

    missing, unexpected = model.load_state_dict(patched_state, strict=False)
    return {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped_mismatch": skipped_mismatch,
    }
