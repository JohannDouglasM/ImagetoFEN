#!/usr/bin/env python3
"""
U-Net-style dual-head candidate template.

Outputs:
- board segmentation mask
- four corner heatmaps
- corner coordinates decoded with soft-argmax
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

DEFAULTS = {
    "candidate_name": "gray_edges_unet_dual_head_lr15e4_wd01_bs48",
    "img_size": 384,
    "input_mode": "gray_edges",
    "batch_size": 48,
    "lr": 0.0015,
    "weight_decay": 0.01,
    "eval_interval_s": 300.0,
    "train_splits": "chessred2k:train,user:train,chess_dataset_recovered:train",
    "val_splits": "chessred2k:val,chess_dataset_recovered:val",
    "report_splits": "chessred2k:val,chess_dataset_recovered:val",
    "max_no_improve_evals": 4,
    "resume_candidates": [],
    "allow_legacy_resume_fallback": False,
    "decoder_size": 96,
    "heatmap_sigma": 2.0,
    "mask_loss_weight": 0.5,
    "heatmap_loss_weight": 1.0,
    "coord_loss_weight": 0.2,
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


def make_aux_heatmap(square_centers, orig_w, orig_h, out_size):
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
        channels.append(make_aux_heatmap(square_centers or [], w, h, out_size))
    return np.stack(channels, axis=0)


def augment_image(image_bgr, rng):
    beta = float(rng.uniform(-45.0, 45.0))
    alpha = float(rng.uniform(0.55, 1.45))
    image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        ksize = int(rng.choice([3, 5, 7]))
        image_bgr = cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)
    if rng.random() < 0.25:
        dx = int(rng.integers(-6, 7))
        dy = int(rng.integers(-6, 7))
        kernel = np.zeros((9, 9), dtype=np.float32)
        cv2.line(kernel, (4 - dx, 4 - dy), (4 + dx, 4 + dy), 1.0, 1)
        kernel /= max(kernel.sum(), 1.0)
        image_bgr = cv2.filter2D(image_bgr, -1, kernel)
    return image_bgr


def convert_conv1_weights(conv_weight, input_channels):
    if input_channels == conv_weight.shape[1]:
        return conv_weight
    mean_weight = conv_weight.mean(dim=1, keepdim=True)
    return mean_weight.repeat(1, input_channels, 1, 1) * (3.0 / float(input_channels))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDualHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        backbone = models.resnet18(weights=None)
        cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
        if cached.exists():
            state = torch.load(cached, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)

        if input_channels != 3:
            original = backbone.conv1
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
            backbone.conv1 = new_conv

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 96)
        self.mask_head = nn.Conv2d(96, 1, kernel_size=1)
        self.heatmap_head = nn.Conv2d(96, 4, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        y = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, c3], dim=1))
        y = F.interpolate(y, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, c2], dim=1))
        y = F.interpolate(y, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, c1], dim=1))

        heatmaps = self.heatmap_head(y)
        mask_logits = self.mask_head(y)
        coords = soft_argmax_decode(heatmaps)
        return {
            "mask_logits": mask_logits,
            "heatmaps": heatmaps,
            "coords": coords,
        }


def build_model(input_channels):
    return UNetDualHead(input_channels)


def make_targets(corners, *, orig_w, orig_h, img_size, defaults):
    decoder_size = int(defaults["decoder_size"])
    sigma = float(defaults["heatmap_sigma"])

    coords = np.array([
        [corners["top_left"][0] / orig_w, corners["top_left"][1] / orig_h],
        [corners["top_right"][0] / orig_w, corners["top_right"][1] / orig_h],
        [corners["bottom_right"][0] / orig_w, corners["bottom_right"][1] / orig_h],
        [corners["bottom_left"][0] / orig_w, corners["bottom_left"][1] / orig_h],
    ], dtype=np.float32)

    xs = np.arange(decoder_size, dtype=np.float32)[None, :]
    ys = np.arange(decoder_size, dtype=np.float32)[:, None]
    heatmaps = np.zeros((4, decoder_size, decoder_size), dtype=np.float32)
    for idx, (x_norm, y_norm) in enumerate(coords):
        cx = x_norm * (decoder_size - 1)
        cy = y_norm * (decoder_size - 1)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        heatmaps[idx] = np.exp(-d2 / (2 * sigma ** 2))

    poly = np.array(
        [[
            [coords[0, 0] * (decoder_size - 1), coords[0, 1] * (decoder_size - 1)],
            [coords[1, 0] * (decoder_size - 1), coords[1, 1] * (decoder_size - 1)],
            [coords[2, 0] * (decoder_size - 1), coords[2, 1] * (decoder_size - 1)],
            [coords[3, 0] * (decoder_size - 1), coords[3, 1] * (decoder_size - 1)],
        ]],
        dtype=np.int32,
    )
    mask = np.zeros((decoder_size, decoder_size), dtype=np.float32)
    cv2.fillConvexPoly(mask, poly[0], 1.0)

    return {
        "coords": torch.from_numpy(coords.reshape(-1)),
        "heatmaps": torch.from_numpy(heatmaps),
        "mask": torch.from_numpy(mask[None, :, :]),
    }


def soft_argmax_decode(heatmaps, beta=20.0):
    b, c, h, w = heatmaps.shape
    flat = heatmaps.view(b, c, -1)
    probs = torch.softmax(flat * beta, dim=-1)
    xs = torch.linspace(0.0, 1.0, w, device=heatmaps.device)
    ys = torch.linspace(0.0, 1.0, h, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = torch.sum(probs * grid_x.reshape(-1), dim=-1)
    y = torch.sum(probs * grid_y.reshape(-1), dim=-1)
    return torch.stack([x, y], dim=-1).view(b, c * 2)


def weighted_mse(pred, target, positive_weight=25.0):
    weights = 1.0 + target * (positive_weight - 1.0)
    return ((pred - target) ** 2 * weights).mean()


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    numer = 2.0 * (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (numer / denom).mean()


def weighted_bce_with_logits(logits, target, positive_weight=3.0):
    pos_weight = torch.tensor([positive_weight], device=logits.device, dtype=logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def compute_loss(outputs, targets):
    mask_loss = 0.5 * weighted_bce_with_logits(outputs["mask_logits"], targets["mask"]) + 0.5 * dice_loss_from_logits(outputs["mask_logits"], targets["mask"])
    heatmap_loss = weighted_mse(torch.sigmoid(outputs["heatmaps"]), targets["heatmaps"])
    coord_loss = nn.functional.smooth_l1_loss(outputs["coords"], targets["coords"])
    total = (
        DEFAULTS["heatmap_loss_weight"] * heatmap_loss
        + DEFAULTS["mask_loss_weight"] * mask_loss
        + DEFAULTS["coord_loss_weight"] * coord_loss
    )
    return total, {
        "mask_loss": float(mask_loss.detach().item()),
        "heatmap_loss": float(heatmap_loss.detach().item()),
        "coord_loss": float(coord_loss.detach().item()),
    }


def decode_coords(outputs):
    return outputs["coords"]


def create_optimizer(model, *, lr, weight_decay, resumed):
    effective_lr = lr * 0.3 if resumed else lr
    return optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)


def create_scheduler(optimizer, *, total_train_steps):
    warmup_steps = max(1, int(total_train_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(1, total_train_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    patched_state = {}
    skipped_mismatch = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        patched_value = value
        if key == "stem.0.weight" and value.shape != target.shape and value.ndim == 4:
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