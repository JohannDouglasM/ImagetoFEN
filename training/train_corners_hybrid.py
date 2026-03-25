#!/usr/bin/env python3
"""
Train a corner detection model for chess board images using hybrid handcrafted inputs.

Input modes:
  - hybrid:     grayscale + Canny edges + square-center heatmap
  - gray_edges: grayscale + Canny edges
  - gray:       grayscale only

Output:
  8 normalized floats for board corners:
  [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

BASE_DIR = Path(__file__).parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = 384
HEATMAP_SIGMA = 8

INPUT_MODE_TO_CHANNELS = {
    "hybrid": 3,
    "gray_edges": 2,
    "gray": 1,
}


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def resolve_data_paths(args):
    annotations_candidates = []
    image_root_candidates = []
    if args.annotations:
        annotations_candidates.append(Path(args.annotations))
    annotations_candidates.extend([
        BASE_DIR / "data" / "annotations.json",
        BASE_DIR.parent / "annotations.json",
    ])

    if args.images_root:
        image_root_candidates.append(Path(args.images_root))
    image_root_candidates.extend([
        BASE_DIR / "data" / "chessred2k",
        BASE_DIR / "data",
    ])

    annotations_path = next((p for p in annotations_candidates if p.exists()), None)
    images_root = next((p for p in image_root_candidates if (p / "images").exists()), None)

    if annotations_path is None:
        print("annotations.json not found")
        sys.exit(1)
    if images_root is None:
        print("images/ directory not found")
        sys.exit(1)
    return annotations_path, images_root


def write_status(status_file, message):
    if not status_file:
        return
    path = Path(status_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def find_empty_squares_for_image(image_bgr):
    """
    Detect empty square centers on a BGR image.
    Simplified version of _find_empty_squares from detect_board_v5.py.
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    min_side = min(h, w) * 0.015
    max_side = min(h, w) * 0.12

    best_pts = []
    best_score = 0

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
                solidity = area / hull_area if hull_area > 0 else 0
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
            consistency = max(0, 1.0 - size_cv)
            score = len(pts_filt) * (0.5 + 0.5 * consistency)
            if score > best_score:
                best_score = score
                best_pts = pts_filt

    return best_pts


def make_heatmap(square_centers, orig_w, orig_h, out_size=IMG_SIZE):
    heatmap = np.zeros((out_size, out_size), dtype=np.float32)
    if not square_centers:
        return heatmap

    sx = out_size / orig_w
    sy = out_size / orig_h
    sigma = HEATMAP_SIGMA

    for cx, cy in square_centers:
        x = cx * sx
        y = cy * sy
        x0 = max(0, int(x - 3 * sigma))
        x1 = min(out_size, int(x + 3 * sigma) + 1)
        y0 = max(0, int(y - 3 * sigma))
        y1 = min(out_size, int(y + 3 * sigma) + 1)

        for iy in range(y0, y1):
            for ix in range(x0, x1):
                d2 = (ix - x) ** 2 + (iy - y) ** 2
                heatmap[iy, ix] = max(heatmap[iy, ix], np.exp(-d2 / (2 * sigma ** 2)))

    return heatmap


def make_input_tensor(image_bgr, square_centers=None, input_mode="hybrid", out_size=IMG_SIZE):
    """
    Build a model input tensor in CHW float32 format.
    """
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


def convert_conv1_weights(conv_weight, input_channels):
    if input_channels == conv_weight.shape[1]:
        return conv_weight
    mean_weight = conv_weight.mean(dim=1, keepdim=True)
    scaled = mean_weight.repeat(1, input_channels, 1, 1) * (3.0 / float(input_channels))
    return scaled


class HybridCornerDataset(Dataset):
    def __init__(self, annotations_path, images_root, split="train",
                 input_mode="hybrid", augment=False, cache_squares=True,
                 split_group="chessred2k"):
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.augment = augment
        self.input_mode = input_mode
        self.requires_heatmap = input_mode == "hybrid"
        self.cache_squares = cache_squares and self.requires_heatmap
        self.square_cache = {}

        self.images = {img["id"]: img for img in data["images"]}
        self.corner_map = {
            corner["image_id"]: corner["corners"]
            for corner in data["annotations"]["corners"]
        }

        split_ids = data["splits"][split_group][split]["image_ids"]
        sample_ids = [
            img_id for img_id in split_ids
            if img_id in self.images and img_id in self.corner_map
        ]

        valid_ids = []
        missing = 0
        for img_id in sample_ids:
            img_path = self.images_root / self.images[img_id]["path"]
            if img_path.exists():
                valid_ids.append(img_id)
            else:
                missing += 1

        self.sample_ids = sorted(valid_ids)
        mode_label = f"{split_group}:{split}/{input_mode}"
        print(f"Hybrid corner dataset {mode_label}: {len(self.sample_ids)} samples", flush=True)
        if missing:
            print(f"  Skipped {missing} missing files", flush=True)

    def _get_squares(self, img_id, image_bgr):
        if not self.requires_heatmap:
            return []
        if img_id in self.square_cache:
            return self.square_cache[img_id]
        centers = find_empty_squares_for_image(image_bgr)
        if self.cache_squares:
            self.square_cache[img_id] = centers
        return centers

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_info = self.images[img_id]
        corners = self.corner_map[img_id]

        img_path = self.images_root / img_info["path"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {img_path}")

        orig_h, orig_w = image_bgr.shape[:2]
        square_centers = self._get_squares(img_id, image_bgr)

        if self.augment:
            beta = np.random.uniform(-40, 40)
            alpha = np.random.uniform(0.6, 1.4)
            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        input_tensor = make_input_tensor(
            image_bgr,
            square_centers=square_centers,
            input_mode=self.input_mode,
            out_size=IMG_SIZE,
        )

        target = torch.tensor([
            corners["top_left"][0] / orig_w,
            corners["top_left"][1] / orig_h,
            corners["top_right"][0] / orig_w,
            corners["top_right"][1] / orig_h,
            corners["bottom_right"][0] / orig_w,
            corners["bottom_right"][1] / orig_h,
            corners["bottom_left"][0] / orig_w,
            corners["bottom_left"][1] / orig_h,
        ], dtype=torch.float32)

        return torch.from_numpy(input_tensor), target


class CombinedCornerDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index = []
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))
        print(f"Combined corner dataset: {len(self.index)} samples from {len(datasets)} splits", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        return self.datasets[dataset_idx][sample_idx]


def build_model(input_channels=3):
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

    n = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(n, 8),
        nn.Sigmoid(),
    )
    return model


def corner_distance(pred, target):
    pred = pred.view(-1, 4, 2)
    target = target.view(-1, 4, 2)
    dists = torch.sqrt(((pred - target) ** 2).sum(dim=2))
    return dists.mean()


def train_epoch(model, loader, criterion, optimizer, device, epoch_str="", status_file=None):
    model.train()
    total_loss = 0.0
    total_dist = 0.0
    total = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dist += corner_distance(outputs, targets).item() * images.size(0)
        total += images.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            message = (
                f"  {epoch_str} train {(i + 1) / n_batches * 100:5.1f}% | "
                f"batch {i + 1}/{n_batches} | loss={total_loss / total:.4f} "
                f"dist={total_dist / total:.4f}"
            )
            print(message, flush=True)
            write_status(status_file, message.strip())

    return total_loss / total, total_dist / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dist = 0.0
    total = 0
    all_dists = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            pred = outputs.view(-1, 4, 2)
            targ = targets.view(-1, 4, 2)
            dists = torch.sqrt(((pred - targ) ** 2).sum(dim=2))
            total_dist += dists.mean().item() * images.size(0)
            total += images.size(0)
            all_dists.append(dists.cpu())

    all_dists = torch.cat(all_dists, dim=0)
    return (
        total_loss / total,
        total_dist / total,
        all_dists.max().item(),
        all_dists.mean(dim=0),
    )


def export_onnx(model_path, input_channels, output_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    export_model = build_model(input_channels=input_channels)
    export_model.load_state_dict(checkpoint["model_state_dict"])
    export_model.eval()

    dummy = torch.randn(1, input_channels, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["corners"],
        dynamic_axes={"image": {0: "batch"}, "corners": {0: "batch"}},
        opset_version=18,
        dynamo=False,
        external_data=False,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX to {output_path} ({size_mb:.1f} MB)", flush=True)


def load_checkpoint_flexible(model, checkpoint_state):
    """
    Load a checkpoint even if the first conv input-channel count changed.
    This is useful for ablation runs that drop the heatmap channel but want
    to start from the current best 3-channel checkpoint.
    """
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
    if skipped_mismatch:
        preview = ", ".join(
            f"{key} {src}->{dst}" for key, src, dst in skipped_mismatch[:4]
        )
        if len(skipped_mismatch) > 4:
            preview += ", ..."
        print(f"  Skipped mismatched resume tensors: {preview}", flush=True)
    return missing, unexpected


def save_checkpoint(path, model, epoch, val_dist, input_mode, input_channels):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_dist": val_dist,
            "img_size": IMG_SIZE,
            "input_mode": input_mode,
            "input_channels": input_channels,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train hybrid corner detection model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument(
        "--input-mode",
        choices=sorted(INPUT_MODE_TO_CHANNELS.keys()),
        default="gray_edges",
        help="gray_edges is the default mixed-data path; hybrid keeps the square heatmap channel",
    )
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-splits",
        type=str,
        default="chessred2k:train,user:train,chess_dataset_recovered:train",
        help="Comma-separated split selectors like chessred2k:train,user:train",
    )
    parser.add_argument(
        "--val-splits",
        type=str,
        default="chessred2k:val,chess_dataset_recovered:val",
        help="Comma-separated split selectors like chessred2k:val,user:val",
    )
    args = parser.parse_args()

    annotations_path, images_root = resolve_data_paths(args)
    models_dir = Path(args.models_dir)
    input_channels = INPUT_MODE_TO_CHANNELS[args.input_mode]

    print(f"Annotations: {annotations_path}")
    print(f"Images root:  {images_root}")
    print(f"Input mode:   {args.input_mode} ({input_channels} channels)")

    device = get_device()

    def parse_split_selectors(raw):
        selectors = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid split selector '{item}', expected group:split")
            group, split = item.split(":", 1)
            selectors.append((group, split))
        return selectors

    train_datasets = [
        HybridCornerDataset(
            annotations_path,
            images_root,
            split=split,
            split_group=group,
            input_mode=args.input_mode,
            augment=True,
            cache_squares=True,
        )
        for group, split in parse_split_selectors(args.train_splits)
    ]
    val_datasets = [
        HybridCornerDataset(
            annotations_path,
            images_root,
            split=split,
            split_group=group,
            input_mode=args.input_mode,
            augment=False,
            cache_squares=True,
        )
        for group, split in parse_split_selectors(args.val_splits)
    ]

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else CombinedCornerDataset(train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else CombinedCornerDataset(val_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model(input_channels=input_channels)
    best_val_dist = float("inf")
    no_improve = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        missing, unexpected = load_checkpoint_flexible(model, ckpt["model_state_dict"])
        if missing:
            print(f"  Missing keys on resume: {missing}", flush=True)
        if unexpected:
            print(f"  Unexpected keys on resume: {unexpected}", flush=True)
        best_val_dist = min(best_val_dist, ckpt.get("val_dist", best_val_dist))

    model = model.to(device)
    criterion = nn.SmoothL1Loss()

    head_epochs = min(3, args.epochs) if not args.resume else 0
    if head_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining corner detector for {args.epochs} epochs on {device}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE} x {input_channels}ch")
    print("=" * 60, flush=True)

    if head_epochs > 0:
        print(f"\nPhase 1: Head only for {head_epochs} epochs", flush=True)
        for epoch in range(head_epochs):
            start = time.time()
            epoch_str = f"[{epoch + 1}/{args.epochs}]"
            train_loss, train_dist = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch_str, args.status_file
            )
            val_loss, val_dist, val_max, per_corner = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start
            summary = (
                f"Epoch {epoch + 1}/{args.epochs} ({elapsed:.0f}s) | "
                f"Train: loss={train_loss:.4f} dist={train_dist:.4f} | "
                f"Val: loss={val_loss:.4f} dist={val_dist:.4f} max={val_max:.4f}"
            )
            print(summary, flush=True)
            write_status(args.status_file, summary)
            if val_dist < best_val_dist:
                best_val_dist = val_dist
                save_checkpoint(
                    models_dir / "best_corner_hybrid.pt",
                    model,
                    epoch,
                    val_dist,
                    args.input_mode,
                    input_channels,
                )
                print(f"  -> New best! dist={val_dist:.4f}", flush=True)

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - head_epochs))

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs (early stop after 7 no-improve)")
    print("=" * 60, flush=True)

    for epoch in range(remaining):
        start = time.time()
        epoch_num = head_epochs + epoch + 1
        epoch_str = f"[{epoch_num}/{args.epochs}]"
        train_loss, train_dist = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch_str, args.status_file
        )
        val_loss, val_dist, val_max, per_corner = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        corner_names = ["TL", "TR", "BR", "BL"]
        per_corner_str = " ".join(f"{name}={dist:.4f}" for name, dist in zip(corner_names, per_corner))
        summary = (
            f"Epoch {epoch_num}/{args.epochs} ({elapsed:.0f}s) | "
            f"Train: loss={train_loss:.4f} dist={train_dist:.4f} | "
            f"Val: loss={val_loss:.4f} dist={val_dist:.4f} max={val_max:.4f} | "
            f"{per_corner_str}"
        )
        print(summary, flush=True)
        write_status(args.status_file, summary)

        if val_dist < best_val_dist:
            best_val_dist = val_dist
            no_improve = 0
            save_checkpoint(
                models_dir / "best_corner_hybrid.pt",
                model,
                epoch_num - 1,
                val_dist,
                args.input_mode,
                input_channels,
            )
            print(f"  -> New best! dist={val_dist:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print("  Early stopping: no improvement for 7 epochs", flush=True)
                break

    if args.save_last:
        save_checkpoint(
            models_dir / "last_corner_hybrid.pt",
            model,
            epoch_num - 1,
            val_dist,
            args.input_mode,
            input_channels,
        )
        print(f"Saved final checkpoint to {models_dir / 'last_corner_hybrid.pt'}", flush=True)

    print(f"\n{'=' * 60}")
    print(f"Done! Best mean corner distance: {best_val_dist:.4f}")
    print(f"  (On a 3072px image, {best_val_dist:.4f} ~ {best_val_dist * 3072:.0f}px per corner)")

    if args.export_onnx:
        print("\nExporting to ONNX...", flush=True)
        export_onnx(
            models_dir / "best_corner_hybrid.pt",
            input_channels=input_channels,
            output_path=models_dir / "corner_hybrid.onnx",
        )


if __name__ == "__main__":
    main()
