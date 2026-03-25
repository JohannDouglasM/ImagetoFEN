#!/usr/bin/env python3
"""
Train a chessboard corner detector that predicts corner heatmaps instead of
regressing coordinates directly.

The model shares the same handcrafted inputs as train_corners_hybrid.py, then
decodes four corner heatmaps (TL/TR/BR/BL) with a small ResNet-18 + decoder.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_corners_hybrid import (
    BASE_DIR,
    IMG_SIZE,
    INPUT_MODE_TO_CHANNELS,
    convert_conv1_weights,
    get_device,
    make_input_tensor,
    resolve_data_paths,
    write_status,
    find_empty_squares_for_image,
)

HEATMAP_SIZE = 96
TARGET_SIGMA = 2.5


def make_target_heatmaps(corners, orig_w, orig_h, size=HEATMAP_SIZE, sigma=TARGET_SIGMA):
    heatmaps = np.zeros((4, size, size), dtype=np.float32)
    coords = np.array([
        [corners["top_left"][0] / orig_w, corners["top_left"][1] / orig_h],
        [corners["top_right"][0] / orig_w, corners["top_right"][1] / orig_h],
        [corners["bottom_right"][0] / orig_w, corners["bottom_right"][1] / orig_h],
        [corners["bottom_left"][0] / orig_w, corners["bottom_left"][1] / orig_h],
    ], dtype=np.float32)

    xs = np.arange(size, dtype=np.float32)[None, :]
    ys = np.arange(size, dtype=np.float32)[:, None]

    for idx, (x_norm, y_norm) in enumerate(coords):
        cx = x_norm * (size - 1)
        cy = y_norm * (size - 1)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        heatmaps[idx] = np.exp(-d2 / (2 * sigma ** 2))

    return heatmaps, coords.reshape(-1)


def argmax_decode(heatmaps):
    b, c, h, w = heatmaps.shape
    flat = heatmaps.view(b, c, -1)
    idx = flat.argmax(dim=-1)
    ys = idx // w
    xs = idx % w
    coords = torch.stack([
        xs.float() / max(1, w - 1),
        ys.float() / max(1, h - 1),
    ], dim=-1)
    return coords.view(b, c * 2)


def soft_argmax_decode(heatmaps, beta=20.0):
    b, c, h, w = heatmaps.shape
    flat = heatmaps.view(b, c, -1)
    probs = torch.softmax(flat * beta, dim=-1)

    xs = torch.linspace(0.0, 1.0, w, device=heatmaps.device)
    ys = torch.linspace(0.0, 1.0, h, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    x = torch.sum(probs * grid_x, dim=-1)
    y = torch.sum(probs * grid_y, dim=-1)
    coords = torch.stack([x, y], dim=-1)
    return coords.view(b, c * 2)


class HeatmapCornerDataset(Dataset):
    def __init__(self, annotations_path, images_root, split="train",
                 input_mode="hybrid", augment=False, cache_squares=True):
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.input_mode = input_mode
        self.requires_heatmap = input_mode == "hybrid"
        self.cache_squares = cache_squares and self.requires_heatmap
        self.square_cache = {}
        self.augment = augment

        self.images = {img["id"]: img for img in data["images"]}
        self.corner_map = {
            corner["image_id"]: corner["corners"]
            for corner in data["annotations"]["corners"]
        }

        split_ids = data["splits"]["chessred2k"][split]["image_ids"]
        sample_ids = [
            img_id for img_id in split_ids
            if img_id in self.images and img_id in self.corner_map
        ]

        valid_ids = []
        for img_id in sample_ids:
            img_path = self.images_root / self.images[img_id]["path"]
            if img_path.exists():
                valid_ids.append(img_id)
        self.sample_ids = sorted(valid_ids)
        print(f"Heatmap corner dataset {split}/{input_mode}: {len(self.sample_ids)} samples", flush=True)

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

        inputs = make_input_tensor(
            image_bgr,
            square_centers=square_centers,
            input_mode=self.input_mode,
            out_size=IMG_SIZE,
        )
        target_heatmaps, target_coords = make_target_heatmaps(corners, orig_w, orig_h)
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(target_heatmaps),
            torch.from_numpy(target_coords),
        )


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


class HeatmapCornerModel(nn.Module):
    def __init__(self, input_channels=3):
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
        self.layer1 = backbone.layer1   # 96x96
        self.layer2 = backbone.layer2   # 48x48
        self.layer3 = backbone.layer3   # 24x24
        self.layer4 = backbone.layer4   # 12x12

        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 96)
        self.head = nn.Conv2d(96, 4, kernel_size=1)

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

        heatmaps = self.head(y)
        return {
            "heatmaps": heatmaps,
            "coords_soft": soft_argmax_decode(heatmaps),
            "coords_argmax": argmax_decode(heatmaps),
        }


def corner_distance(pred, target):
    pred = pred.view(-1, 4, 2)
    target = target.view(-1, 4, 2)
    dists = torch.sqrt(((pred - target) ** 2).sum(dim=2))
    return dists.mean()


def train_epoch(model, loader, heatmap_loss_fn, coord_loss_fn, optimizer, device, epoch_str="", status_file=None):
    model.train()
    total_loss = 0.0
    total_dist = 0.0
    total = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)

    for i, (images, target_heatmaps, target_coords) in enumerate(loader):
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        heatmap_loss = heatmap_loss_fn(torch.sigmoid(outputs["heatmaps"]), target_heatmaps)
        coord_loss = coord_loss_fn(outputs["coords_soft"], target_coords)
        loss = heatmap_loss + 0.1 * coord_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dist += corner_distance(outputs["coords_argmax"], target_coords).item() * images.size(0)
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


def validate(model, loader, heatmap_loss_fn, coord_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dist = 0.0
    total = 0
    all_dists = []

    with torch.no_grad():
        for images, target_heatmaps, target_coords in loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)
            target_coords = target_coords.to(device)

            outputs = model(images)
            heatmap_loss = heatmap_loss_fn(torch.sigmoid(outputs["heatmaps"]), target_heatmaps)
            coord_loss = coord_loss_fn(outputs["coords_soft"], target_coords)
            loss = heatmap_loss + 0.1 * coord_loss

            total_loss += loss.item() * images.size(0)
            pred = outputs["coords_argmax"].view(-1, 4, 2)
            targ = target_coords.view(-1, 4, 2)
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
    model = HeatmapCornerModel(input_channels=input_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class ExportWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            outputs = self.wrapped(x)
            return outputs["coords_argmax"], torch.sigmoid(outputs["heatmaps"])

    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, input_channels, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["corners", "heatmaps"],
        dynamic_axes={"image": {0: "batch"}, "corners": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=18,
        dynamo=False,
        external_data=False,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX to {output_path} ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train corner heatmap model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=str(BASE_DIR / "models"))
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODE_TO_CHANNELS.keys()), default="hybrid")
    args = parser.parse_args()

    annotations_path, images_root = resolve_data_paths(args)
    models_dir = Path(args.models_dir)
    input_channels = INPUT_MODE_TO_CHANNELS[args.input_mode]

    print(f"Annotations: {annotations_path}")
    print(f"Images root:  {images_root}")
    print(f"Input mode:   {args.input_mode} ({input_channels} channels)")

    device = get_device()
    train_dataset = HeatmapCornerDataset(
        annotations_path,
        images_root,
        split="train",
        input_mode=args.input_mode,
        augment=True,
        cache_squares=True,
    )
    val_dataset = HeatmapCornerDataset(
        annotations_path,
        images_root,
        split="val",
        input_mode=args.input_mode,
        augment=False,
        cache_squares=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = HeatmapCornerModel(input_channels=input_channels)
    best_val_dist = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_dist = min(best_val_dist, ckpt.get("val_dist", best_val_dist))

    model = model.to(device)
    heatmap_loss_fn = nn.MSELoss()
    coord_loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining HEATMAP corner detector for {args.epochs} epochs on {device}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE} x {input_channels}ch")
    print("=" * 60, flush=True)

    no_improve = 0
    for epoch in range(args.epochs):
        start = time.time()
        epoch_str = f"[{epoch + 1}/{args.epochs}]"
        train_loss, train_dist = train_epoch(
            model, train_loader, heatmap_loss_fn, coord_loss_fn, optimizer, device, epoch_str, args.status_file
        )
        val_loss, val_dist, val_max, per_corner = validate(
            model, val_loader, heatmap_loss_fn, coord_loss_fn, device
        )
        scheduler.step()
        elapsed = time.time() - start

        corner_names = ["TL", "TR", "BR", "BL"]
        per_corner_str = " ".join(f"{name}={dist:.4f}" for name, dist in zip(corner_names, per_corner))
        summary = (
            f"Epoch {epoch + 1}/{args.epochs} ({elapsed:.0f}s) | "
            f"Train: loss={train_loss:.4f} dist={train_dist:.4f} | "
            f"Val: loss={val_loss:.4f} dist={val_dist:.4f} max={val_max:.4f} | "
            f"{per_corner_str}"
        )
        print(summary, flush=True)
        write_status(args.status_file, summary)

        if val_dist < best_val_dist:
            best_val_dist = val_dist
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dist": val_dist,
                    "img_size": IMG_SIZE,
                    "heatmap_size": HEATMAP_SIZE,
                    "input_mode": args.input_mode,
                    "input_channels": input_channels,
                },
                models_dir / "best_corner_heatmap.pt",
            )
            print(f"  -> New best! dist={val_dist:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print("  Early stopping: no improvement for 7 epochs", flush=True)
                break

    print(f"\n{'=' * 60}")
    print(f"Done! Best mean corner distance: {best_val_dist:.4f}")
    print(f"  (On a 3072px image, {best_val_dist:.4f} ~ {best_val_dist * 3072:.0f}px per corner)")

    if args.export_onnx:
        print("\nExporting to ONNX...", flush=True)
        export_onnx(
            models_dir / "best_corner_heatmap.pt",
            input_channels=input_channels,
            output_path=models_dir / "corner_heatmap.onnx",
        )


if __name__ == "__main__":
    main()
