#!/usr/bin/env python3
"""
Whole-board piece classifier candidate template.

Pipeline assumption: the dataset (fixed_harness_board.BoardDataset) has already
warped the input image to a square RGB board using GT corners, so this module
only sees a canonical top-down 8x8 board. That means:

- `make_input_tensor` expects an already-warped BGR board and just normalizes.
- `make_targets` takes a `pieces` dict (position_str -> class_id) instead of
  corner coordinates; the `corners` argument is kept for interface
  compatibility and ignored.
- `decode_coords` is kept for contract compatibility with fixed_harness_board
  but returns argmax cell predictions of shape [B, 64] int64 (row-major,
  row 0 = rank 8, col 0 = file a).

Class ordering matches `src/ml/inference.ts:CLASS_TO_PIECE` so the existing
FEN builder and on-device inference path work unchanged:
    0  b  black_bishop
    1  k  black_king
    2  n  black_knight
    3  p  black_pawn
    4  q  black_queen
    5  r  black_rook
    6     empty
    7  B  white_bishop
    8  K  white_king
    9  N  white_knight
    10 P  white_pawn
    11 Q  white_queen
    12 R  white_rook
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

DEFAULTS = {
    "candidate_name": "resnet18_whole_board_baseline",
    "img_size": 256,
    "input_mode": "rgb",
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "eval_interval_s": 300.0,
    "train_splits": "chessred2k:train,chess_dataset_recovered:train,synthetic:train",
    "val_splits": "chessred2k:val,chess_dataset_recovered:val,synthetic:val,user:train",
    "report_splits": "chessred2k:val,chess_dataset_recovered:val,synthetic:val,user:train",
    "max_no_improve_evals": 8,
    "resume_candidates": [],
    "allow_legacy_resume_fallback": False,
}

INPUT_MODE_TO_CHANNELS = {"rgb": 3}

NUM_CLASSES = 13
EMPTY_CLASS_ID = 6

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def get_defaults():
    return dict(DEFAULTS)


def needs_square_centers(input_mode):
    return False


def find_empty_squares_for_image(image_bgr):
    return []


def make_input_tensor(image_bgr, square_centers=None, input_mode="rgb", out_size=256):
    if input_mode != "rgb":
        raise ValueError(f"Unsupported input mode: {input_mode}")
    h, w = image_bgr.shape[:2]
    if (h, w) != (out_size, out_size):
        image_bgr = cv2.resize(image_bgr, (out_size, out_size))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return image_rgb.astype(np.float32)


def augment_image(image_bgr, rng):
    img = image_bgr.astype(np.float32)
    alpha = float(rng.uniform(0.7, 1.3))
    beta = float(rng.uniform(-30.0, 30.0))
    img = np.clip(alpha * img + beta, 0, 255)

    if rng.random() < 0.4:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-8.0, 8.0))) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * float(rng.uniform(0.75, 1.25)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < 0.25:
        ksize = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    img = img.astype(np.uint8)

    if rng.random() < 0.3:
        h, w = img.shape[:2]
        cell = h // 8
        row = int(rng.integers(0, 8))
        col = int(rng.integers(0, 8))
        y0 = row * cell
        x0 = col * cell
        img[y0:y0 + cell, x0:x0 + cell] = int(rng.integers(0, 256))

    return img


def convert_conv1_weights(conv_weight, input_channels):
    if input_channels == conv_weight.shape[1]:
        return conv_weight
    mean_weight = conv_weight.mean(dim=1, keepdim=True)
    return mean_weight.repeat(1, input_channels, 1, 1) * (3.0 / float(input_channels))


class WholeBoardClassifier(nn.Module):
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
        self.head = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if x.shape[-2:] != (8, 8):
            x = F.adaptive_avg_pool2d(x, (8, 8))
        logits = self.head(x)
        return {"logits": logits}


def build_model(input_channels):
    return WholeBoardClassifier(input_channels)


def _pos_to_rowcol(pos):
    file_idx = ord(pos[0]) - ord("a")
    rank_idx = int(pos[1]) - 1
    return 7 - rank_idx, file_idx


def make_targets(corners, pieces, *, orig_w, orig_h, img_size, defaults):
    """Build an 8x8 int64 tensor of class IDs.

    `corners` is accepted for interface compatibility with CornerDataset-style
    harnesses and is ignored (the board is already warped upstream).
    `pieces` is a dict of position_str -> class_id produced by BoardDataset.
    """
    labels = np.full((8, 8), EMPTY_CLASS_ID, dtype=np.int64)
    for pos, class_id in pieces.items():
        row, col = _pos_to_rowcol(pos)
        labels[row, col] = class_id
    return {"labels": torch.from_numpy(labels)}


def compute_loss(outputs, targets):
    logits = outputs["logits"]
    labels = targets["labels"]
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        cell_acc = (preds == labels).float().mean().item()
    return loss, {
        "ce_loss": float(loss.detach().item()),
        "cell_acc": float(cell_acc),
    }


def decode_coords(outputs):
    """Return cell predictions as [B, 64] int64 row-major.

    Reused as the evaluation surface; fixed_harness_board.evaluate_loader
    interprets the result as per-cell class predictions instead of coords.
    """
    logits = outputs["logits"]
    return logits.argmax(dim=1).view(logits.shape[0], -1)


def create_optimizer(model, *, lr, weight_decay, resumed):
    effective_lr = lr * 0.1 if resumed else lr
    return optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )


def create_scheduler(optimizer, *, total_train_steps):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_train_steps),
    )


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
