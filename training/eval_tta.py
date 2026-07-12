#!/usr/bin/env python3
"""Evaluate the current best whole-board classifier with test-time augmentation.

TTA scheme: 8 views = {0, 90, 180, 270} CW rotations × {identity, hflip}.
Each view is run through the model independently; the resulting [13,8,8]
logits are un-rotated/un-flipped back to the canonical orientation, then
averaged in log-space (i.e. mean of logits ≈ geometric mean of softmax).
Rotation/flip of the input doesn't change piece-class semantics, so class
indices are preserved across views.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "training" / "autoresearch_v3"
CHECKPOINT = REPO_ROOT / "training" / "checkpoints" / "whole_board_0399b00.pt"
CANDIDATE_PY = REPO_ROOT / "training" / "checkpoints" / "whole_board_0399b00.candidate.py"
ANNOTATIONS = str(REPO_ROOT / "annotations.json")
IMAGES_ROOT = str(REPO_ROOT)

VAL_SPLITS = [
    ("chessred2k", "val"),
    ("chess_dataset_recovered", "val"),
    ("synthetic", "val"),
]
LETTER = ["b", "k", "n", "p", "q", "r", ".", "B", "K", "N", "P", "Q", "R"]
EMPTY = 6


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_views(images):
    """Identity + horizontal flip only.

    Training augmentation (candidate.augment_image) does NOT include
    rotation or flip, so any geometric TTA is out-of-distribution.
    Rotations + shifts both empirically break the model. hflip alone is
    the only candidate worth trying with the current checkpoint.
    """
    views = [(images, lambda lg: lg)]
    flipped = torch.flip(images, dims=(3,))
    views.append((flipped, lambda lg: torch.flip(lg, dims=(3,))))
    return views


def evaluate(model, loader, device, use_tta):
    model.eval()
    correct_cells = 0
    total_cells = 0
    per_board_err = []
    confusion = np.zeros((13, 13), dtype=np.int64)
    with torch.no_grad():
        for images, targets, _meta in loader:
            images = images.to(device)
            labels = targets["labels"].to(device)
            B = images.size(0)
            if use_tta:
                acc = None
                views = make_views(images)
                for view, un in views:
                    logits = model(view)["logits"]
                    aligned = un(logits)
                    acc = aligned if acc is None else acc + aligned
                logits = acc / len(views)
            else:
                logits = model(images)["logits"]
            preds = logits.argmax(dim=1)  # [B, 8, 8]
            correct = (preds == labels).view(B, -1).float()
            correct_cells += int(correct.sum().item())
            total_cells += B * 64
            per_board_err.append(1.0 - correct.mean(dim=1).cpu().numpy())
            for c in range(B):
                for r in range(8):
                    for cc in range(8):
                        confusion[int(labels[c, r, cc]), int(preds[c, r, cc])] += 1
    per_board_err = np.concatenate(per_board_err)
    return {
        "samples": len(per_board_err),
        "cell_acc": correct_cells / total_cells,
        "mean_dist": float(per_board_err.mean()),
        "max_dist": float(per_board_err.max()),
        "p95_dist": float(np.quantile(per_board_err, 0.95)),
        "board_err": float((per_board_err > 0).mean()),
        "confusion": confusion,
    }


def main():
    sys.path.insert(0, str(HARNESS_DIR))
    candidate = load_module("autoresearch_candidate", CANDIDATE_PY)
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = candidate.build_model(input_channels=3)
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    candidate.load_checkpoint(model, ckpt.get("model_state_dict", ckpt))
    model.eval().to(device)
    defaults = candidate.get_defaults()

    loaders = {}
    for group, split in VAL_SPLITS:
        ds = harness.BoardDataset(
            ANNOTATIONS, IMAGES_ROOT,
            group=group, split=split,
            candidate=candidate, candidate_defaults=defaults,
            input_mode=defaults["input_mode"], img_size=defaults["img_size"],
            augment=False, seed=1337,
        )
        if len(ds) == 0:
            print(f"Skipping empty split {group}:{split}")
            continue
        loaders[f"{group}:{split}"] = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    print(f"\n{'split':<35} {'mode':>6} {'cell_acc':>9} {'mean_err':>9} {'max_err':>8} {'board_err':>10}")
    print("-" * 90)

    combined_per_board = {"baseline": [], "tta": []}
    for name, loader in loaders.items():
        for mode in ("baseline", "tta"):
            m = evaluate(model, loader, device, use_tta=(mode == "tta"))
            print(f"{name:<35} {mode:>6} {m['cell_acc']:>9.4f} {m['mean_dist']:>9.4f} {m['max_dist']:>8.4f} {m['board_err']:>10.4f}")
        # Re-run to keep per-board for combined view
        for mode in ("baseline", "tta"):
            with torch.no_grad():
                preds_per_board = []
                labels_per_board = []
                for images, targets, _meta in loader:
                    images = images.to(device)
                    labels = targets["labels"].to(device)
                    if mode == "tta":
                        acc = None
                        views = make_views(images)
                        for view, un in views:
                            logits = un(model(view)["logits"])
                            acc = logits if acc is None else acc + logits
                        logits = acc / len(views)
                    else:
                        logits = model(images)["logits"]
                    preds = logits.argmax(dim=1)
                    err = (preds != labels).view(preds.size(0), -1).float().mean(dim=1)
                    combined_per_board[mode].append(err.cpu().numpy())
        print()

    print("-" * 90)
    for mode in ("baseline", "tta"):
        all_err = np.concatenate(combined_per_board[mode])
        print(f"COMBINED ({mode:<8}) cell_acc={1-all_err.mean():.4f}  mean_err={all_err.mean():.4f}  "
              f"max={all_err.max():.4f}  p95={np.quantile(all_err,0.95):.4f}  board_err={(all_err>0).mean():.4f}")


if __name__ == "__main__":
    main()
