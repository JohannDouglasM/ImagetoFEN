#!/usr/bin/env python3
"""Aggregate failure analysis for the current best whole-board classifier."""

import sys
import importlib.util
from collections import Counter
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

    all_confusion = np.zeros((13, 13), dtype=np.int64)  # all splits combined
    per_split = {}

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
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        confusion = np.zeros((13, 13), dtype=np.int64)
        row_err = np.zeros(8, dtype=np.int64)
        row_total = np.zeros(8, dtype=np.int64)
        n_imgs = 0
        with torch.no_grad():
            for images, targets, meta in loader:
                images = images.to(device)
                labels = targets["labels"].to(device)
                preds = model(images)["logits"].argmax(dim=1)
                p = preds.cpu().numpy()
                l = labels.cpu().numpy()
                B = p.shape[0]
                n_imgs += B
                for c in range(B):
                    for r in range(8):
                        for cc in range(8):
                            confusion[l[c, r, cc], p[c, r, cc]] += 1
                    row_err += (p[c] != l[c]).sum(axis=1)
                    row_total += 8
        per_split[group] = {"confusion": confusion, "row_err": row_err, "row_total": row_total, "n": n_imgs}
        all_confusion += confusion

    # Print per-split + combined
    print("\n" + "="*70)
    print("CONFUSION (rows=GT, cols=pred). Top off-diagonal entries.")
    print("="*70)
    for name, d in list(per_split.items()) + [("COMBINED", {"confusion": all_confusion})]:
        c = d["confusion"]
        print(f"\n--- {name} ---")
        total = c.sum()
        diag = np.trace(c)
        print(f"cell acc: {diag/total:.4f}  ({diag}/{total})")
        # Per-class recall (rows)
        row_sums = c.sum(axis=1)
        print("per-class recall (GT -> correct):")
        for i in range(13):
            if row_sums[i] == 0:
                continue
            print(f"  {LETTER[i]:>2}  n={row_sums[i]:>5}  recall={c[i,i]/row_sums[i]:.4f}")
        # Top off-diagonal confusions
        flat = []
        for i in range(13):
            for j in range(13):
                if i != j and c[i, j] > 0:
                    flat.append((c[i, j], i, j))
        flat.sort(reverse=True)
        print("top off-diagonals (gt -> pred : count):")
        for cnt, gt, pr in flat[:10]:
            print(f"  {LETTER[gt]:>2} -> {LETTER[pr]:<2} : {cnt}")
        # FP/FN on empty
        fn_piece = sum(c[i, EMPTY] for i in range(13) if i != EMPTY)  # GT piece predicted empty
        fp_piece = sum(c[EMPTY, j] for j in range(13) if j != EMPTY)  # GT empty predicted piece
        wrong_piece_to_piece = sum(c[i, j] for i in range(13) for j in range(13) if i != j and i != EMPTY and j != EMPTY)
        total_errors = (c.sum() - np.trace(c))
        if total_errors:
            print(f"errors: missed-piece={fn_piece} ({fn_piece/total_errors:.0%})  "
                  f"hallucinated-piece={fp_piece} ({fp_piece/total_errors:.0%})  "
                  f"piece->wrong-piece={wrong_piece_to_piece} ({wrong_piece_to_piece/total_errors:.0%})")

    print("\n" + "="*70)
    print("PER-ROW ERROR RATE (row 0 = rank 8 = far side from camera)")
    print("="*70)
    for name, d in per_split.items():
        if "row_err" not in d:
            continue
        rates = d["row_err"] / np.maximum(d["row_total"], 1)
        print(f"\n--- {name} ---")
        for r in range(8):
            bar = "#" * int(rates[r] * 200)
            print(f"  row {r}: {rates[r]:.4f}  {bar}")

    # Specifically inspect id=12875
    print("\n" + "="*70)
    print("OUTLIER 12875 — predicted vs GT board")
    print("="*70)
    ds = harness.BoardDataset(
        ANNOTATIONS, IMAGES_ROOT,
        group="chess_dataset_recovered", split="val",
        candidate=candidate, candidate_defaults=defaults,
        input_mode=defaults["input_mode"], img_size=defaults["img_size"],
        augment=False, seed=1337,
    )
    for idx, item in enumerate(ds.items):
        if item.image_id == 12875:
            sample = ds[idx]
            img, tgt, meta = sample
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(device))["logits"]
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            label = tgt["labels"].numpy()
            print(f"path: {meta['path']}")
            print(f"GT n_pieces (non-empty): {(label != EMPTY).sum()}")
            print(f"Pred n_pieces (non-empty): {(pred != EMPTY).sum()}")
            print("\nGT:")
            for r in range(8):
                print("  " + " ".join(LETTER[label[r, c]] for c in range(8)))
            print("\nPred:")
            for r in range(8):
                print("  " + " ".join(LETTER[pred[r, c]] for c in range(8)))
            break


if __name__ == "__main__":
    main()
