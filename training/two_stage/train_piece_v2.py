#!/usr/bin/env python3
"""Train piece classifier v2: cached warps + hflip aug + bigger batch."""

import sys
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
WORKTREE = Path("/home/johann/autoresearch/20260423-whole_board_classifier")
HARNESS_DIR = WORKTREE / "training" / "autoresearch_v3"
ANNOTATIONS = "/home/johann/ImagetoFEN/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN"
OUT_DIR = THIS_DIR / "run2"

LETTER_PIECE = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(time_budget_s=3600.0, batch_size=128, lr=5e-4, weight_decay=0.01, eval_every_s=180.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HARNESS_DIR))
    candidate = load_module("autoresearch_candidate", HARNESS_DIR / "candidate.py")
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")
    from piece_dataset import PieceCropDataset, NUM_PIECE_CLASSES
    from piece_model import build_piece_model

    defaults = candidate.get_defaults()
    train_groups = [("chessred2k", "train"), ("chess_dataset_recovered", "train"), ("synthetic", "train")]
    val_groups = [("chessred2k", "val"), ("chess_dataset_recovered", "val"), ("synthetic", "val")]

    def make_combined(groups, augment):
        ds_list = []
        for g, s in groups:
            bd = harness.BoardDataset(
                ANNOTATIONS, IMAGES_ROOT, group=g, split=s,
                candidate=candidate, candidate_defaults=defaults,
                input_mode=defaults["input_mode"], img_size=defaults["img_size"],
                augment=False, seed=1337,
            )
            ds_list.append(PieceCropDataset(bd, augment=augment))
        from torch.utils.data import ConcatDataset
        return ConcatDataset(ds_list)

    train_ds = make_combined(train_groups, augment=True)
    val_ds = make_combined(val_groups, augment=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_piece_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    # Estimate total steps generously; cosine to ~5% of initial.
    est_steps_per_s = 4.0
    total_steps = int(time_budget_s * est_steps_per_s)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.05)

    best_val = 0.0
    start = time.time()
    step = 0
    last_eval = start

    while time.time() - start < time_budget_s:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if time.time() - start >= time_budget_s:
                break
            if time.time() - last_eval >= eval_every_s:
                acc = evaluate(model, val_loader, device)
                elapsed = time.time() - start
                print(f"step={step:>6} elapsed={elapsed:.0f}s loss={loss.item():.4f} val_acc={acc:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if acc > best_val:
                    best_val = acc
                    torch.save({"model_state_dict": model.state_dict(), "val_acc": acc, "step": step},
                               OUT_DIR / "best.pt")
                    print(f"  -> saved best @ {acc:.4f}", flush=True)
                last_eval = time.time()

    acc = evaluate(model, val_loader, device)
    confusion = evaluate_confusion(model, val_loader, device)
    print(f"\nFINAL val_acc={acc:.4f}  best={best_val:.4f}", flush=True)
    print("\nConfusion (rows=GT, cols=pred):")
    print("    " + " ".join(f"{p:>5}" for p in LETTER_PIECE))
    for i in range(NUM_PIECE_CLASSES):
        row = " ".join(f"{int(confusion[i,j]):>5}" for j in range(NUM_PIECE_CLASSES))
        print(f" {LETTER_PIECE[i]}  {row}")
    np.save(OUT_DIR / "confusion.npy", confusion)
    if acc > best_val:
        torch.save({"model_state_dict": model.state_dict(), "val_acc": acc, "step": step},
                   OUT_DIR / "best.pt")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(1, total)


def evaluate_confusion(model, loader, device):
    from piece_dataset import NUM_PIECE_CLASSES
    model.eval()
    conf = np.zeros((NUM_PIECE_CLASSES, NUM_PIECE_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            for t, p in zip(y.numpy(), preds):
                conf[t, p] += 1
    return conf


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    main(time_budget_s=args.time_budget_s, batch_size=args.batch_size)
