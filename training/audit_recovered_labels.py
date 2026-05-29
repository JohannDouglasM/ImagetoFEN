#!/usr/bin/env python3
"""Audit chess_dataset_recovered for corner/label bugs.

Strategy: run the current best model over both train and val splits, rank
images by per-image error rate, and render the bottom decile with warped
board + predicted/GT overlay so the bugs are visible at a glance.
"""

import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

WORKTREE = Path("/home/johann/autoresearch/20260423-whole_board_classifier")
HARNESS_DIR = WORKTREE / "training" / "autoresearch_v3"
ART = HARNESS_DIR / "runs" / "20260423-whole_board_classifier" / "artifacts" / "20260425T091625Z_0399b00"
CHECKPOINT = ART / "best.pt"
CANDIDATE_PY = ART / "candidate.py"
ANNOTATIONS = "/home/johann/ImagetoFEN/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN"
OUT_DIR = Path("/home/johann/ImagetoFEN/training/recovered_audit")

LETTER = ["b", "k", "n", "p", "q", "r", ".", "B", "K", "N", "P", "Q", "R"]
EMPTY = 6


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HARNESS_DIR))
    candidate = load_module("autoresearch_candidate", CANDIDATE_PY)
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = candidate.build_model(input_channels=3)
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    candidate.load_checkpoint(model, ckpt.get("model_state_dict", ckpt))
    model.eval().to(device)

    defaults = candidate.get_defaults()

    all_records = []
    for split in ("train", "val"):
        ds = harness.BoardDataset(
            ANNOTATIONS, IMAGES_ROOT,
            group="chess_dataset_recovered", split=split,
            candidate=candidate, candidate_defaults=defaults,
            input_mode=defaults["input_mode"], img_size=defaults["img_size"],
            augment=False, seed=1337,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        idx = 0
        with torch.no_grad():
            for images, targets, meta in loader:
                images = images.to(device)
                labels = targets["labels"].to(device)
                preds = model(images)["logits"].argmax(dim=1)
                wrong = (preds != labels).view(preds.shape[0], -1).sum(dim=1)
                for i in range(preds.shape[0]):
                    all_records.append({
                        "split": split,
                        "image_id": int(meta["image_id"][i]),
                        "path": meta["path"][i],
                        "wrong": int(wrong[i].item()),
                        "preds": preds[i].cpu().numpy(),
                        "labels": labels[i].cpu().numpy(),
                        "ds": ds, "ds_idx": idx + i,
                    })
                idx += preds.shape[0]

    all_records.sort(key=lambda r: -r["wrong"])
    total = len(all_records)
    n_zero = sum(1 for r in all_records if r["wrong"] == 0)
    print(f"\nchess_dataset_recovered total: {total}  (zero-error: {n_zero}, non-zero: {total - n_zero})")
    print(f"\nWorst 30 images:")
    print(f"{'rank':>4} {'split':>6} {'id':>6} {'wrong':>6}  path")
    for r, rec in enumerate(all_records[:30], 1):
        print(f"{r:>4} {rec['split']:>6} {rec['image_id']:>6} {rec['wrong']:>6}  {Path(rec['path']).name}")

    # Render top-30 worst as a grid
    panels = []
    cell_px = 360
    title_h = 30
    for rank, rec in enumerate(all_records[:30], 1):
        warped = rec["ds"]._load_base_sample(rec["ds"].items[rec["ds_idx"]])
        tile = make_tile(warped, rec["preds"], rec["labels"],
                         title=f"#{rank} {rec['split']} id={rec['image_id']} wrong={rec['wrong']}/64",
                         size=cell_px, title_h=title_h)
        panels.append(tile)

    cols = 5
    rows = (len(panels) + cols - 1) // cols
    W = cols * cell_px
    H = rows * (cell_px + title_h)
    grid = Image.new("RGB", (W, H), (20, 20, 20))
    for i, tile in enumerate(panels):
        c = i % cols
        r = i // cols
        grid.paste(tile, (c * cell_px, r * (cell_px + title_h)))
    grid_path = OUT_DIR / "audit_grid_top30.png"
    grid.save(grid_path)
    print(f"\nGrid: {grid_path}")

    # Write candidate-bad TSV (anything with >=4 wrong is a candidate)
    threshold = 4
    suspects = [r for r in all_records if r["wrong"] >= threshold]
    tsv_path = OUT_DIR / f"suspects_wrong_ge_{threshold}.tsv"
    with open(tsv_path, "w") as f:
        f.write("split\timage_id\twrong\tpath\n")
        for r in suspects:
            f.write(f"{r['split']}\t{r['image_id']}\t{r['wrong']}\t{r['path']}\n")
    print(f"\nCandidate-bad list (wrong >= {threshold}): {len(suspects)} images -> {tsv_path}")


def _try_font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_outlined(draw, pos, text, font, fill, outline):
    x, y = pos
    for dx in (-1, 1):
        for dy in (-1, 1):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def make_tile(warped_bgr, preds, labels, title, size, title_h):
    img = cv2.resize(warped_bgr, (size, size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell = size // 8
    big = _try_font(int(cell * 0.55))
    small = _try_font(int(cell * 0.28))
    for r in range(8):
        for c in range(8):
            x0, y0 = c * cell, r * cell
            pred_id = int(preds[r, c])
            label_id = int(labels[r, c])
            correct = pred_id == label_id
            pred_ch = LETTER[pred_id]
            label_ch = LETTER[label_id]
            if not correct:
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(255, 60, 60, 90))
            if correct and pred_id == EMPTY:
                continue
            color = (40, 220, 90, 255) if correct else (255, 230, 60, 255)
            outline = (0, 0, 0, 255)
            tw, th = _text_size(draw, pred_ch, big)
            _draw_outlined(draw, (x0 + (cell - tw) // 2, y0 + (cell - th) // 2 - 4), pred_ch, big, color, outline)
            if not correct:
                gt = f"({label_ch})"
                tw2, th2 = _text_size(draw, gt, small)
                _draw_outlined(draw, (x0 + cell - tw2 - 4, y0 + cell - th2 - 4), gt, small, (255, 255, 255, 255), outline)
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    out = Image.new("RGB", (size, size + title_h), (20, 20, 20))
    out.paste(pil, (0, title_h))
    d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(255, 255, 255), font=_try_font(16))
    return out


if __name__ == "__main__":
    main()
