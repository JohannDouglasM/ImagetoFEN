#!/usr/bin/env python3
"""Find top-3 worst images per val split for the current best whole-board
classifier and render warped boards with predicted-piece letters overlaid."""

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
CHECKPOINT = HARNESS_DIR / "runs" / "20260423-whole_board_classifier" / "artifacts" / "20260425T091625Z_0399b00" / "best.pt"
CANDIDATE_PY = HARNESS_DIR / "runs" / "20260423-whole_board_classifier" / "artifacts" / "20260425T091625Z_0399b00" / "candidate.py"
ANNOTATIONS = "/home/johann/ImagetoFEN/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN"
OUT_DIR = Path("/home/johann/ImagetoFEN/training/worst_predictions")

VAL_SPLITS = [
    ("chessred2k", "val"),
    ("chess_dataset_recovered", "val"),
    ("synthetic", "val"),
]
TOP_K = 3
RENDER_BOARD_PX = 800

CLASS_TO_LETTER = ["b", "k", "n", "p", "q", "r", ".", "B", "K", "N", "P", "Q", "R"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HARNESS_DIR))

    candidate = load_module("autoresearch_candidate", CANDIDATE_PY)
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = candidate.build_model(input_channels=3)
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    candidate.load_checkpoint(model, state)
    model.eval().to(device)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    defaults = candidate.get_defaults()

    overall = []  # for global ranking print
    per_split_results = {}

    for group, split in VAL_SPLITS:
        ds = harness.BoardDataset(
            ANNOTATIONS,
            IMAGES_ROOT,
            group=group,
            split=split,
            candidate=candidate,
            candidate_defaults=defaults,
            input_mode=defaults["input_mode"],
            img_size=defaults["img_size"],
            augment=False,
            seed=1337,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        records = []
        idx = 0
        with torch.no_grad():
            for images, targets, meta in loader:
                images = images.to(device)
                labels = targets["labels"].to(device)
                outputs = model(images)
                preds = outputs["logits"].argmax(dim=1)  # [B, 8, 8]
                wrong = (preds != labels).view(preds.shape[0], -1).sum(dim=1)
                for i in range(preds.shape[0]):
                    records.append({
                        "group": meta["group"][i],
                        "split": meta["split"][i],
                        "image_id": int(meta["image_id"][i]),
                        "path": meta["path"][i],
                        "wrong": int(wrong[i].item()),
                        "preds": preds[i].detach().cpu().numpy(),
                        "labels": labels[i].detach().cpu().numpy(),
                        "ds_idx": idx + i,
                    })
                idx += preds.shape[0]

        records.sort(key=lambda r: -r["wrong"])
        per_split_results[group] = (ds, records)
        for r in records[:TOP_K]:
            overall.append(r)

        print(f"\n{group}:{split} — n={len(records)}, mean_wrong/64={np.mean([r['wrong'] for r in records])/64:.4f}")
        for rank, r in enumerate(records[:TOP_K], 1):
            print(f"  {rank}. id={r['image_id']:>5}  wrong={r['wrong']}/64  path={Path(r['path']).name}")

    # Render each top-K
    for group, (ds, records) in per_split_results.items():
        for rank, r in enumerate(records[:TOP_K], 1):
            warped = ds._load_base_sample(ds.items[r["ds_idx"]])
            out_path = OUT_DIR / f"{group}_rank{rank}_id{r['image_id']}_wrong{r['wrong']}.png"
            render(warped, r["preds"], r["labels"], out_path, title=f"{group} #{rank}  image_id={r['image_id']}  wrong={r['wrong']}/64")
            print(f"  wrote {out_path}")

    # Also a grid summary
    write_grid(per_split_results, OUT_DIR / "summary_grid.png")
    print(f"\nSummary grid: {OUT_DIR / 'summary_grid.png'}")


def render(warped_bgr, preds, labels, out_path, title=""):
    """Draw the warped board with predicted piece letter on each square.
    Letter color: green if correct, red if wrong (with GT shown in subscript)."""
    img = cv2.resize(warped_bgr, (RENDER_BOARD_PX, RENDER_BOARD_PX))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell = RENDER_BOARD_PX // 8
    title_h = 60
    legend_h = 40

    big = _try_font(int(cell * 0.55))
    small = _try_font(int(cell * 0.28))
    title_font = _try_font(28)

    for r in range(8):
        for c in range(8):
            x0, y0 = c * cell, r * cell
            pred_id = int(preds[r, c])
            label_id = int(labels[r, c])
            correct = pred_id == label_id
            pred_ch = CLASS_TO_LETTER[pred_id]
            label_ch = CLASS_TO_LETTER[label_id]

            # Background tint to make wrong cells obvious
            if not correct:
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(255, 60, 60, 90))

            # Skip drawing for empty predictions on correct empty squares (less clutter)
            if correct and pred_id == 6:
                continue

            text_color = (40, 220, 90, 255) if correct else (255, 230, 60, 255)
            outline = (0, 0, 0, 255)

            # Center predicted letter
            tw, th = _text_size(draw, pred_ch, big)
            tx = x0 + (cell - tw) // 2
            ty = y0 + (cell - th) // 2 - 4
            _draw_outlined(draw, (tx, ty), pred_ch, big, text_color, outline)

            # If wrong, show GT in bottom-right
            if not correct:
                gt_text = f"({label_ch})"
                tw2, th2 = _text_size(draw, gt_text, small)
                _draw_outlined(
                    draw,
                    (x0 + cell - tw2 - 4, y0 + cell - th2 - 4),
                    gt_text,
                    small,
                    (255, 255, 255, 255),
                    outline,
                )

    pil = Image.alpha_composite(pil, overlay)

    # Pad with title and legend
    final = Image.new("RGB", (RENDER_BOARD_PX, RENDER_BOARD_PX + title_h + legend_h), (20, 20, 20))
    final.paste(pil.convert("RGB"), (0, title_h))
    fdraw = ImageDraw.Draw(final)
    fdraw.text((10, 12), title, fill=(255, 255, 255), font=title_font)
    legend_y = title_h + RENDER_BOARD_PX + 8
    fdraw.text(
        (10, legend_y),
        "green=correct prediction   yellow=wrong prediction (GT in white parens)",
        fill=(220, 220, 220),
        font=_try_font(20),
    )
    final.save(out_path)


def _try_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
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
    for dx in (-2, -1, 1, 2):
        for dy in (-2, -1, 1, 2):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def write_grid(per_split_results, out_path):
    panels = []
    cell_px = 360
    title_h = 30
    for group, (ds, records) in per_split_results.items():
        for rank, r in enumerate(records[:TOP_K], 1):
            warped = ds._load_base_sample(ds.items[r["ds_idx"]])
            tile = _make_tile(
                warped,
                r["preds"],
                r["labels"],
                title=f"{group} #{rank}  wrong={r['wrong']}/64",
                size=cell_px,
                title_h=title_h,
            )
            panels.append((group, tile))

    cols = TOP_K
    rows = len(VAL_SPLITS)
    W = cols * cell_px
    H = rows * (cell_px + title_h)
    grid = Image.new("RGB", (W, H), (20, 20, 20))
    for idx, (_g, tile) in enumerate(panels):
        c = idx % cols
        r = idx // cols
        grid.paste(tile, (c * cell_px, r * (cell_px + title_h)))
    grid.save(out_path)


def _make_tile(warped_bgr, preds, labels, title, size, title_h):
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
            pred_ch = CLASS_TO_LETTER[pred_id]
            label_ch = CLASS_TO_LETTER[label_id]
            if not correct:
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(255, 60, 60, 90))
            if correct and pred_id == 6:
                continue
            text_color = (40, 220, 90, 255) if correct else (255, 230, 60, 255)
            outline = (0, 0, 0, 255)
            tw, th = _text_size(draw, pred_ch, big)
            tx = x0 + (cell - tw) // 2
            ty = y0 + (cell - th) // 2 - 4
            _draw_outlined(draw, (tx, ty), pred_ch, big, text_color, outline)
            if not correct:
                gt_text = f"({label_ch})"
                tw2, th2 = _text_size(draw, gt_text, small)
                _draw_outlined(
                    draw,
                    (x0 + cell - tw2 - 4, y0 + cell - th2 - 4),
                    gt_text,
                    small,
                    (255, 255, 255, 255),
                    outline,
                )
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    out = Image.new("RGB", (size, size + title_h), (20, 20, 20))
    out.paste(pil, (0, title_h))
    d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(255, 255, 255), font=_try_font(18))
    return out


if __name__ == "__main__":
    main()
