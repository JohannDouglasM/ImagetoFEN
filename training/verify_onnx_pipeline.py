#!/usr/bin/env python3
"""Reference run of the exported ONNX two-stage pipeline on real val images.

Measures, per image and averaged:
- corner error (normalized, vs GT corners)
- cell accuracy of the whole-board classifier fed with DETECTED-corner warps
  (the published 99.09% uses GT-corner warps; this is the honest app-side number)

Two variants:
- fullres: warp sampled from the original image
- app1024: image first downscaled to max side 1024 (what the app will do)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "assets" / "models"
IMAGES_ROOT = REPO / "valsample"

CLASS_LAYOUT = ["b", "k", "n", "p", "q", "r", None, "B", "K", "N", "P", "Q", "R"]
CATEGORY_NAME_TO_CLASS_ID = {
    "white-pawn": 10, "white-rook": 12, "white-knight": 9, "white-bishop": 7,
    "white-queen": 11, "white-king": 8,
    "black-pawn": 3, "black-rook": 5, "black-knight": 2, "black-bishop": 0,
    "black-queen": 4, "black-king": 1,
    "empty": 6,
}
EMPTY = 6

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def corner_input(image_bgr):
    resized = cv2.resize(image_bgr, (384, 384))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 80, 200).astype(np.float32) / 255.0
    return np.stack([gray, edges])[None]


def warp(image_bgr, corners_px, out_size=256):
    src = np.array(corners_px, dtype=np.float32)
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(image_bgr, H, (out_size, out_size))


def board_input(warped_bgr):
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((np.transpose(rgb, (2, 0, 1)) - MEAN) / STD)[None]


def pos_to_rowcol(pos):
    return 7 - (int(pos[1]) - 1), ord(pos[0]) - ord("a")


def main():
    ann = json.load(open(REPO / "annotations.json"))
    by_id = {im["id"]: im for im in ann["images"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    corner_map = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    piece_map = {}
    for p in ann["annotations"]["pieces"]:
        piece_map.setdefault(p["image_id"], {})[p["chessboard_position"]] = \
            CATEGORY_NAME_TO_CLASS_ID.get(cats.get(p["category_id"], "empty"), EMPTY)

    sample_paths = sorted(IMAGES_ROOT.rglob("*.jpg"))
    path_to_id = {im["path"]: im["id"] for im in ann["images"]}

    corner_sess = ort.InferenceSession(str(MODELS / "corner_unet.onnx"), providers=["CPUExecutionProvider"])
    board_sess = ort.InferenceSession(str(MODELS / "whole_board.onnx"), providers=["CPUExecutionProvider"])

    stats = {"fullres": [], "app1024": []}
    corner_errs = []
    print(f"{'image':<28} {'corner_err%':>11} {'acc_full':>9} {'acc_1024':>9}")
    for path in sample_paths:
        rel = str(path.relative_to(IMAGES_ROOT))
        image_id = path_to_id[rel]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]

        gt = corner_map[image_id]
        gt_px = np.array([gt["top_left"], gt["top_right"], gt["bottom_right"], gt["bottom_left"]],
                         dtype=np.float32)

        labels = np.full((8, 8), EMPTY, dtype=np.int64)
        for pos, cid in piece_map.get(image_id, {}).items():
            r, c = pos_to_rowcol(pos)
            labels[r, c] = cid

        variants = {}
        scale = 1024.0 / max(h, w)
        small = cv2.resize(img, (round(w * scale), round(h * scale)))
        variants["fullres"] = img
        variants["app1024"] = small

        row = {}
        for name, im in variants.items():
            ih, iw = im.shape[:2]
            coords = corner_sess.run(["coords"], {"input": corner_input(im)})[0][0]
            corners_px = [(coords[2 * i] * iw, coords[2 * i + 1] * ih) for i in range(4)]
            if name == "fullres":
                diag = np.hypot(w, h)
                err = np.mean([np.hypot(px - gx, py - gy)
                               for (px, py), (gx, gy) in zip(corners_px, gt_px)]) / diag
                corner_errs.append(err)
            warped = warp(im, corners_px)
            logits = board_sess.run(["logits"], {"input": board_input(warped)})[0][0]
            preds = logits.argmax(axis=0)
            acc = float((preds == labels).mean())
            stats[name].append(acc)
            row[name] = acc

        print(f"{rel.split('/')[-1]:<28} {corner_errs[-1]*100:>10.3f}% {row['fullres']:>9.4f} {row['app1024']:>9.4f}")

    print("-" * 60)
    print(f"mean corner err: {np.mean(corner_errs)*100:.3f}% of diagonal")
    for name, accs in stats.items():
        print(f"mean cell acc [{name}]: {np.mean(accs):.4f}")


if __name__ == "__main__":
    sys.exit(main())
