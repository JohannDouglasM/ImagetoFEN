#!/usr/bin/env python3
"""Sanity-check the warp+crop pipeline on random images from each dataset.

Picks 3 random images from chessred2k and 3 from chess-dataset/labeled_originals,
runs the best corner detector, warps the board, crops all 64 squares, and saves
a visualization per image showing: original with corners, warped board, and
the 8x8 grid of cropped squares.
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

AUTORESEARCH_ROOT = Path("/home/johann/autoresearch/20260331-unet_dual_head")
sys.path.insert(0, str(AUTORESEARCH_ROOT / "training" / "autoresearch_v3"))
sys.path.insert(0, str(AUTORESEARCH_ROOT / "training"))

import candidate

BEST_CHECKPOINT = AUTORESEARCH_ROOT / "training/autoresearch_v3/runs/20260331-unet_dual_head/artifacts/20260406T122113Z_7e15e8e/best.pt"

CHESSRED2K_ROOT = Path("/home/johann/ImagetoFEN/chessred2k/images")
CHESS_DATASET_ROOT = Path("/home/johann/ImagetoFEN/chess-dataset/labeled_originals")
SYNTHETIC_ROOT = Path("/home/johann/ImagetoFEN/training/data")

OUT_DIR = Path("/home/johann/ImagetoFEN/training/crop_test_output")
SEED = 20260411
N_PER_DATASET = 3

# Must match prepare_squares.py
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)


def sort_corner_points(points):
    points = points[points[:, 1].argsort()]
    points[:2] = points[:2][points[:2, 0].argsort()]
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]
    return points


def warp_board(img_rgb, corners):
    src = sort_corner_points(corners.astype(np.float32))
    dst = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img_rgb, M, (IMG_SIZE, IMG_SIZE)), src


def crop_square(warped, row, col):
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)
    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
    cropped = warped[y1:y2, x1:x2]
    if col < 4:
        cropped = cv2.flip(cropped, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    result[OUT_HEIGHT - h:, :w] = cropped
    return result


def load_model():
    model = candidate.build_model(input_channels=2)
    ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    candidate.load_checkpoint(model, ckpt["model_state_dict"])
    model.eval()
    return model


def predict_corners(model, img_bgr):
    h, w = img_bgr.shape[:2]
    x = candidate.make_input_tensor(img_bgr, input_mode="gray_edges", out_size=384)
    x = torch.from_numpy(x).unsqueeze(0).float()
    with torch.no_grad():
        out = model(x)
        coords = candidate.decode_coords(out)  # [1, 8] normalized
    pts = coords[0].numpy().reshape(4, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts  # TL, TR, BR, BL from heatmap order


def crop_square_clean(warped, row, col, margin_frac=0.15):
    """Tight crop of a single square with a small margin. No mirror, no padding.

    margin_frac is the extra fraction of SQUARE_SIZE added on each side.
    """
    m = int(SQUARE_SIZE * margin_frac)
    x1 = int(MARGIN + SQUARE_SIZE * col) - m
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1)) + m
    y1 = int(MARGIN + SQUARE_SIZE * row) - m
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1)) + m
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
    return warped[y1:y2, x1:x2]


def make_clean_grid(warped_rgb, tile_size=80, gap=4, border=1):
    """Visualization: tight square crops in a1..h8 layout, no mirror, no padding."""
    bg = 40
    border_color = (90, 90, 90)
    t = tile_size + 2 * border
    grid_h = 8 * t + 9 * gap
    grid_w = 8 * t + 9 * gap
    grid = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            sq_rgb = crop_square_clean(warped_rgb, row, col)
            sq_bgr = cv2.cvtColor(sq_rgb, cv2.COLOR_RGB2BGR)
            sq_resized = cv2.resize(sq_bgr, (tile_size, tile_size))
            tile = np.full((t, t, 3), border_color, dtype=np.uint8)
            tile[border:border + tile_size, border:border + tile_size] = sq_resized
            y = gap + row * (t + gap)
            x = gap + col * (t + gap)
            grid[y:y + t, x:x + t] = tile
    return grid


def make_grid_image(squares_by_pos, gap=6, border=2):
    """Assemble 8x8 grid with visible gaps + borders around each tile.

    Tiles are shown exactly as the square classifier sees them (mirrored for
    col<4, with top padding and width-overlap context). The gap makes clear
    that each tile is an independent crop, not a continuous board.
    """
    bg = 40  # dark gray background
    border_color = (90, 90, 90)
    tile_h = OUT_HEIGHT + 2 * border
    tile_w = OUT_WIDTH + 2 * border

    grid_h = 8 * tile_h + 9 * gap
    grid_w = 8 * tile_w + 9 * gap
    grid = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            sq = squares_by_pos[row][col]
            tile = np.full((tile_h, tile_w, 3), border_color, dtype=np.uint8)
            tile[border:border + OUT_HEIGHT, border:border + OUT_WIDTH] = sq
            y = gap + row * (tile_h + gap)
            x = gap + col * (tile_w + gap)
            grid[y:y + tile_h, x:x + tile_w] = tile
    return grid


def annotate_original(img_bgr, corners_px, max_dim=900):
    img = img_bgr.copy()
    ordered = sort_corner_points(corners_px.copy().astype(np.float32))
    cv2.polylines(img, [ordered.astype(np.int32)], True, (0, 255, 0), 4)
    for i, (x, y) in enumerate(ordered):
        cv2.circle(img, (int(x), int(y)), 12, (0, 0, 255), -1)
        cv2.putText(img, ["TL", "TR", "BR", "BL"][i], (int(x) + 16, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def pad_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


def process_image(model, img_path, out_path, label):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [!] Could not read {img_path}")
        return
    corners_px = predict_corners(model, img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    warped_rgb, _ = warp_board(img_rgb, corners_px)
    warped_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)

    squares_by_pos = [[None] * 8 for _ in range(8)]
    for row in range(8):
        for col in range(8):
            sq_rgb = crop_square(warped_rgb, row, col)
            squares_by_pos[row][col] = cv2.cvtColor(sq_rgb, cv2.COLOR_RGB2BGR)

    grid = make_grid_image(squares_by_pos)
    clean_grid = make_clean_grid(warped_rgb)
    orig_annotated = annotate_original(img_bgr, corners_px)

    # Top row: original + warped, same height
    top_h = 800
    orig_top = pad_to_height(orig_annotated, top_h)
    warped_top = pad_to_height(warped_bgr, top_h)
    sep = np.full((top_h, 12, 3), 255, dtype=np.uint8)
    top_row = np.hstack([orig_top, sep, warped_top])

    # Put the classifier-format grid next to the clean grid (side by side)
    grids_h = max(grid.shape[0], clean_grid.shape[0])
    def pad_v(img, h):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 40, dtype=np.uint8)
        return np.vstack([img, pad])
    grids_row = np.hstack([
        pad_v(clean_grid, grids_h),
        np.full((grids_h, 20, 3), 40, dtype=np.uint8),
        pad_v(grid, grids_h),
    ])

    total_w = max(top_row.shape[1], grids_row.shape[1])
    if top_row.shape[1] < total_w:
        pad = np.full((top_h, total_w - top_row.shape[1], 3), 40, dtype=np.uint8)
        top_row = np.hstack([top_row, pad])
    if grids_row.shape[1] < total_w:
        left = (total_w - grids_row.shape[1]) // 2
        right = total_w - grids_row.shape[1] - left
        lp = np.full((grids_row.shape[0], left, 3), 40, dtype=np.uint8)
        rp = np.full((grids_row.shape[0], right, 3), 40, dtype=np.uint8)
        grids_row = np.hstack([lp, grids_row, rp])

    gap_row = np.full((16, total_w, 3), 40, dtype=np.uint8)

    banner = np.full((40, total_w, 3), 30, dtype=np.uint8)
    cv2.putText(banner, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    combined = np.vstack([banner, top_row, gap_row, grids_row])

    cv2.imwrite(str(out_path), combined)
    print(f"  -> {out_path.name}  ({img_path.name})")


def sample_chessred2k(n, rng):
    all_imgs = sorted(CHESSRED2K_ROOT.rglob("*.jpg"))
    return rng.sample(all_imgs, n)


def sample_chess_dataset(n, rng):
    all_imgs = sorted(CHESS_DATASET_ROOT.glob("*.JPG"))
    return rng.sample(all_imgs, n)


def sample_synthetic(n, rng):
    all_imgs = sorted(SYNTHETIC_ROOT.glob("render_*.png"))
    return rng.sample(all_imgs, n)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("Loading corner model...")
    model = load_model()

    print(f"\nchessred2k ({CHESSRED2K_ROOT}):")
    for i, p in enumerate(sample_chessred2k(N_PER_DATASET, rng), 1):
        out = OUT_DIR / f"chessred2k_{i}_{p.stem}.jpg"
        process_image(model, p, out, f"chessred2k | {p.name}")

    print(f"\nchess-dataset ({CHESS_DATASET_ROOT}):")
    for i, p in enumerate(sample_chess_dataset(N_PER_DATASET, rng), 1):
        out = OUT_DIR / f"chess-dataset_{i}_{p.stem}.jpg"
        process_image(model, p, out, f"chess-dataset | {p.name}")

    print(f"\nsynthetic ({SYNTHETIC_ROOT}):")
    for i, p in enumerate(sample_synthetic(N_PER_DATASET, rng), 1):
        out = OUT_DIR / f"synthetic_{i}_{p.stem}.jpg"
        process_image(model, p, out, f"synthetic | {p.name}")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
