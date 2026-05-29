"""Per-cell crop dataset for the second-stage piece classifier.

Reads pre-warped 512x512 boards from a cache directory and extracts a tight
upward-extended crop per piece-bearing cell. Hflip augmentation is applied
50% of the time at training; piece classes are unchanged under hflip (a
black bishop stays a black bishop).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Class layout: 0..5 black (b,k,n,p,q,r), 7..12 white (B,K,N,P,Q,R).
WHOLE_TO_PIECE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                  7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11}
PIECE_TO_WHOLE = {v: k for k, v in WHOLE_TO_PIECE.items()}
NUM_PIECE_CLASSES = 12

CROP_SIZE = 96
WARP_BOARD_PX = 512
CELL_PX = WARP_BOARD_PX // 8  # 64
HEIGHT_INCREASE = 1.5

CACHE_DIR = Path(__file__).resolve().parent / "warp_cache"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def warp_board(image_bgr, corners, out_size=WARP_BOARD_PX):
    src = np.array(
        [corners["top_left"], corners["top_right"],
         corners["bottom_right"], corners["bottom_left"]],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
                   dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(image_bgr, H, (out_size, out_size))


def crop_cell(warped_bgr, row, col, out_size=CROP_SIZE):
    h, w = warped_bgr.shape[:2]
    cell = w // 8
    extra_top = int(cell * HEIGHT_INCREASE)
    x_center = col * cell + cell // 2
    y_bottom = (row + 1) * cell
    y_top = max(0, y_bottom - cell - extra_top)
    crop_h = y_bottom - y_top
    x0 = max(0, x_center - crop_h // 2)
    x1 = min(w, x0 + crop_h)
    if x1 - x0 < crop_h:
        x0 = max(0, x1 - crop_h)
    patch = warped_bgr[y_top:y_bottom, x0:x1]
    if patch.shape[0] != patch.shape[1]:
        side = max(patch.shape[0], patch.shape[1])
        padded = np.zeros((side, side, 3), dtype=patch.dtype)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = padded
    return cv2.resize(patch, (out_size, out_size))


def patch_to_tensor(patch_bgr):
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.astype(np.float32))


def _aug_color(img, rng):
    img = img.astype(np.float32)
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
    return img.astype(np.uint8)


class PieceCropDataset(Dataset):
    """One sample per (image, piece-cell). Reads pre-warped boards from disk."""

    def __init__(self, board_dataset, augment=False, cache_dir=CACHE_DIR):
        self.bd = board_dataset
        self.augment = augment
        self.cache_dir = Path(cache_dir)
        self.rng = np.random.default_rng(1337)
        self.index = []  # list of (image_id, row, col, piece_class)
        for item in board_dataset.items:
            cache_path = self.cache_dir / f"{item.image_id}.png"
            if not cache_path.exists():
                continue
            for pos, class_id in item.pieces.items():
                if class_id not in WHOLE_TO_PIECE:
                    continue
                file_idx = ord(pos[0]) - ord("a")
                rank_idx = int(pos[1]) - 1
                row = 7 - rank_idx
                col = file_idx
                self.index.append((item.image_id, row, col, WHOLE_TO_PIECE[class_id]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        image_id, row, col, piece_cls = self.index[idx]
        warped = cv2.imread(str(self.cache_dir / f"{image_id}.png"))
        if warped is None:
            raise FileNotFoundError(self.cache_dir / f"{image_id}.png")
        if self.augment:
            warped = _aug_color(warped, self.rng)
            # Hflip the whole board with 50% prob; remap target column.
            if self.rng.random() < 0.5:
                warped = warped[:, ::-1].copy()
                col = 7 - col
        patch = crop_cell(warped, row, col)
        return patch_to_tensor(patch), piece_cls
