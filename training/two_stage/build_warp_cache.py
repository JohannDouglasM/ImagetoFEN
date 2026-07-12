#!/usr/bin/env python3
"""Pre-warp every image we need for piece-classifier training and save as
PNG. One-time cost; the cached PieceCropDataset reads from this dir.

Cache layout: <CACHE_DIR>/<image_id>.png  (512x512 BGR PNG)
"""

import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
HARNESS_DIR = REPO_ROOT / "training" / "autoresearch_v3"
CANDIDATE_PATH = REPO_ROOT / "training" / "checkpoints" / "whole_board_0399b00.candidate.py"
ANNOTATIONS = str(REPO_ROOT / "annotations.json")
IMAGES_ROOT = str(REPO_ROOT)
CACHE_DIR = THIS_DIR / "warp_cache"
WARP_SIZE = 512


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def warp_one(item):
    image_id = item.image_id
    out = CACHE_DIR / f"{image_id}.png"
    if out.exists():
        return image_id, "skipped"
    image = cv2.imread(str(item.image_path))
    if image is None:
        return image_id, "missing"
    src = np.array(
        [item.corners["top_left"], item.corners["top_right"],
         item.corners["bottom_right"], item.corners["bottom_left"]],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]],
                   dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(image, H, (WARP_SIZE, WARP_SIZE))
    cv2.imwrite(str(out), warped)
    return image_id, "ok"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HARNESS_DIR))
    candidate = load_module("autoresearch_candidate", CANDIDATE_PATH)
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")

    defaults = candidate.get_defaults()
    seen = set()
    items = []
    for group, split in [
        ("chessred2k", "train"), ("chessred2k", "val"),
        ("chess_dataset_recovered", "train"), ("chess_dataset_recovered", "val"),
        ("synthetic", "train"), ("synthetic", "val"),
    ]:
        bd = harness.BoardDataset(
            ANNOTATIONS, IMAGES_ROOT, group=group, split=split,
            candidate=candidate, candidate_defaults=defaults,
            input_mode=defaults["input_mode"], img_size=defaults["img_size"],
            augment=False, seed=1337,
        )
        for it in bd.items:
            if it.image_id in seen:
                continue
            seen.add(it.image_id)
            items.append(it)
    print(f"\nUnique images to warp: {len(items)}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = missing = skipped = 0
        for i, (iid, status) in enumerate(ex.map(warp_one, items), 1):
            if status == "ok":
                ok += 1
            elif status == "missing":
                missing += 1
            else:
                skipped += 1
            if i % 200 == 0:
                print(f"  {i}/{len(items)}  ok={ok} skipped={skipped} missing={missing}", flush=True)
    print(f"\nDone: ok={ok}, skipped={skipped}, missing={missing}")
    print(f"Cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
