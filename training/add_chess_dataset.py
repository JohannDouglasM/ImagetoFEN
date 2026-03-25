#!/usr/bin/env python3
"""
Reverse engineer board corners for the samryan18/chess-dataset images and merge them
into the shared annotations file.

The source dataset encodes the FEN in each filename, so we can add:
  - image entries
  - piece annotations derived from the FEN
  - corner annotations from detect_board_v5.py

This script is idempotent: existing entries for the same image path are updated.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from detect_board_v5 import detect_board_corners

BASE_DIR = Path(__file__).parent
DEFAULT_ANNOTATIONS = BASE_DIR / "data" / "annotations.json"
DEFAULT_DATASET_ROOT = BASE_DIR / "data" / "chess-dataset" / "labeled_originals"
ROOT_ANNOTATIONS = BASE_DIR.parent / "annotations.json"

FEN_CHAR_TO_CATEGORY = {
    "P": "white-pawn",
    "R": "white-rook",
    "N": "white-knight",
    "B": "white-bishop",
    "Q": "white-queen",
    "K": "white-king",
    "p": "black-pawn",
    "r": "black-rook",
    "n": "black-knight",
    "b": "black-bishop",
    "q": "black-queen",
    "k": "black-king",
}


def fen_from_filename(path: Path) -> str:
    return path.stem.replace("-", "/")


def load_annotations(path: Path):
    with open(path) as f:
        return json.load(f)


def next_id(items):
    if not items:
        return 0
    return max((item.get("id", -1) for item in items), default=-1) + 1


def build_piece_annotations(image_id, fen, category_to_id, next_piece_id):
    annotations = []
    piece_id = next_piece_id
    ranks = fen.split("/")
    for row, rank_str in enumerate(ranks):
        col = 0
        for char in rank_str:
            if char.isdigit():
                col += int(char)
                continue
            category_name = FEN_CHAR_TO_CATEGORY[char]
            square = f"{chr(ord('a') + col)}{8 - row}"
            annotations.append({
                "image_id": image_id,
                "category_id": category_to_id[category_name],
                "chessboard_position": square,
                "id": piece_id,
            })
            piece_id += 1
            col += 1
    return annotations


def ensure_split(data, split_name):
    split = data["splits"].setdefault(split_name, {})
    split.setdefault("train", {"image_ids": []})
    split.setdefault("val", {"image_ids": []})
    return split


def dedupe_ids(values):
    return sorted(set(values))


def main():
    parser = argparse.ArgumentParser(description="Merge chess-dataset into annotations.json")
    parser.add_argument("--annotations", type=str, default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-name", type=str, default="chess_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--write", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sync-root-annotations", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    dataset_root = Path(args.dataset_root)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    data = load_annotations(annotations_path)
    category_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    image_by_path = {img["path"]: img for img in data["images"]}
    corner_by_image = {ann["image_id"]: ann for ann in data["annotations"]["corners"]}
    pieces = data["annotations"]["pieces"]

    split = ensure_split(data, args.split_name)
    next_image_id = next_id(data["images"])
    next_corner_id = next_id(data["annotations"]["corners"])
    next_piece_id = next_id(pieces)

    images = sorted(
        p for p in dataset_root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images:
        images = images[:args.max_images]

    random.seed(args.seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    val_count = int(round(len(shuffled) * args.val_ratio))
    val_paths = {p.resolve() for p in shuffled[:val_count]}

    merged = 0
    created = 0
    failed = []

    for image_path in images:
        resolved_path = str(image_path.resolve())
        fen = fen_from_filename(image_path)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            failed.append((image_path.name, f"open_failed: {exc}"))
            continue

        corners = detect_board_corners(str(image_path), debug_dir=debug_dir)
        if corners is None:
            failed.append((image_path.name, "corner_detection_failed"))
            continue

        img_entry = image_by_path.get(resolved_path)
        if img_entry is None:
            img_entry = {
                "file_name": image_path.name,
                "path": resolved_path,
                "width": width,
                "height": height,
                "camera": "samryan18/chess-dataset",
                "id": next_image_id,
            }
            next_image_id += 1
            data["images"].append(img_entry)
            image_by_path[resolved_path] = img_entry
            created += 1

        image_id = img_entry["id"]
        existing_corner = corner_by_image.get(image_id)
        corner_payload = {
            "top_left": [float(corners[0][0]), float(corners[0][1])],
            "top_right": [float(corners[1][0]), float(corners[1][1])],
            "bottom_right": [float(corners[2][0]), float(corners[2][1])],
            "bottom_left": [float(corners[3][0]), float(corners[3][1])],
        }
        if existing_corner is None:
            new_corner = {"image_id": image_id, "corners": corner_payload, "id": next_corner_id}
            next_corner_id += 1
            data["annotations"]["corners"].append(new_corner)
            corner_by_image[image_id] = new_corner
        else:
            existing_corner["corners"] = corner_payload

        pieces[:] = [ann for ann in pieces if ann["image_id"] != image_id]
        new_pieces = build_piece_annotations(image_id, fen, category_to_id, next_piece_id)
        next_piece_id += len(new_pieces)
        pieces.extend(new_pieces)

        target_split = "val" if image_path.resolve() in val_paths else "train"
        split[target_split]["image_ids"].append(image_id)
        merged += 1

    split["train"]["image_ids"] = dedupe_ids(split["train"]["image_ids"])
    split["val"]["image_ids"] = dedupe_ids(split["val"]["image_ids"])
    split["train"]["n_samples"] = len(split["train"]["image_ids"])
    split["val"]["n_samples"] = len(split["val"]["image_ids"])

    report = {
        "dataset_root": str(dataset_root),
        "annotations_path": str(annotations_path),
        "split_name": args.split_name,
        "merged": merged,
        "created_images": created,
        "failed": failed,
    }

    report_path = BASE_DIR / "data" / f"{args.split_name}_merge_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Merged {merged} images into split '{args.split_name}'")
    print(f"Created {created} new image entries")
    print(f"Failures: {len(failed)}")
    print(f"Report: {report_path}")

    if failed:
        print("First failures:")
        for name, reason in failed[:10]:
            print(f"  {name}: {reason}")

    if args.write:
        annotations_path.write_text(json.dumps(data))
        print(f"Wrote {annotations_path}")
        if args.sync_root_annotations and ROOT_ANNOTATIONS.exists() and ROOT_ANNOTATIONS.resolve() != annotations_path.resolve():
            ROOT_ANNOTATIONS.write_text(json.dumps(data))
            print(f"Synced {ROOT_ANNOTATIONS}")
    else:
        print("Dry run only. Re-run with `--write` to persist the merge.")


if __name__ == "__main__":
    main()
