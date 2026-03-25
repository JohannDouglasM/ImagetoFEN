#!/usr/bin/env python3
"""
Merge accepted recovered chess-dataset corners into annotations.json.

This imports only entries marked `"accepted": true` from
`chess_dataset_recovered_corners.json`, creates image/piece/corner annotations,
and assigns them to a dedicated split so they live alongside ChessReD2K without
changing the existing ChessReD splits.
"""

import argparse
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_RECOVERED = BASE_DIR / "data" / "chess_dataset_recovered_corners.json"
DEFAULT_ANNOTATIONS = BASE_DIR / "data" / "annotations.json"
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


def next_id(items):
    return max((item.get("id", -1) for item in items), default=-1) + 1


def dedupe_ids(values):
    return sorted(set(values))


def build_piece_annotations(image_id, fen, category_to_id, next_piece_id):
    annotations = []
    piece_id = next_piece_id
    for row, rank_str in enumerate(fen.split("/")):
        col = 0
        for char in rank_str:
            if char.isdigit():
                col += int(char)
                continue
            square = f"{chr(ord('a') + col)}{8 - row}"
            annotations.append({
                "image_id": image_id,
                "category_id": category_to_id[FEN_CHAR_TO_CATEGORY[char]],
                "chessboard_position": square,
                "id": piece_id,
            })
            piece_id += 1
            col += 1
    return annotations


def normalize_fen(fen):
    # Some duplicated source files are named like "...-1K6_2.JPG".
    # Strip a trailing duplicate suffix so the stored FEN stays legal.
    ranks = fen.split("/")
    if ranks:
        ranks[-1] = re.sub(r"_\d+$", "", ranks[-1])
    return "/".join(ranks)


def ensure_split(data, split_name):
    split = data["splits"].setdefault(split_name, {})
    split.setdefault("train", {"image_ids": []})
    split.setdefault("val", {"image_ids": []})
    return split


def main():
    parser = argparse.ArgumentParser(description="Merge recovered chess-dataset corners")
    parser.add_argument("--recovered-json", type=str, default=str(DEFAULT_RECOVERED))
    parser.add_argument("--annotations", type=str, default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--split-name", type=str, default="chess_dataset_recovered")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-root-annotations", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    recovered_path = Path(args.recovered_json)
    annotations_path = Path(args.annotations)

    recovered = json.loads(recovered_path.read_text())
    accepted = [item for item in recovered if item.get("accepted")]

    data = json.loads(annotations_path.read_text())
    split = ensure_split(data, args.split_name)
    category_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}

    image_by_path = {img["path"]: img for img in data["images"]}
    corner_by_image = {ann["image_id"]: ann for ann in data["annotations"]["corners"]}
    pieces = data["annotations"]["pieces"]

    next_image_id = next_id(data["images"])
    next_corner_id = next_id(data["annotations"]["corners"])
    next_piece_id = next_id(pieces)

    random.seed(args.seed)
    stems = [item["stem"] for item in accepted]
    shuffled = stems[:]
    random.shuffle(shuffled)
    val_count = int(round(len(shuffled) * args.val_ratio))
    val_stems = set(shuffled[:val_count])

    # Reset this split and refill it from the accepted set.
    split["train"]["image_ids"] = []
    split["val"]["image_ids"] = []

    created_images = 0
    updated_images = 0

    for item in accepted:
        image_path = item["image_path"]
        stem = item["stem"]
        img_entry = image_by_path.get(image_path)
        if img_entry is None:
            img_entry = {
                "id": next_image_id,
                "file_name": Path(image_path).name,
                "path": image_path,
                "width": item["width"],
                "height": item["height"],
                "camera": "samryan18/chess-dataset",
            }
            next_image_id += 1
            data["images"].append(img_entry)
            image_by_path[image_path] = img_entry
            created_images += 1
        else:
            img_entry["width"] = item["width"]
            img_entry["height"] = item["height"]
            img_entry["camera"] = img_entry.get("camera") or "samryan18/chess-dataset"
            updated_images += 1

        image_id = img_entry["id"]
        corners = item["corners"]
        corner_payload = {
            "top_left": corners["top_left"],
            "top_right": corners["top_right"],
            "bottom_right": corners["bottom_right"],
            "bottom_left": corners["bottom_left"],
        }

        corner_ann = corner_by_image.get(image_id)
        if corner_ann is None:
            corner_ann = {
                "image_id": image_id,
                "corners": corner_payload,
                "id": next_corner_id,
            }
            next_corner_id += 1
            data["annotations"]["corners"].append(corner_ann)
            corner_by_image[image_id] = corner_ann
        else:
            corner_ann["corners"] = corner_payload
            corner_ann.setdefault("id", next_corner_id)
            if corner_ann["id"] == next_corner_id:
                next_corner_id += 1

        pieces[:] = [ann for ann in pieces if ann["image_id"] != image_id]
        normalized_fen = normalize_fen(item["fen"])
        new_pieces = build_piece_annotations(image_id, normalized_fen, category_to_id, next_piece_id)
        next_piece_id += len(new_pieces)
        pieces.extend(new_pieces)

        target_split = "val" if stem in val_stems else "train"
        split[target_split]["image_ids"].append(image_id)

    split["train"]["image_ids"] = dedupe_ids(split["train"]["image_ids"])
    split["val"]["image_ids"] = dedupe_ids(split["val"]["image_ids"])
    split["train"]["n_samples"] = len(split["train"]["image_ids"])
    split["val"]["n_samples"] = len(split["val"]["image_ids"])

    report = {
        "recovered_json": str(recovered_path),
        "annotations_path": str(annotations_path),
        "split_name": args.split_name,
        "accepted_examples": len(accepted),
        "created_images": created_images,
        "updated_images": updated_images,
        "train_samples": split["train"]["n_samples"],
        "val_samples": split["val"]["n_samples"],
    }

    report_path = BASE_DIR / "data" / f"{args.split_name}_merge_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if args.write:
        annotations_path.write_text(json.dumps(data))
        if args.sync_root_annotations and ROOT_ANNOTATIONS.exists() and ROOT_ANNOTATIONS.resolve() != annotations_path.resolve():
            ROOT_ANNOTATIONS.write_text(json.dumps(data))

    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
