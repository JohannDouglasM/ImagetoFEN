#!/usr/bin/env python3
"""Measure 'top-down-ness' of each image from its GT corners, then report
what fraction of each dataset passes various thresholds.

Score = min(top_edge, bot_edge) / max(top_edge, bot_edge), combined with the
same ratio for left/right edges. A perfect top-down square shot gives 1.0;
a steep angle where the back edge is much shorter than the front gives << 1.
"""

import json
from pathlib import Path

import numpy as np

ANNOTATIONS = Path("/home/johann/ImagetoFEN/annotations.json")
OUT_DIR = Path("/home/johann/ImagetoFEN/training/crop_test_output")


def edge_ratio(corners):
    tl = np.array(corners["top_left"], dtype=float)
    tr = np.array(corners["top_right"], dtype=float)
    br = np.array(corners["bottom_right"], dtype=float)
    bl = np.array(corners["bottom_left"], dtype=float)
    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    hratio = min(top, bot) / max(top, bot)  # front/back tilt
    vratio = min(left, right) / max(left, right)  # side/side tilt
    return hratio, vratio, min(hratio, vratio)


def main():
    ann = json.load(open(ANNOTATIONS))
    corner_map = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    images = {img["id"]: img for img in ann["images"]}
    splits = ann["splits"]

    groups = ["chessred2k", "chess_dataset_recovered", "synthetic"]
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95]

    print(f"{'group':<28}{'n':>6}  " + "  ".join(f"≥{t:.2f}" for t in thresholds))
    print("-" * 78)

    all_rows = []
    for group in groups:
        ids = []
        for split in ["train", "val", "test"]:
            if split in splits[group]:
                ids.extend(splits[group][split]["image_ids"])

        scores = []
        for img_id in ids:
            if img_id not in corner_map:
                continue
            _, _, score = edge_ratio(corner_map[img_id])
            scores.append(score)
        scores = np.array(scores)
        n = len(scores)

        row = [group, n]
        for t in thresholds:
            passing = (scores >= t).sum()
            row.append(f"{passing:>4d} ({100*passing/n:.0f}%)")
        all_rows.append(row)

        # Print summary stats
        print(f"{group:<28}{n:>6}  " +
              "  ".join(f"{r:>9}" for r in row[2:]))

    print()
    print("Per-group score distribution (min/p10/median/p90/max):")
    for group in groups:
        ids = []
        for split in ["train", "val", "test"]:
            if split in splits[group]:
                ids.extend(splits[group][split]["image_ids"])
        scores = np.array([
            edge_ratio(corner_map[img_id])[2]
            for img_id in ids if img_id in corner_map
        ])
        p = np.percentile(scores, [0, 10, 50, 90, 100])
        print(f"  {group:<28} min={p[0]:.2f}  p10={p[1]:.2f}  "
              f"median={p[2]:.2f}  p90={p[3]:.2f}  max={p[4]:.2f}")

    # Save examples at a few thresholds for visual inspection
    print("\nRepresentative images per score bucket (chessred2k):")
    cr2k_ids = []
    for split in ["train", "val", "test"]:
        cr2k_ids.extend(splits["chessred2k"][split]["image_ids"])
    cr2k_scored = []
    for img_id in cr2k_ids:
        if img_id not in corner_map:
            continue
        score = edge_ratio(corner_map[img_id])[2]
        cr2k_scored.append((score, img_id))
    cr2k_scored.sort()
    buckets = [
        ("steepest", cr2k_scored[:3]),
        ("bucket_0.80", [x for x in cr2k_scored if abs(x[0] - 0.80) < 0.01][:3]),
        ("bucket_0.90", [x for x in cr2k_scored if abs(x[0] - 0.90) < 0.01][:3]),
        ("flattest", cr2k_scored[-3:]),
    ]
    for name, entries in buckets:
        print(f"  {name}:")
        for score, img_id in entries:
            img = images[img_id]
            print(f"    score={score:.3f}  {img['file_name']}")


if __name__ == "__main__":
    main()
