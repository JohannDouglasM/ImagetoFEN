#!/usr/bin/env python3
"""
Recover raw-image board corners for samryan18/chess-dataset by aligning each raw
image to its paired preprocessed board warp.

The paired preprocessed image is treated as the canonical board crop, so its image
bounds correspond to the four board corners. We recover a homography
raw -> preprocessed, invert it, and project the preprocessed image corners back
into the raw image.

Outputs:
  - recovered corner JSON
  - summary report JSON
  - optional debug overlays
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
DEFAULT_DATASET_ROOT = BASE_DIR / "data" / "chess-dataset"


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    return image


def resize_max_side(image, max_side):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image, 1.0
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale)))), scale


def detect_and_describe(gray):
    sift = cv2.SIFT_create(nfeatures=5000)
    return sift.detectAndCompute(gray, None)


def match_descriptors(des1, des2, ratio=0.75):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def reprojection_error(H, pts_src, pts_dst):
    proj = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(proj - pts_dst, axis=1)


def recover_pair(orig_path, prep_path, max_orig_side=1800, ransac_thresh=5.0):
    orig = load_image(orig_path)
    prep = load_image(prep_path)

    orig_small, scale = resize_max_side(orig, max_orig_side)
    prep_small = prep

    gray_orig = cv2.cvtColor(orig_small, cv2.COLOR_BGR2GRAY)
    gray_prep = cv2.cvtColor(prep_small, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detect_and_describe(gray_orig)
    kp2, des2 = detect_and_describe(gray_prep)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, {"reason": "too_few_keypoints", "kp1": len(kp1), "kp2": len(kp2)}

    matches = match_descriptors(des1, des2)
    if len(matches) < 12:
        return None, {"reason": "too_few_matches", "matches": len(matches)}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, {"reason": "homography_failed", "matches": len(matches)}

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 10:
        return None, {
            "reason": "too_few_inliers",
            "matches": len(matches),
            "inliers": inlier_count,
        }

    Hinv = np.linalg.inv(H)
    prep_h, prep_w = prep.shape[:2]
    prep_corners = np.float32([
        [0, 0],
        [prep_w - 1, 0],
        [prep_w - 1, prep_h - 1],
        [0, prep_h - 1],
    ]).reshape(-1, 1, 2)
    raw_corners_small = cv2.perspectiveTransform(prep_corners, Hinv).reshape(-1, 2)
    raw_corners = raw_corners_small / scale

    errors = reprojection_error(H, pts1[inlier_mask], pts2[inlier_mask])
    mean_error = float(errors.mean()) if len(errors) else None
    max_error = float(errors.max()) if len(errors) else None
    inlier_ratio = inlier_count / max(1, len(matches))

    stats = {
        "kp_raw": len(kp1),
        "kp_preprocessed": len(kp2),
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "mean_reprojection_error": mean_error,
        "max_reprojection_error": max_error,
        "scale": scale,
        "prep_size": [prep_w, prep_h],
        "raw_size": [orig.shape[1], orig.shape[0]],
    }
    return raw_corners, stats


def corners_inside_image(corners, width, height, margin=0.15):
    min_x = -width * margin
    max_x = width * (1.0 + margin)
    min_y = -height * margin
    max_y = height * (1.0 + margin)
    return bool(
        (corners[:, 0] >= min_x).all() and
        (corners[:, 0] <= max_x).all() and
        (corners[:, 1] >= min_y).all() and
        (corners[:, 1] <= max_y).all()
    )


def save_debug_overlay(image_path, corners, out_path, label):
    image = load_image(image_path)
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 8)
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], pts):
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(image, name, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main():
    parser = argparse.ArgumentParser(description="Recover chess-dataset corners from paired warped images")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-orig-side", type=int, default=1800)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--max-mean-error", type=float, default=6.0)
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "data" / "chess_dataset_recovered_corners.json"))
    parser.add_argument("--report", type=str, default=str(BASE_DIR / "data" / "chess_dataset_recovery_report.json"))
    parser.add_argument("--debug-dir", type=str, default=str(BASE_DIR / "debug_output" / "chess_dataset_recovery"))
    parser.add_argument("--debug-limit", type=int, default=25)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    report_path = Path(args.report)
    debug_dir = Path(args.debug_dir)

    originals = {p.stem: p for p in (dataset_root / "labeled_originals").glob("*") if p.is_file()}
    preprocessed = {p.stem: p for p in (dataset_root / "labeled_preprocessed").glob("*") if p.is_file()}
    shared_stems = sorted(originals.keys() & preprocessed.keys())
    if args.max_images:
        shared_stems = shared_stems[:args.max_images]

    recovered = []
    failures = []
    debug_saved = 0

    for idx, stem in enumerate(shared_stems, 1):
        orig_path = originals[stem]
        prep_path = preprocessed[stem]

        try:
            corners, stats = recover_pair(
                orig_path,
                prep_path,
                max_orig_side=args.max_orig_side,
            )
        except Exception as exc:
            failures.append({"stem": stem, "reason": f"exception: {exc}"})
            continue

        if corners is None:
            failures.append({"stem": stem, **stats})
            continue

        raw = load_image(orig_path)
        height, width = raw.shape[:2]
        ok = (
            stats["inlier_ratio"] >= args.min_inlier_ratio and
            (stats["mean_reprojection_error"] is None or stats["mean_reprojection_error"] <= args.max_mean_error) and
            corners_inside_image(corners, width, height)
        )

        entry = {
            "stem": stem,
            "image_path": str(orig_path.resolve()),
            "preprocessed_path": str(prep_path.resolve()),
            "width": width,
            "height": height,
            "fen": stem.replace("-", "/"),
            "corners": {
                "top_left": [float(corners[0][0]), float(corners[0][1])],
                "top_right": [float(corners[1][0]), float(corners[1][1])],
                "bottom_right": [float(corners[2][0]), float(corners[2][1])],
                "bottom_left": [float(corners[3][0]), float(corners[3][1])],
            },
            "quality": stats,
            "accepted": ok,
        }
        recovered.append(entry)

        if debug_saved < args.debug_limit:
            label = f"{stem} | ok={ok} inliers={stats['inliers']}/{stats['matches']} err={stats['mean_reprojection_error']:.2f}"
            save_debug_overlay(orig_path, corners, debug_dir / f"{stem}.jpg", label)
            debug_saved += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recovered, indent=2))

    accepted = [item for item in recovered if item["accepted"]]
    report = {
        "dataset_root": str(dataset_root),
        "total_pairs": len(shared_stems),
        "recovered": len(recovered),
        "accepted": len(accepted),
        "failed": len(failures),
        "failures": failures[:100],
        "output": str(output_path),
        "debug_dir": str(debug_dir),
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Pairs:     {len(shared_stems)}")
    print(f"Recovered: {len(recovered)}")
    print(f"Accepted:  {len(accepted)}")
    print(f"Failed:    {len(failures)}")
    print(f"Corners:   {output_path}")
    print(f"Report:    {report_path}")
    print(f"Debug:     {debug_dir}")


if __name__ == "__main__":
    main()
