#!/usr/bin/env python3
"""
Fixed benchmark harness for chess corner autoresearch v3.

This file owns:
- data loading
- validation metrics
- fixed-time training budget
- checkpoint serialization
- split reporting
- result JSON schema

The intended mutable surface is `candidate.py`.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
DEFAULT_CANDIDATE = THIS_DIR / "candidate.py"
DEFAULT_NUM_WORKERS = 0 if (os.cpu_count() or 1) <= 1 else min(4, os.cpu_count() or 1)
EVAL_SAMPLE_CACHE = {}


def write_status(status_file, message):
    if not status_file:
        return
    path = Path(status_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)", flush=True)
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
        return torch.device("cuda")
    print("Using CPU", flush=True)
    return torch.device("cpu")


def discover_worktree_roots():
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(TRAINING_DIR.parent),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    roots = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]).resolve())
    return roots


def resolve_data_paths(annotations_arg=None, images_root_arg=None):
    annotations_candidates = []
    image_root_candidates = []
    if annotations_arg:
        annotations_candidates.append(Path(annotations_arg))
    annotations_candidates.extend([
        TRAINING_DIR / "data" / "annotations.json",
        TRAINING_DIR.parent / "annotations.json",
    ])

    if images_root_arg:
        image_root_candidates.append(Path(images_root_arg))
    image_root_candidates.extend([
        TRAINING_DIR / "data" / "chessred2k",
        TRAINING_DIR / "data",
    ])

    for root in discover_worktree_roots():
        annotations_candidates.extend([
            root / "annotations.json",
            root / "training" / "data" / "annotations.json",
        ])
        image_root_candidates.extend([
            root / "training" / "data" / "chessred2k",
            root / "training" / "data",
        ])

    annotations_path = next((p for p in annotations_candidates if p.exists()), None)
    images_root = next((p for p in image_root_candidates if (p / "images").exists()), None)

    if annotations_path is None:
        raise FileNotFoundError("annotations.json not found")
    if images_root is None:
        raise FileNotFoundError("images/ directory not found")
    return annotations_path, images_root


def load_candidate_module(path):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("autoresearch_candidate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_split_selectors(raw):
    selectors = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid split selector '{item}', expected group:split")
        group, split = item.split(":", 1)
        selectors.append((group, split))
    return selectors


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


@dataclass
class DatasetItem:
    group: str
    split: str
    image_id: str
    path: str
    corners: dict


CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def corners_to_array(corners):
    return np.array([corners[key] for key in CORNER_KEYS], dtype=np.float32)


def canonicalize_corners(corners):
    pts = corners_to_array(corners)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    # Rotate so the visually top-left image corner comes first.
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)

    # The two neighbors of the top-left point are the top-right and bottom-left
    # corners. On steep diamond views the visually top-right corner is often not
    # the more rightward neighbor; it is the neighbor that sits higher in the
    # image (smaller y). Use x only as a tie-breaker.
    next_is_tr = (
        pts[1, 1] < pts[-1, 1]
        or (abs(float(pts[1, 1] - pts[-1, 1])) <= 1e-3 and pts[1, 0] > pts[-1, 0])
    )
    if not next_is_tr:
        pts = np.array([pts[0], pts[-1], pts[-2], pts[-3]], dtype=np.float32)

    return {
        key: [float(x), float(y)]
        for key, (x, y) in zip(CORNER_KEYS, pts.tolist())
    }


class CornerDataset(Dataset):
    def __init__(
        self,
        annotations_path,
        images_root,
        *,
        group,
        split,
        candidate,
        candidate_defaults,
        input_mode,
        img_size,
        augment,
        seed,
    ):
        with open(annotations_path) as handle:
            data = json.load(handle)

        self.images_root = Path(images_root)
        self.group = group
        self.split = split
        self.candidate_path = str(Path(candidate.__file__).resolve())
        self._candidate_module = candidate
        self.candidate_defaults = candidate_defaults
        self.input_mode = input_mode
        self.img_size = img_size
        self.augment = augment
        self.square_cache = {}
        self.square_cache_enabled = bool(candidate.needs_square_centers(input_mode))
        self.base_sample_cache = {}
        self.rng = np.random.default_rng(seed)

        self.images = {image["id"]: image for image in data["images"]}
        self.corner_map = {
            annotation["image_id"]: annotation["corners"]
            for annotation in data["annotations"]["corners"]
        }

        split_ids = data["splits"][group][split]["image_ids"]
        self.items = []
        missing = 0
        reordered = 0
        for image_id in split_ids:
            if image_id not in self.images or image_id not in self.corner_map:
                continue
            image_path = self.images_root / self.images[image_id]["path"]
            if not image_path.exists():
                missing += 1
                continue
            corners = self.corner_map[image_id]
            canonical_corners = canonicalize_corners(corners)
            if not np.allclose(
                corners_to_array(corners),
                corners_to_array(canonical_corners),
                atol=1e-3,
            ):
                reordered += 1
            self.items.append(
                DatasetItem(
                    group=group,
                    split=split,
                    image_id=image_id,
                    path=self.images[image_id]["path"],
                    corners=canonical_corners,
                )
            )

        label = f"{group}:{split}/{input_mode}"
        print(f"Dataset {label}: {len(self.items)} samples", flush=True)
        if missing:
            print(f"  Skipped {missing} missing files", flush=True)
        if reordered:
            print(f"  Canonicalized {reordered} corner orders", flush=True)

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_candidate_module"] = None
        return state

    def _candidate(self):
        if self._candidate_module is None:
            self._candidate_module = load_candidate_module(self.candidate_path)
        return self._candidate_module

    def _get_square_centers(self, image_id, image_bgr):
        if not self.square_cache_enabled:
            return []
        if image_id in self.square_cache:
            return self.square_cache[image_id]
        centers = self._candidate().find_empty_squares_for_image(image_bgr)
        self.square_cache[image_id] = centers
        return centers

    def _load_base_sample(self, item):
        cache_key = (item.image_id, self.img_size, self.input_mode)
        cached = self.base_sample_cache.get(cache_key)
        if cached is not None:
            return cached

        image_path = self.images_root / item.path
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {image_path}")

        orig_h, orig_w = image_bgr.shape[:2]
        square_centers = self._get_square_centers(item.image_id, image_bgr)
        resized_bgr = cv2.resize(image_bgr, (self.img_size, self.img_size))
        cached = (resized_bgr, orig_h, orig_w, square_centers)
        self.base_sample_cache[cache_key] = cached
        return cached

    def _get_eval_sample(self, item):
        cache_key = (
            item.image_id,
            self.img_size,
            self.input_mode,
            self.candidate_defaults["candidate_name"],
        )
        cached = EVAL_SAMPLE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        candidate = self._candidate()
        resized_bgr, orig_h, orig_w, square_centers = self._load_base_sample(item)
        input_tensor = candidate.make_input_tensor(
            resized_bgr,
            square_centers=square_centers,
            input_mode=self.input_mode,
            out_size=self.img_size,
        )
        targets = candidate.make_targets(
            item.corners,
            orig_w=orig_w,
            orig_h=orig_h,
            img_size=self.img_size,
            defaults=self.candidate_defaults,
        )
        meta = {
            "group": item.group,
            "split": item.split,
            "image_id": item.image_id,
            "path": item.path,
        }
        cached = (torch.from_numpy(input_tensor), targets, meta)
        EVAL_SAMPLE_CACHE[cache_key] = cached
        return cached

    def __getitem__(self, idx):
        item = self.items[idx]
        if not self.augment:
            return self._get_eval_sample(item)

        candidate = self._candidate()
        resized_bgr, orig_h, orig_w, square_centers = self._load_base_sample(item)
        image_bgr = candidate.augment_image(resized_bgr.copy(), self.rng)

        input_tensor = candidate.make_input_tensor(
            image_bgr,
            square_centers=square_centers,
            input_mode=self.input_mode,
            out_size=self.img_size,
        )

        targets = candidate.make_targets(
            item.corners,
            orig_w=orig_w,
            orig_h=orig_h,
            img_size=self.img_size,
            defaults=self.candidate_defaults,
        )

        meta = {
            "group": item.group,
            "split": item.split,
            "image_id": item.image_id,
            "path": item.path,
        }
        return torch.from_numpy(input_tensor), targets, meta


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index = []
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))
        print(
            f"Combined dataset: {len(self.index)} samples from {len(datasets)} splits",
            flush=True,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        return self.datasets[dataset_idx][sample_idx]


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def corner_distances(pred_coords, target_coords):
    pred = pred_coords.view(-1, 4, 2)
    targ = target_coords.view(-1, 4, 2)
    return torch.sqrt(((pred - targ) ** 2).sum(dim=2))


def evaluate_loader(model, loader, candidate, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_dists = []

    with torch.no_grad():
        for images, targets, _meta in loader:
            images = images.to(device)
            targets = move_to_device(targets, device)
            outputs = model(images)
            loss, _loss_info = candidate.compute_loss(outputs, targets)
            pred_coords = candidate.decode_coords(outputs)
            dists = corner_distances(pred_coords, targets["coords"])

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_dists.append(dists.cpu())

    if total_samples == 0:
        raise RuntimeError("Validation loader is empty")

    dist_tensor = torch.cat(all_dists, dim=0)
    return {
        "samples": total_samples,
        "loss": float(total_loss / total_samples),
        "mean_dist": float(dist_tensor.mean().item()),
        "max_dist": float(dist_tensor.max().item()),
        "p95_dist": float(torch.quantile(dist_tensor.view(-1), 0.95).item()),
        "per_corner": [float(x) for x in dist_tensor.mean(dim=0).tolist()],
    }


def save_checkpoint(path, model, payload):
    checkpoint = dict(payload)
    checkpoint["model_state_dict"] = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def select_resume_checkpoint(explicit_resume, candidate_defaults):
    if explicit_resume:
        return Path(explicit_resume)

    for raw in candidate_defaults.get("resume_candidates", []):
        candidate_path = Path(raw)
        if not candidate_path.is_absolute():
            candidate_path = TRAINING_DIR / raw
        if candidate_path.exists():
            return candidate_path

    if not candidate_defaults.get("allow_legacy_resume_fallback", True):
        return None

    legacy = TRAINING_DIR / "autoresearch_gray_edges_mixed_models" / "best_corner_hybrid.pt"
    if legacy.exists():
        return legacy
    return None


def parse_args(candidate_defaults):
    parser = argparse.ArgumentParser(description="Fixed benchmark harness for autoresearch v3")
    parser.add_argument("--candidate", type=str, default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--time-budget-s", type=float, default=1800.0)
    parser.add_argument("--eval-interval-s", type=float, default=candidate_defaults["eval_interval_s"])
    parser.add_argument("--batch-size", type=int, default=candidate_defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=candidate_defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=candidate_defaults["weight_decay"])
    parser.add_argument("--img-size", type=int, default=candidate_defaults["img_size"])
    parser.add_argument("--input-mode", type=str, default=candidate_defaults["input_mode"])
    parser.add_argument("--train-splits", type=str, default=candidate_defaults["train_splits"])
    parser.add_argument("--val-splits", type=str, default=candidate_defaults["val_splits"])
    parser.add_argument("--report-splits", type=str, default=candidate_defaults["report_splits"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    initial_candidate = load_candidate_module(DEFAULT_CANDIDATE)
    initial_defaults = initial_candidate.get_defaults()
    args = parse_args(initial_defaults)
    candidate = load_candidate_module(args.candidate)
    candidate_defaults = candidate.get_defaults()

    set_seed(args.seed)
    annotations_path, images_root = resolve_data_paths(args.annotations, args.images_root)
    device = get_device()
    input_channels = candidate.INPUT_MODE_TO_CHANNELS[args.input_mode]

    print(f"Candidate:    {candidate_defaults['candidate_name']}", flush=True)
    print(f"Annotations:  {annotations_path}", flush=True)
    print(f"Images root:  {images_root}", flush=True)
    print(f"Input mode:   {args.input_mode} ({input_channels}ch)", flush=True)
    print(f"Batch size:   {args.batch_size}", flush=True)
    print(f"LR:           {args.lr}", flush=True)
    print(f"Weight decay: {args.weight_decay}", flush=True)
    print(f"Num workers:  {args.num_workers}", flush=True)
    print(f"Wall budget:  {args.time_budget_s}", flush=True)
    print(f"Eval every s: {args.eval_interval_s}", flush=True)

    train_datasets = [
        CornerDataset(
            annotations_path,
            images_root,
            group=group,
            split=split,
            candidate=candidate,
            candidate_defaults=candidate_defaults,
            input_mode=args.input_mode,
            img_size=args.img_size,
            augment=True,
            seed=args.seed + index,
        )
        for index, (group, split) in enumerate(parse_split_selectors(args.train_splits))
    ]
    report_datasets = {
        f"{group}:{split}": CornerDataset(
            annotations_path,
            images_root,
            group=group,
            split=split,
            candidate=candidate,
            candidate_defaults=candidate_defaults,
            input_mode=args.input_mode,
            img_size=args.img_size,
            augment=False,
            seed=args.seed + 100 + index,
        )
        for index, (group, split) in enumerate(parse_split_selectors(args.report_splits))
    }
    combined_val_datasets = [
        CornerDataset(
            annotations_path,
            images_root,
            group=group,
            split=split,
            candidate=candidate,
            candidate_defaults=candidate_defaults,
            input_mode=args.input_mode,
            img_size=args.img_size,
            augment=False,
            seed=args.seed + 200 + index,
        )
        for index, (group, split) in enumerate(parse_split_selectors(args.val_splits))
    ]

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else CombinedDataset(train_datasets)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, dataset in report_datasets.items()
    }
    val_loaders["combined"] = DataLoader(
        combined_val_datasets[0] if len(combined_val_datasets) == 1 else CombinedDataset(combined_val_datasets),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = candidate.build_model(input_channels=input_channels)
    resume_path = select_resume_checkpoint(args.resume, candidate_defaults)
    resumed = False
    resume_meta = None
    if resume_path and resume_path.exists():
        resumed = True
        print(f"Resume:       {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        resume_meta = candidate.load_checkpoint(model, checkpoint["model_state_dict"])
        if resume_meta.get("missing"):
            print(f"  Missing keys: {resume_meta['missing']}", flush=True)
        if resume_meta.get("unexpected"):
            print(f"  Unexpected keys: {resume_meta['unexpected']}", flush=True)
        if resume_meta.get("skipped_mismatch"):
            preview = ", ".join(
                f"{key} {src}->{dst}"
                for key, src, dst in resume_meta["skipped_mismatch"][:4]
            )
            if len(resume_meta["skipped_mismatch"]) > 4:
                preview += ", ..."
            print(f"  Skipped mismatch: {preview}", flush=True)

    model = model.to(device)
    estimated_steps = max(1, int(math.ceil(args.time_budget_s / max(1.0, args.eval_interval_s))) * len(train_loader))
    optimizer = candidate.create_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        resumed=resumed,
    )
    scheduler = candidate.create_scheduler(optimizer, total_train_steps=estimated_steps)

    best_metrics = None
    best_checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else None
    no_improve_evals = 0
    train_elapsed_s = 0.0
    step_count = 0
    last_eval_at = 0.0
    overall_start = time.perf_counter()

    print("=" * 60, flush=True)
    print("Training under fixed wall-clock budget", flush=True)
    print("=" * 60, flush=True)

    loader_iter = cycle_loader(train_loader)
    status_template = "elapsed_s={train_elapsed_s:.1f} step={step_count} loss={loss:.4f} mean={mean:.4f}"

    while train_elapsed_s < args.time_budget_s:
        images, targets, _meta = next(loader_iter)
        images = images.to(device)
        targets = move_to_device(targets, device)

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss, _loss_info = candidate.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred_coords = candidate.decode_coords(outputs).detach()
        batch_dists = corner_distances(pred_coords, targets["coords"])
        batch_mean = float(batch_dists.mean().item())
        step_count += 1
        train_elapsed_s = time.perf_counter() - overall_start

        if step_count == 1 or step_count % 25 == 0:
            message = status_template.format(
                train_elapsed_s=train_elapsed_s,
                step_count=step_count,
                loss=float(loss.item()),
                mean=batch_mean,
            )
            print(message, flush=True)
            write_status(args.status_file, message)

        should_eval = train_elapsed_s >= args.time_budget_s or (train_elapsed_s - last_eval_at) >= args.eval_interval_s
        if not should_eval:
            continue

        last_eval_at = train_elapsed_s
        split_metrics = {
            name: evaluate_loader(model, loader, candidate, device)
            for name, loader in val_loaders.items()
        }
        combined = split_metrics["combined"]
        summary = (
            f"eval elapsed_s={train_elapsed_s:.1f} step={step_count} "
            f"combined_mean={combined['mean_dist']:.6f} "
            f"combined_max={combined['max_dist']:.6f} "
            f"combined_p95={combined['p95_dist']:.6f}"
        )
        print(summary, flush=True)
        for name in [*report_datasets.keys(), "combined"]:
            metrics = split_metrics[name]
            print(
                f"  {name}: mean={metrics['mean_dist']:.6f} "
                f"max={metrics['max_dist']:.6f} "
                f"p95={metrics['p95_dist']:.6f}",
                flush=True,
            )
        write_status(args.status_file, summary)

        improved = best_metrics is None or combined["mean_dist"] < best_metrics["combined"]["mean_dist"]
        if improved:
            best_metrics = {
                "step_count": step_count,
                "train_elapsed_s": train_elapsed_s,
                "split_metrics": split_metrics,
                "combined": combined,
            }
            no_improve_evals = 0
            print(f"  -> New best combined mean: {combined['mean_dist']:.6f}", flush=True)
            if best_checkpoint_path:
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    {
                        "train_elapsed_s": train_elapsed_s,
                        "step_count": step_count,
                        "input_mode": args.input_mode,
                        "input_channels": input_channels,
                        "candidate_name": candidate_defaults["candidate_name"],
                        "best_metrics": best_metrics,
                    },
                )
        else:
            no_improve_evals += 1
            if no_improve_evals >= candidate_defaults["max_no_improve_evals"]:
                print("Early stop: exceeded no-improve eval limit", flush=True)
                break

    if best_metrics is None:
        raise RuntimeError("No evaluation was completed")

    results = {
        "candidate_name": candidate_defaults["candidate_name"],
        "candidate_path": str(Path(args.candidate).resolve()),
        "annotations_path": str(annotations_path),
        "images_root": str(images_root),
        "resume_path": str(resume_path) if resume_path else None,
        "resumed": resumed,
        "resume_meta": resume_meta,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "time_budget_s": args.time_budget_s,
        "eval_interval_s": args.eval_interval_s,
        "train_splits": args.train_splits,
        "val_splits": args.val_splits,
        "report_splits": args.report_splits,
        "train_elapsed_s": train_elapsed_s,
        "wall_elapsed_s": train_elapsed_s,
        "step_count": step_count,
        "best_step_count": best_metrics["step_count"],
        "best_train_elapsed_s": best_metrics["train_elapsed_s"],
        "best_wall_elapsed_s": best_metrics["train_elapsed_s"],
        "split_metrics": best_metrics["split_metrics"],
        "primary_metric": best_metrics["combined"]["mean_dist"],
        "checkpoint_out": str(best_checkpoint_path) if best_checkpoint_path else None,
    }

    print("=" * 60, flush=True)
    print(f"primary_mean_dist: {results['primary_metric']:.6f}", flush=True)
    for name, metrics in results["split_metrics"].items():
        print(
            f"{name}: mean={metrics['mean_dist']:.6f} max={metrics['max_dist']:.6f} p95={metrics['p95_dist']:.6f}",
            flush=True,
        )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        raise
