#!/usr/bin/env python3
"""
Fixed benchmark harness for the whole-board classifier autoresearch track.

Parallel to fixed_harness.py but owns a different Dataset and evaluate_loader:
- BoardDataset warps each image to a canonical square board using GT corners
  and loads piece labels from two sources (chessred2k / chess_dataset_recovered
  / user use annotations.pieces; synthetic uses training/data/data.json FEN).
- evaluate_loader computes per-cell and per-board accuracy; metric key names
  mirror fixed_harness.py (mean_dist, max_dist, p95_dist, per_corner) so
  run_commit.py / llm_controller.py / results.tsv keep working unchanged.

Mutable surface remains `candidate.py`. The harness-track pairing is wired by
init_run.py which copies the correct harness file into the worktree as
`fixed_harness.py`.
"""

import argparse
import importlib.util
import json
import math
import os
import re
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

NUM_CLASSES = 13
EMPTY_CLASS_ID = 6

# Class ordering matches src/ml/inference.ts CLASS_TO_PIECE.
CATEGORY_NAME_TO_CLASS_ID = {
    "black-bishop": 0,
    "black-king": 1,
    "black-knight": 2,
    "black-pawn": 3,
    "black-queen": 4,
    "black-rook": 5,
    "empty": EMPTY_CLASS_ID,
    "white-bishop": 7,
    "white-king": 8,
    "white-knight": 9,
    "white-pawn": 10,
    "white-queen": 11,
    "white-rook": 12,
}

FEN_CHAR_TO_CLASS_ID = {
    "b": 0,
    "k": 1,
    "n": 2,
    "p": 3,
    "q": 4,
    "r": 5,
    "B": 7,
    "K": 8,
    "N": 9,
    "P": 10,
    "Q": 11,
    "R": 12,
}


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
        TRAINING_DIR.parent,
    ])

    for root in discover_worktree_roots():
        annotations_candidates.extend([
            root / "annotations.json",
            root / "training" / "data" / "annotations.json",
        ])
        image_root_candidates.extend([
            root / "training" / "data" / "chessred2k",
            root / "training" / "data",
            root,
        ])

    annotations_path = next((p for p in annotations_candidates if p.exists()), None)

    # Images may live under several sibling roots (training/data for synthetic,
    # chessred2k/ for chessred2k, chess-dataset/ for recovered). Pick the first
    # existing one; BoardDataset resolves per-group paths below.
    images_root = None
    for candidate_root in image_root_candidates:
        if (candidate_root / "images").exists() or candidate_root.exists():
            images_root = candidate_root
            break

    if annotations_path is None:
        raise FileNotFoundError("annotations.json not found")
    if images_root is None:
        raise FileNotFoundError("images root not found")
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
    image_id: int
    image_path: Path
    corners: dict
    pieces: dict  # position_str -> class_id


def fen_to_pieces(fen):
    """Parse the placement field of a FEN string into {position: class_id}."""
    placement = fen.split(" ")[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN placement (need 8 ranks): {fen}")
    pieces = {}
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            if ch not in FEN_CHAR_TO_CLASS_ID:
                raise ValueError(f"Unexpected FEN char {ch!r} in {fen}")
            class_id = FEN_CHAR_TO_CLASS_ID[ch]
            pos = f"{chr(ord('a') + file_idx)}{8 - rank_idx}"
            pieces[pos] = class_id
            file_idx += 1
        if file_idx != 8:
            raise ValueError(f"FEN rank does not span 8 files: {rank_str!r} in {fen}")
    return pieces


def load_synthetic_piece_map(annotations_path):
    """Build a mapping from synthetic image file_name -> {pos: class_id} by
    parsing FEN entries in training/data/data.json."""
    search_roots = [TRAINING_DIR / "data", annotations_path.parent / "training" / "data"]
    for worktree_root in discover_worktree_roots():
        search_roots.append(worktree_root / "training" / "data")
    data_json_path = None
    for root in search_roots:
        candidate = root / "data.json"
        if candidate.exists():
            data_json_path = candidate
            break
    if data_json_path is None:
        return {}, None
    with open(data_json_path) as handle:
        entries = json.load(handle)
    piece_map = {}
    for entry in entries:
        image_name = entry.get("image")
        fen = entry.get("fen")
        if not image_name or not fen:
            continue
        try:
            piece_map[image_name] = fen_to_pieces(fen)
        except ValueError:
            continue
    return piece_map, data_json_path.parent


def parse_user_fen_from_filename(path_or_name):
    """User images are named with the FEN placement field using ':' as the rank
    separator (since '/' is a path separator). Strip the extension and any
    '(N)' dedupe suffix, then convert ':' back to '/' and parse."""
    name = Path(path_or_name).name
    stem = name.rsplit(".", 1)[0]
    # strip dedupe suffix like "(2)"
    stem = re.sub(r"\(\d+\)$", "", stem)
    placement = stem.replace(":", "/")
    return fen_to_pieces(placement)


def load_user_piece_map(annotations_path, data):
    """Build {image_id: pieces} for the user split by parsing FEN from filenames."""
    user_ids = set()
    for split_info in data.get("splits", {}).get("user", {}).values():
        if isinstance(split_info, dict):
            user_ids.update(split_info.get("image_ids", []))
    piece_map = {}
    for image in data["images"]:
        if image["id"] not in user_ids:
            continue
        raw = image.get("path") or image.get("file_name") or ""
        try:
            piece_map[image["id"]] = parse_user_fen_from_filename(raw)
        except ValueError:
            continue
    return piece_map


def resolve_image_path(group, image_info, images_root, annotations_root, synthetic_root):
    """Return the filesystem path for this image, walking known group roots."""
    relative_candidates = []
    raw_path = image_info.get("path")
    raw_name = image_info.get("file_name")
    if raw_path:
        relative_candidates.append(raw_path)
    if raw_name and raw_name not in relative_candidates:
        relative_candidates.append(raw_name)

    roots = [images_root, annotations_root]
    if group == "synthetic" and synthetic_root is not None:
        roots.append(synthetic_root)
    if group == "user":
        # annotations.json stores path as 'user/images/<fen>.jpeg' but the
        # actual files live in the repo's assets/ directory.
        user_roots = [
            annotations_root / "assets",
            images_root / "assets",
        ]
        for worktree_root in discover_worktree_roots():
            user_roots.append(worktree_root / "assets")
        roots.extend(user_roots)
        basename = Path(raw_path or raw_name or "").name
        if basename and basename not in relative_candidates:
            relative_candidates.append(basename)
    if group == "chessred2k":
        roots.extend([
            annotations_root / "chessred2k",
            images_root / "chessred2k",
            TRAINING_DIR / "data" / "chessred2k",
        ])
    if group == "chess_dataset_recovered":
        roots.extend([
            annotations_root / "chess-dataset" / "labeled_originals",
            images_root / "chess-dataset" / "labeled_originals",
            TRAINING_DIR / "data" / "chess-dataset" / "labeled_originals",
        ])

    for root in roots:
        if root is None:
            continue
        for rel in relative_candidates:
            path = Path(root) / rel
            if path.exists():
                return path
    return None


class BoardDataset(Dataset):
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
        annotations_path = Path(annotations_path)
        with open(annotations_path) as handle:
            data = json.load(handle)

        self.images_root = Path(images_root)
        self.annotations_root = annotations_path.parent
        self.group = group
        self.split = split
        self.candidate_path = str(Path(candidate.__file__).resolve())
        self._candidate_module = candidate
        self.candidate_defaults = candidate_defaults
        self.input_mode = input_mode
        self.img_size = img_size
        self.augment = augment
        self.base_sample_cache = {}
        self.rng = np.random.default_rng(seed)

        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        self.images = {image["id"]: image for image in data["images"]}
        self.corner_map = {
            annotation["image_id"]: annotation["corners"]
            for annotation in data["annotations"]["corners"]
        }

        ann_piece_map = {}
        for piece in data["annotations"]["pieces"]:
            image_id = piece["image_id"]
            pos = piece["chessboard_position"]
            cat_name = self.categories.get(piece["category_id"], "empty")
            class_id = CATEGORY_NAME_TO_CLASS_ID.get(cat_name, EMPTY_CLASS_ID)
            ann_piece_map.setdefault(image_id, {})[pos] = class_id

        synthetic_piece_map, synthetic_root = load_synthetic_piece_map(annotations_path)
        self.synthetic_root = synthetic_root
        user_piece_map = load_user_piece_map(annotations_path, data)

        split_ids = data["splits"][group][split]["image_ids"]
        self.items = []
        missing_corners = 0
        missing_pieces = 0
        missing_files = 0
        for image_id in split_ids:
            if image_id not in self.images or image_id not in self.corner_map:
                missing_corners += 1
                continue
            image_info = self.images[image_id]

            if group == "synthetic":
                key = image_info.get("file_name") or image_info.get("path")
                pieces = synthetic_piece_map.get(key)
            elif group == "user":
                pieces = user_piece_map.get(image_id)
            else:
                pieces = ann_piece_map.get(image_id)

            if not pieces:
                missing_pieces += 1
                continue

            image_path = resolve_image_path(
                group,
                image_info,
                self.images_root,
                self.annotations_root,
                self.synthetic_root,
            )
            if image_path is None:
                missing_files += 1
                continue

            self.items.append(
                DatasetItem(
                    group=group,
                    split=split,
                    image_id=image_id,
                    image_path=image_path,
                    corners=self.corner_map[image_id],
                    pieces=pieces,
                )
            )

        label = f"{group}:{split}/{input_mode}"
        print(f"Dataset {label}: {len(self.items)} samples", flush=True)
        if missing_corners:
            print(f"  Skipped {missing_corners} missing-corner entries", flush=True)
        if missing_pieces:
            print(f"  Skipped {missing_pieces} missing-piece entries", flush=True)
        if missing_files:
            print(f"  Skipped {missing_files} missing image files", flush=True)

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

    def _warp_board(self, image_bgr, corners, out_size):
        # Use dict labels directly (board-space convention: top_left=a8). Do
        # NOT sort corners by image position; the labels are board-space in
        # annotations.json, verified by comparing corner coords to piece
        # bboxes on chessred2k.
        src = np.array(
            [
                corners["top_left"],
                corners["top_right"],
                corners["bottom_right"],
                corners["bottom_left"],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [0.0, 0.0],
                [out_size, 0.0],
                [out_size, out_size],
                [0.0, out_size],
            ],
            dtype=np.float32,
        )
        homography, _ = cv2.findHomography(src, dst)
        return cv2.warpPerspective(image_bgr, homography, (out_size, out_size))

    def _load_base_sample(self, item):
        cache_key = (item.image_id, self.img_size)
        cached = self.base_sample_cache.get(cache_key)
        if cached is not None:
            return cached

        image_bgr = cv2.imread(str(item.image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {item.image_path}")
        warped = self._warp_board(image_bgr, item.corners, self.img_size)
        self.base_sample_cache[cache_key] = warped
        return warped

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
        warped = self._load_base_sample(item)
        input_tensor = candidate.make_input_tensor(
            warped,
            square_centers=None,
            input_mode=self.input_mode,
            out_size=self.img_size,
        )
        targets = candidate.make_targets(
            None,
            item.pieces,
            orig_w=self.img_size,
            orig_h=self.img_size,
            img_size=self.img_size,
            defaults=self.candidate_defaults,
        )
        meta = {
            "group": item.group,
            "split": item.split,
            "image_id": item.image_id,
            "path": str(item.image_path),
        }
        cached = (torch.from_numpy(input_tensor), targets, meta)
        EVAL_SAMPLE_CACHE[cache_key] = cached
        return cached

    def __getitem__(self, idx):
        item = self.items[idx]
        if not self.augment:
            return self._get_eval_sample(item)

        candidate = self._candidate()
        warped = self._load_base_sample(item)
        warped = candidate.augment_image(warped.copy(), self.rng)
        input_tensor = candidate.make_input_tensor(
            warped,
            square_centers=None,
            input_mode=self.input_mode,
            out_size=self.img_size,
        )
        targets = candidate.make_targets(
            None,
            item.pieces,
            orig_w=self.img_size,
            orig_h=self.img_size,
            img_size=self.img_size,
            defaults=self.candidate_defaults,
        )
        meta = {
            "group": item.group,
            "split": item.split,
            "image_id": item.image_id,
            "path": str(item.image_path),
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


def cell_errors_from_preds(preds, labels):
    """preds: [B, 64] int64 row-major; labels: [B, 8, 8] int64.
    Returns per-board cell error rate as [B] float tensor."""
    flat_labels = labels.view(labels.shape[0], -1)
    wrong = (preds != flat_labels).float()
    return wrong.mean(dim=1)


def evaluate_loader(model, loader, candidate, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    per_board_cell_err = []
    correct_cells = 0
    total_cells = 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    with torch.no_grad():
        for images, targets, _meta in loader:
            images = images.to(device)
            targets = move_to_device(targets, device)
            outputs = model(images)
            loss, _loss_info = candidate.compute_loss(outputs, targets)

            preds = candidate.decode_coords(outputs)
            labels = targets["labels"]
            if preds.dim() != 2 or preds.shape[1] != 64:
                raise RuntimeError(
                    f"decode_coords must return [B, 64], got {tuple(preds.shape)}"
                )

            batch_size = images.size(0)
            batch_cell_err = cell_errors_from_preds(preds, labels).detach().cpu()
            per_board_cell_err.append(batch_cell_err)

            flat_labels = labels.view(batch_size, -1)
            correct_cells += int((preds == flat_labels).sum().item())
            total_cells += batch_size * 64

            preds_cpu = preds.detach().cpu().view(-1)
            labels_cpu = flat_labels.detach().cpu().view(-1)
            for t, p in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                confusion[t, p] += 1

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation loader is empty")

    cell_err_tensor = torch.cat(per_board_cell_err, dim=0)
    mean_cell_err = float(cell_err_tensor.mean().item())
    worst_cell_err = float(cell_err_tensor.max().item())
    p95_cell_err = float(torch.quantile(cell_err_tensor, 0.95).item())
    board_err = float((cell_err_tensor > 0).float().mean().item())

    return {
        "samples": total_samples,
        "loss": float(total_loss / total_samples),
        "mean_dist": mean_cell_err,
        "max_dist": worst_cell_err,
        "p95_dist": p95_cell_err,
        "per_corner": [mean_cell_err, board_err, 0.0, 0.0],
        "cell_accuracy": (correct_cells / total_cells) if total_cells else 0.0,
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
    return None


def parse_args(candidate_defaults):
    parser = argparse.ArgumentParser(description="Fixed benchmark harness for autoresearch v3 (whole-board)")
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
    # Pre-parse just --candidate so defaults come from the target module, not
    # from whatever sits at DEFAULT_CANDIDATE (which may be a corner-track
    # candidate with incompatible defaults like input_mode="gray_edges").
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--candidate", type=str, default=str(DEFAULT_CANDIDATE))
    pre_args, _ = pre_parser.parse_known_args()
    candidate = load_candidate_module(pre_args.candidate)
    candidate_defaults = candidate.get_defaults()
    args = parse_args(candidate_defaults)

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
        BoardDataset(
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
        f"{group}:{split}": BoardDataset(
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
        BoardDataset(
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
    status_template = "elapsed_s={train_elapsed_s:.1f} step={step_count} loss={loss:.4f} cell_err={cell_err:.4f}"

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

        with torch.no_grad():
            preds = candidate.decode_coords(outputs)
            flat_labels = targets["labels"].view(preds.shape[0], -1)
            batch_cell_err = float((preds != flat_labels).float().mean().item())
        step_count += 1
        train_elapsed_s = time.perf_counter() - overall_start

        if step_count == 1 or step_count % 25 == 0:
            message = status_template.format(
                train_elapsed_s=train_elapsed_s,
                step_count=step_count,
                loss=float(loss.item()),
                cell_err=batch_cell_err,
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
            f"combined_cell_err={combined['mean_dist']:.6f} "
            f"combined_board_err={combined['per_corner'][1]:.6f} "
            f"combined_worst={combined['max_dist']:.6f}"
        )
        print(summary, flush=True)
        for name in [*report_datasets.keys(), "combined"]:
            metrics = split_metrics[name]
            print(
                f"  {name}: cell_err={metrics['mean_dist']:.6f} "
                f"board_err={metrics['per_corner'][1]:.6f} "
                f"worst={metrics['max_dist']:.6f}",
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
            print(f"  -> New best combined cell_err: {combined['mean_dist']:.6f}", flush=True)
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
    print(f"primary_cell_err: {results['primary_metric']:.6f}", flush=True)
    for name, metrics in results["split_metrics"].items():
        print(
            f"{name}: cell_err={metrics['mean_dist']:.6f} "
            f"board_err={metrics['per_corner'][1]:.6f} "
            f"worst={metrics['max_dist']:.6f}",
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
