#!/usr/bin/env python3
"""
Autoresearch: autonomous experiment loop for the mixed-data corner detector.

Defaults target the faster gray_edges input mode on the combined dataset:
  - train: chessred2k + user + recovered green-board images
  - val:   chessred2k + recovered green-board images

The loop patches train_corners_hybrid.py, runs short experiments, and keeps only
improvements relative to the current best checkpoint under this mixed validation.
"""

import atexit
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_SCRIPT = BASE_DIR / "train_corners_hybrid.py"
BACKUP_FILE = BASE_DIR / ".autoresearch_backup"
DEFAULT_RESULTS_FILE = BASE_DIR / "autoresearch_gray_edges_mixed_v2_results.tsv"
DEFAULT_MODELS_DIR = BASE_DIR / "autoresearch_gray_edges_mixed_v2_models"
DEFAULT_TRAIN_SPLITS = "chessred2k:train,user:train,chess_dataset_recovered:train"
DEFAULT_VAL_SPLITS = "chessred2k:val,chess_dataset_recovered:val"
DEFAULT_RESUME_CANDIDATES = [
    BASE_DIR / "autoresearch_gray_edges_mixed_v2_models" / "best_corner_hybrid.pt",
    BASE_DIR / "autoresearch_gray_edges_mixed_models" / "best_corner_hybrid.pt",
    BASE_DIR / "ablation" / "gray_edges_10ep" / "last_corner_hybrid.pt",
    BASE_DIR / "models" / "best_corner_hybrid.pt",
]

# Each experiment: {id, name, patches: [(old, new), ...], args: [...], batch_size}
EXPERIMENTS = [
    {
        "id": "001", "name": "lr=0.00006",
        "patches": [],
        "args": ["--lr", "0.00006"],
    },
    {
        "id": "002", "name": "lr=0.00008",
        "patches": [],
        "args": ["--lr", "0.00008"],
    },
    {
        "id": "003", "name": "lr=0.00012",
        "patches": [],
        "args": ["--lr", "0.00012"],
    },
    {
        "id": "004", "name": "lr=0.00015",
        "patches": [],
        "args": ["--lr", "0.00015"],
    },
    {
        "id": "005", "name": "lr=0.0002",
        "patches": [],
        "args": ["--lr", "0.0002"],
    },
    {
        "id": "006", "name": "dropout=0.0",
        "patches": [
            ('nn.Dropout(0.2),', 'nn.Dropout(0.0),'),
        ],
    },
    {
        "id": "007", "name": "dropout=0.05",
        "patches": [
            ('nn.Dropout(0.2),', 'nn.Dropout(0.05),'),
        ],
    },
    {
        "id": "008", "name": "dropout=0.1",
        "patches": [
            ('nn.Dropout(0.2),', 'nn.Dropout(0.1),'),
        ],
    },
    {
        "id": "009", "name": "MSELoss",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.MSELoss()'),
        ],
    },
    {
        "id": "010", "name": "SmoothL1(beta=0.25)",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.SmoothL1Loss(beta=0.25)'),
        ],
    },
    {
        "id": "011", "name": "SmoothL1(beta=0.5)",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.SmoothL1Loss(beta=0.5)'),
        ],
    },
    {
        "id": "012", "name": "SmoothL1(beta=0.75)",
        "patches": [
            ('criterion = nn.SmoothL1Loss()',
             'criterion = nn.SmoothL1Loss(beta=0.75)'),
        ],
    },
    {
        "id": "013", "name": "weight_decay=0.0005",
        "patches": [
            ('weight_decay=0.001)',
             'weight_decay=0.0005)'),
        ],
    },
    {
        "id": "014", "name": "weight_decay=0.002",
        "patches": [
            ('weight_decay=0.001)',
             'weight_decay=0.002)'),
        ],
    },
    {
        "id": "015", "name": "weight_decay=0.005",
        "patches": [
            ('weight_decay=0.001)',
             'weight_decay=0.005)'),
        ],
    },
    {
        "id": "016", "name": "Canny 70/180",
        "patches": [
            ('edges = cv2.Canny(blurred, 80, 200)',
             'edges = cv2.Canny(blurred, 70, 180)'),
        ],
    },
    {
        "id": "017", "name": "Canny 90/220",
        "patches": [
            ('edges = cv2.Canny(blurred, 80, 200)',
             'edges = cv2.Canny(blurred, 90, 220)'),
        ],
    },
    {
        "id": "018", "name": "Canny 100/250",
        "patches": [
            ('edges = cv2.Canny(blurred, 80, 200)',
             'edges = cv2.Canny(blurred, 100, 250)'),
        ],
    },
    {
        "id": "019", "name": "brightness ±30",
        "patches": [
            ('beta = np.random.uniform(-40, 40)',
             'beta = np.random.uniform(-30, 30)'),
        ],
    },
    {
        "id": "020", "name": "brightness ±50",
        "patches": [
            ('beta = np.random.uniform(-40, 40)',
             'beta = np.random.uniform(-50, 50)'),
        ],
    },
    {
        "id": "021", "name": "contrast 0.5-1.5",
        "patches": [
            ('alpha = np.random.uniform(0.6, 1.4)',
             'alpha = np.random.uniform(0.5, 1.5)'),
        ],
    },
    {
        "id": "022", "name": "gaussian_noise_std5",
        "patches": [
            ('            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)',
             """            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            # Gaussian noise
            noise = np.random.normal(0, 5, image_bgr.shape).astype(np.float32)
            image_bgr = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)"""),
        ],
    },
    {
        "id": "023", "name": "phase2_lr_mult=0.05",
        "patches": [
            ('optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.001)',
             'optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.05, weight_decay=0.001)'),
        ],
    },
    {
        "id": "024", "name": "phase2_lr_mult=0.2",
        "patches": [
            ('optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.001)',
             'optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.2, weight_decay=0.001)'),
        ],
    },
    {
        "id": "025", "name": "batch_size=24",
        "patches": [],
        "batch_size": 24,
    },
]


def apply_patches(source, patches):
    """Apply find/replace patches to source string. Returns patched source."""
    result = source
    for old, new in patches:
        if old not in result:
            raise ValueError(f"Patch target not found in source:\n{old[:100]}...")
        result = result.replace(old, new, 1)
    return result


def inject_skip_onnx(source):
    """Insert a return statement before the ONNX export block."""
    marker = "    # Export to ONNX"
    if marker not in source:
        return source
    return source.replace(
        marker,
        "    return  # autoresearch: skip ONNX export\n\n" + marker,
    )


def parse_best_dist(output):
    """Extract 'Best mean corner distance: X.XXXX' from training output."""
    # Match the final summary line
    match = re.search(r"Best mean corner distance:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    # Fallback: look for best val_dist in epoch logs
    best = None
    for m in re.finditer(r"New best! dist=([\d.]+)", output):
        val = float(m.group(1))
        if best is None or val < best:
            best = val
    return best


def load_completed(tsv_path):
    """Load completed experiments from TSV. Returns {id: {status, val_dist, ...}}."""
    completed = {}
    if not tsv_path.exists():
        return completed
    with open(tsv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                completed[parts[0]] = {
                    "name": parts[1],
                    "status": parts[2],
                    "val_dist": float(parts[3]) if parts[3] != "N/A" else None,
                }
    return completed


def reconstruct_baseline(original_source, experiments, completed):
    """Replay winning patches on original source to get current baseline."""
    source = original_source
    for exp in experiments:
        exp_id = exp["id"]
        if exp_id in completed and completed[exp_id]["status"] == "KEPT":
            try:
                source = apply_patches(source, exp["patches"])
            except ValueError:
                print(f"  Warning: could not replay patches for {exp_id}, "
                      f"continuing with current baseline", flush=True)
    return source


def log_result(tsv_path, exp_id, name, status, val_dist, elapsed, notes=""):
    """Append result to TSV."""
    if not tsv_path.exists():
        with open(tsv_path, "w") as f:
            f.write("id\tname\tstatus\tval_dist\telapsed_s\tnotes\n")
    dist_str = f"{val_dist:.6f}" if val_dist is not None else "N/A"
    with open(tsv_path, "a") as f:
        f.write(f"{exp_id}\t{name}\t{status}\t{dist_str}\t{elapsed:.0f}\t{notes}\n")


def restore_script(backup_path, script_path):
    """Restore training script from backup."""
    if backup_path.exists():
        shutil.copy2(backup_path, script_path)


def resolve_default_resume():
    for candidate in DEFAULT_RESUME_CANDIDATES:
        if candidate.exists():
            return candidate
    return BASE_DIR / "models" / "best_corner_hybrid.pt"


def load_checkpoint_val_dist(checkpoint_path):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        value = ckpt.get("val_dist")
        return float(value) if value is not None else None
    except Exception:
        return None


def run_experiment(
    exp,
    baseline_source,
    *,
    epochs=10,
    default_batch_size=16,
    timeout=1200,
    resume_checkpoint=None,
    input_mode="gray_edges",
    train_splits=DEFAULT_TRAIN_SPLITS,
    val_splits=DEFAULT_VAL_SPLITS,
    models_dir=DEFAULT_MODELS_DIR,
):
    """
    Run a single experiment:
    1. Apply patches to baseline source
    2. Write modified script
    3. Run training (optionally resuming from checkpoint)
    4. Parse result
    Returns (val_dist, stdout) or (None, error_msg)
    """
    exp_id = exp["id"]
    name = exp["name"]

    # Apply patches
    try:
        patched = apply_patches(baseline_source, exp["patches"])
    except ValueError as e:
        return None, f"PATCH_FAILED: {e}"

    # Skip ONNX export
    patched = inject_skip_onnx(patched)

    # Write patched script
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(patched)

    # Build command
    batch_size = exp.get("batch_size", default_batch_size)
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--input-mode", input_mode,
        "--train-splits", train_splits,
        "--val-splits", val_splits,
        "--models-dir", str(models_dir),
        "--status-file", str(Path(models_dir) / f"exp_{exp_id}.status.txt"),
    ]
    cmd.extend(exp.get("args", []))
    if resume_checkpoint:
        resume_abs = str(Path(resume_checkpoint).resolve())
        if Path(resume_abs).exists():
            cmd.extend(["--resume", resume_abs])

    print(f"  Running: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE_DIR),
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            # Print last few lines for debugging
            lines = output.strip().split("\n")
            tail = "\n".join(lines[-10:])
            print(f"  FAILED (exit {result.returncode}):\n{tail}", flush=True)
            return None, f"EXIT_{result.returncode}: {tail[-200:]}"

        val_dist = parse_best_dist(output)
        if val_dist is None:
            return None, "PARSE_FAILED: could not find val_dist in output"

        return val_dist, output

    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch: autonomous experiment loop")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per experiment (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Default batch size (default: 16)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout per experiment in seconds (default: no timeout)")
    parser.add_argument("--baseline-dist", type=float, default=None,
                        help="Starting best val_dist (default: infer from resume checkpoint)")
    parser.add_argument("--resume", type=str,
                        default=str(resolve_default_resume()),
                        help="Checkpoint to resume from (default: best gray_edges mixed checkpoint if available)")
    parser.add_argument("--input-mode", type=str, default="gray_edges",
                        choices=["gray_edges", "hybrid", "gray"],
                        help="Input mode used for all experiments by default")
    parser.add_argument("--train-splits", type=str, default=DEFAULT_TRAIN_SPLITS,
                        help="Comma-separated train split selectors")
    parser.add_argument("--val-splits", type=str, default=DEFAULT_VAL_SPLITS,
                        help="Comma-separated val split selectors")
    parser.add_argument("--results-file", type=str, default=str(DEFAULT_RESULTS_FILE),
                        help="TSV file for experiment results")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Directory for autoresearch experiment artifacts")
    args = parser.parse_args()

    tsv_file = Path(args.results_file)
    models_dir = Path(args.models_dir)
    baseline_dist = args.baseline_dist
    if baseline_dist is None:
        baseline_dist = load_checkpoint_val_dist(args.resume)
    if baseline_dist is None:
        baseline_dist = 0.03

    print("=" * 70)
    print("AUTORESEARCH: Autonomous Experiment Loop")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Default batch size: {args.batch_size}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Baseline val_dist: {baseline_dist:.4f}")
    print(f"  Resume from: {args.resume}")
    print(f"  Input mode: {args.input_mode}")
    print(f"  Train splits: {args.train_splits}")
    print(f"  Val splits: {args.val_splits}")
    print(f"  Results file: {tsv_file}")
    print(f"  Models dir: {models_dir}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print("=" * 70, flush=True)

    # Read original source
    original_source = TRAIN_SCRIPT.read_text()

    # Create backup
    shutil.copy2(TRAIN_SCRIPT, BACKUP_FILE)
    print(f"\nBackup saved to {BACKUP_FILE}", flush=True)

    # Register cleanup
    def cleanup():
        restore_script(BACKUP_FILE, TRAIN_SCRIPT)

    atexit.register(cleanup)

    # Load completed experiments
    completed = load_completed(tsv_file)
    if completed:
        print(f"\nFound {len(completed)} completed experiments, resuming...", flush=True)
        for eid, info in completed.items():
            print(f"  {eid}: {info['name']} -> {info['status']} "
                  f"(dist={info['val_dist']})", flush=True)

    # Each experiment patches the ORIGINAL source independently.
    # Winning experiments update the resume checkpoint (model weights carry the improvement).

    # Determine current best dist and resume checkpoint
    best_dist = baseline_dist
    resume_ckpt = args.resume
    for exp in EXPERIMENTS:
        eid = exp["id"]
        if eid in completed and completed[eid]["status"] == "KEPT":
            if completed[eid]["val_dist"] is not None and completed[eid]["val_dist"] < best_dist:
                best_dist = completed[eid]["val_dist"]
                # Use this experiment's checkpoint if it exists
                exp_ckpt = models_dir / f"corner_hybrid_exp_{eid}.pt"
                if exp_ckpt.exists():
                    resume_ckpt = str(exp_ckpt)

    print(f"\nCurrent best val_dist: {best_dist:.6f}", flush=True)
    print(f"Resume checkpoint: {resume_ckpt}", flush=True)

    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        for exp in EXPERIMENTS:
            exp_id = exp["id"]
            name = exp["name"]

            # Skip completed
            if exp_id in completed:
                print(f"\n[{exp_id}] {name} — SKIPPED (already completed)", flush=True)
                continue

            print(f"\n{'='*70}")
            print(f"[{exp_id}] {name}")
            print(f"  Current best: {best_dist:.6f}")
            print(f"  Resume: {Path(resume_ckpt).name}")
            print(f"  Patches: {len(exp['patches'])}", flush=True)

            start_time = time.time()
            val_dist, output = run_experiment(
                exp, original_source,
                epochs=args.epochs,
                default_batch_size=args.batch_size,
                timeout=args.timeout,
                resume_checkpoint=resume_ckpt,
                input_mode=args.input_mode,
                train_splits=args.train_splits,
                val_splits=args.val_splits,
                models_dir=models_dir,
            )
            elapsed = time.time() - start_time

            if val_dist is None:
                status = "ERROR"
                print(f"  RESULT: ERROR ({output[:100]})", flush=True)
                log_result(tsv_file, exp_id, name, status, None, elapsed, output[:200])
            elif val_dist < best_dist:
                status = "KEPT"
                improvement = best_dist - val_dist
                pct = improvement / best_dist * 100
                print(f"  RESULT: KEPT! val_dist={val_dist:.6f} "
                      f"(improved by {improvement:.6f}, {pct:.1f}%)", flush=True)

                # Update best and resume checkpoint
                best_dist = val_dist
                src_ckpt = models_dir / "best_corner_hybrid.pt"
                dst_ckpt = models_dir / f"corner_hybrid_exp_{exp_id}.pt"
                if src_ckpt.exists():
                    shutil.copy2(src_ckpt, dst_ckpt)
                    resume_ckpt = str(dst_ckpt)
                    print(f"  Checkpoint saved: {dst_ckpt.name}", flush=True)

                log_result(tsv_file, exp_id, name, status, val_dist, elapsed)
            else:
                status = "DISCARDED"
                diff = val_dist - best_dist
                print(f"  RESULT: DISCARDED val_dist={val_dist:.6f} "
                      f"(worse by {diff:.6f})", flush=True)
                log_result(tsv_file, exp_id, name, status, val_dist, elapsed)

            # Always restore original before next experiment
            restore_script(BACKUP_FILE, TRAIN_SCRIPT)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!", flush=True)
    finally:
        # Restore original script
        restore_script(BACKUP_FILE, TRAIN_SCRIPT)
        print(f"\nScript restored from backup.", flush=True)

    # Print summary
    print(f"\n{'='*70}")
    print("AUTORESEARCH SUMMARY")
    print(f"{'='*70}")
    completed = load_completed(tsv_file)
    kept = [e for e in completed.values() if e["status"] == "KEPT"]
    discarded = [e for e in completed.values() if e["status"] == "DISCARDED"]
    errors = [e for e in completed.values() if e["status"] == "ERROR"]
    print(f"  Total: {len(completed)} | Kept: {len(kept)} | "
          f"Discarded: {len(discarded)} | Errors: {len(errors)}")
    print(f"  Best val_dist: {best_dist:.6f}")
    if kept:
        print("\n  Winning experiments:")
        for eid, info in completed.items():
            if info["status"] == "KEPT":
                print(f"    {eid}: {info['name']} -> {info['val_dist']:.6f}")
    print(f"\nResults: {tsv_file}", flush=True)


if __name__ == "__main__":
    main()
