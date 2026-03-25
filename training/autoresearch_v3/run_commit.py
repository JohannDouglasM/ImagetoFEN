#!/usr/bin/env python3
"""
Run the current committed candidate under the fixed harness, log an experiment card,
and optionally apply keep/promising/discard to the branch.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
RUNS_DIR = THIS_DIR / "runs"
RESULTS_HEADER = (
    "commit\tparent\tstatus\tprimary_mean\trecovered_mean\trecovered_max\t"
    "combined_max\tcombined_p95\tdescription\n"
)


def run(cmd, *, cwd):
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def git(args, *, cwd):
    return run(["git", *args], cwd=cwd)


def require_clean_worktree(repo_root):
    raw = git(["status", "--porcelain"], cwd=repo_root)
    if raw.strip():
        raise RuntimeError(
            "Worktree is dirty. Commit the candidate change before running the benchmark."
        )


def current_branch(repo_root):
    return git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)


def current_commit(repo_root):
    return git(["rev-parse", "--short=7", "HEAD"], cwd=repo_root)


def parent_commit(repo_root):
    try:
        return git(["rev-parse", "--short=7", "HEAD^"], cwd=repo_root)
    except RuntimeError:
        return ""


def infer_tag(branch_name):
    if branch_name.startswith("autoresearch-"):
        return branch_name[len("autoresearch-"):]
    return branch_name.replace("/", "_")


def ensure_results_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(RESULTS_HEADER)


def load_best_keep_metrics(results_path):
    if not results_path.exists():
        return None
    best = None
    with open(results_path) as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "keep":
                continue
            current = {
                "commit": parts[0],
                "primary_mean": float(parts[3]),
                "recovered_mean": float(parts[4]),
                "recovered_max": float(parts[5]),
                "combined_max": float(parts[6]),
                "combined_p95": float(parts[7]),
            }
            if best is None or current["primary_mean"] < best["primary_mean"]:
                best = current
    return best


def decide(metrics, best, *, improve_epsilon, recovered_guard, recovered_max_guard, promising_margin):
    if best is None:
        return "keep", "baseline"

    primary_gain = best["primary_mean"] - metrics["primary_mean"]
    recovered_regression = metrics["recovered_mean"] - best["recovered_mean"]
    recovered_max_regression = metrics["recovered_max"] - best["recovered_max"]

    if recovered_regression > recovered_guard:
        return "discard", f"recovered mean regressed by {recovered_regression:.6f}"
    if recovered_max_regression > recovered_max_guard:
        return "discard", f"recovered max regressed by {recovered_max_regression:.6f}"
    if primary_gain >= improve_epsilon:
        return "keep", f"primary improved by {primary_gain:.6f}"
    if metrics["primary_mean"] <= best["primary_mean"] + promising_margin:
        return "promising", f"within {promising_margin:.6f} of best without violating guards"
    return "discard", f"primary gain {primary_gain:.6f} below epsilon {improve_epsilon:.6f}"


def append_results(results_path, row):
    ensure_results_file(results_path)
    with open(results_path, "a") as handle:
        handle.write(row)


def save_experiment_card(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_patch(repo_root, destination):
    patch = git(["diff", "HEAD^", "HEAD", "--", "training/autoresearch_v3/candidate.py"], cwd=repo_root)
    destination.write_text(patch + ("\n" if patch and not patch.endswith("\n") else ""))


def copy_candidate(repo_root, destination):
    destination.write_text((repo_root / "training" / "autoresearch_v3" / "candidate.py").read_text())


def run_and_stream(cmd, *, cwd, log_path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return proc.wait()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current committed candidate experiment")
    parser.add_argument("--description", required=True, help="Short hypothesis text for the experiment log")
    parser.add_argument("--candidate", type=str, default="training/autoresearch_v3/candidate.py")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--time-budget-s", type=float, default=1800.0)
    parser.add_argument("--eval-interval-s", type=float, default=180.0)
    parser.add_argument("--improve-epsilon", type=float, default=0.0001)
    parser.add_argument("--recovered-guard", type=float, default=0.0015)
    parser.add_argument("--recovered-max-guard", type=float, default=0.05)
    parser.add_argument("--promising-margin", type=float, default=0.0006)
    parser.add_argument("--apply-decision", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = TRAINING_DIR.parent
    branch = current_branch(repo_root)
    if not branch.startswith("autoresearch-"):
        raise RuntimeError("Refusing to run outside an autoresearch-* branch")

    require_clean_worktree(repo_root)

    tag = infer_tag(branch)
    run_dir = RUNS_DIR / tag
    results_path = run_dir / "results.tsv"
    jsonl_path = run_dir / "experiments.jsonl"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = current_commit(repo_root)
    parent = parent_commit(repo_root)
    artifact_dir = run_dir / "artifacts" / f"{timestamp}_{commit}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifact_dir / "metrics.json"
    checkpoint_path = artifact_dir / "best.pt"
    status_path = artifact_dir / "status.txt"
    log_path = artifact_dir / "run.log"
    patch_path = artifact_dir / "candidate.patch"
    candidate_snapshot = artifact_dir / "candidate.py"

    cmd = [
        sys.executable,
        str(THIS_DIR / "fixed_harness.py"),
        "--candidate",
        str((repo_root / args.candidate).resolve()),
        "--time-budget-s",
        str(args.time_budget_s),
        "--eval-interval-s",
        str(args.eval_interval_s),
        "--json-out",
        str(metrics_path),
        "--checkpoint-out",
        str(checkpoint_path),
        "--status-file",
        str(status_path),
    ]
    if args.resume:
        cmd.extend(["--resume", str(Path(args.resume).resolve())])

    print(f"artifact_dir: {artifact_dir}")
    print(f"status_path: {status_path}")
    print(f"log_path: {log_path}")
    print("streaming harness output...", flush=True)

    returncode = run_and_stream(cmd, cwd=repo_root, log_path=log_path)
    copy_candidate(repo_root, candidate_snapshot)
    save_patch(repo_root, patch_path)

    card = {
        "timestamp_utc": timestamp,
        "branch": branch,
        "commit": commit,
        "parent_commit": parent,
        "description": args.description,
        "command": cmd,
        "returncode": returncode,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
        "status_path": str(status_path),
        "patch_path": str(patch_path),
        "candidate_snapshot": str(candidate_snapshot),
    }

    if returncode != 0 or not metrics_path.exists():
        status = "crash"
        reason = f"benchmark failed with exit code {returncode}"
        row = f"{commit}\t{parent}\t{status}\t0.000000\t0.000000\t0.000000\t0.000000\t0.000000\t{args.description}\n"
        append_results(results_path, row)
        card["status"] = status
        card["decision_reason"] = reason
        save_experiment_card(jsonl_path, card)
        print(f"{status}: {reason}")
        if args.apply_decision:
            git(["reset", "--hard", "HEAD^"], cwd=repo_root)
            print("branch reset to parent commit")
        return

    metrics = json.loads(metrics_path.read_text())
    best = load_best_keep_metrics(results_path)
    flat_metrics = {
        "primary_mean": float(metrics["primary_metric"]),
        "recovered_mean": float(metrics["split_metrics"]["chess_dataset_recovered:val"]["mean_dist"]),
        "recovered_max": float(metrics["split_metrics"]["chess_dataset_recovered:val"]["max_dist"]),
        "combined_max": float(metrics["split_metrics"]["combined"]["max_dist"]),
        "combined_p95": float(metrics["split_metrics"]["combined"]["p95_dist"]),
    }
    status, reason = decide(
        flat_metrics,
        best,
        improve_epsilon=args.improve_epsilon,
        recovered_guard=args.recovered_guard,
        recovered_max_guard=args.recovered_max_guard,
        promising_margin=args.promising_margin,
    )

    row = (
        f"{commit}\t{parent}\t{status}\t{flat_metrics['primary_mean']:.6f}\t"
        f"{flat_metrics['recovered_mean']:.6f}\t{flat_metrics['recovered_max']:.6f}\t"
        f"{flat_metrics['combined_max']:.6f}\t{flat_metrics['combined_p95']:.6f}\t"
        f"{args.description}\n"
    )
    append_results(results_path, row)

    card["status"] = status
    card["decision_reason"] = reason
    card["metrics"] = metrics
    save_experiment_card(jsonl_path, card)

    print(f"status: {status}")
    print(f"reason: {reason}")
    print(f"primary_mean: {flat_metrics['primary_mean']:.6f}")
    print(f"recovered_mean: {flat_metrics['recovered_mean']:.6f}")
    print(f"recovered_max: {flat_metrics['recovered_max']:.6f}")
    print(f"artifact_dir: {artifact_dir}")

    if args.apply_decision and status != "keep":
        git(["reset", "--hard", "HEAD^"], cwd=repo_root)
        print("branch reset to parent commit")


if __name__ == "__main__":
    main()
