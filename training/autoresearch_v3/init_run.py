#!/usr/bin/env python3
"""
Create a dedicated git worktree for an autoresearch v3 run.

This avoids interfering with the main repo state or any long-running local jobs.
"""

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
REPO_ROOT = TRAINING_DIR.parent
TEMPLATES_DIR = THIS_DIR / "templates"
TRACK_TO_TEMPLATE = {
    "resnet_coords": TEMPLATES_DIR / "resnet_coords.py",
    "unet_dual_head": TEMPLATES_DIR / "unet_dual_head.py",
}


def run(cmd, *, cwd):
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize an autoresearch v3 worktree")
    parser.add_argument("--tag", type=str, default=None, help="Run tag, defaults to current UTC date")
    parser.add_argument(
        "--track",
        type=str,
        choices=sorted(TRACK_TO_TEMPLATE.keys()),
        default="resnet_coords",
        help="Candidate family to initialize in the worktree",
    )
    parser.add_argument(
        "--worktree-root",
        type=str,
        default="/tmp/codexchess_autoresearch",
        help="Parent directory for dedicated run worktrees",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"{tag}-{args.track}"
    branch = f"autoresearch-{run_name}"
    worktree_dir = Path(args.worktree_root).expanduser().resolve() / run_name
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    if worktree_dir.exists():
        raise RuntimeError(f"Worktree path already exists: {worktree_dir}")

    existing_branches = run(["git", "branch", "--list", branch], cwd=REPO_ROOT)
    if existing_branches.strip():
        raise RuntimeError(f"Branch already exists: {branch}")

    run(
        ["git", "worktree", "add", "-b", branch, str(worktree_dir), "HEAD"],
        cwd=REPO_ROOT,
    )

    dst_autoresearch_dir = worktree_dir / "training" / "autoresearch_v3"
    dst_autoresearch_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        THIS_DIR,
        dst_autoresearch_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("runs", "__pycache__", "*.pyc"),
    )
    candidate_path = dst_autoresearch_dir / "candidate.py"
    shutil.copy2(TRACK_TO_TEMPLATE[args.track], candidate_path)

    run_dir = worktree_dir / "training" / "autoresearch_v3" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.tsv"
    if not results_path.exists():
        results_path.write_text(
            "commit\tparent\tstatus\tprimary_mean\trecovered_mean\trecovered_max\tcombined_max\tcombined_p95\tdescription\n"
        )

    snapshot_path = run_dir / "program_snapshot.md"
    snapshot_path.write_text((worktree_dir / "training" / "autoresearch_v3" / "program.md").read_text())
    (run_dir / "track.txt").write_text(args.track + "\n")

    print(f"branch:       {branch}")
    print(f"track:        {args.track}")
    print(f"worktree:     {worktree_dir}")
    print(f"results file: {results_path}")
    print("")
    print("Next steps:")
    print(f"  cd {worktree_dir}")
    print("  open training/autoresearch_v3/program.md")
    print("  edit only training/autoresearch_v3/candidate.py")
    print("  git commit -am 'baseline or experiment'")
    print("  python3 training/autoresearch_v3/run_commit.py --description 'baseline'")


if __name__ == "__main__":
    main()
