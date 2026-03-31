#!/usr/bin/env python3
"""
Prepare and launch an autoresearch v3 loop for a specific track.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
REPO_ROOT = TRAINING_DIR.parent
DEFAULT_WORKTREE_ROOT = Path("/tmp/codexchess_autoresearch")
TRACK_DEFAULTS: Dict[str, Dict[str, object]] = {
    "resnet_coords": {
        "initial_description": "baseline",
        "baseline_commit_message": "baseline: resnet coords template",
        "time_budget_s": 1800.0,
        "eval_interval_s": 180.0,
    },
    "unet_dual_head": {
        "initial_description": "baseline dual-head unet",
        "baseline_commit_message": "baseline: unet dual-head template",
        "time_budget_s": 3600.0,
        "eval_interval_s": 300.0,
    },
}


def run(cmd: List[str], *, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def git(repo_root: Path, args: List[str], *, check: bool = True) -> str:
    return run(["git", *args], cwd=repo_root, check=check)


def compute_run_name(tag: str, track: str) -> str:
    return f"{tag}-{track}"


def default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def worktree_from_inputs(*, worktree: Optional[str], worktree_root: str, tag: str, track: str) -> Path:
    if worktree:
        return Path(worktree).expanduser().resolve()
    return Path(worktree_root).expanduser().resolve() / compute_run_name(tag, track)


def parse_changed_files(status_output: str) -> List[str]:
    files = []
    for line in status_output.splitlines():
        if len(line) < 4:
            continue
        files.append(line[3:])
    return files


def ensure_llm_available(backend: str) -> None:
    if backend == "codex":
        status = subprocess.run(["codex", "login", "status"], capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError(
                f"codex login status failed:\nstdout:\n{status.stdout}\nstderr:\n{status.stderr}"
            )
        login_text = f"{status.stdout}\n{status.stderr}"
        if "Logged in" not in login_text:
            raise RuntimeError("codex is not logged in")
    else:
        status = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError("claude CLI not found or not working")


def init_worktree(track: str, tag: str, worktree_root: str) -> Path:
    root = Path(worktree_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(THIS_DIR / "init_run.py"),
            "--track",
            track,
            "--tag",
            tag,
            "--worktree-root",
            str(root),
        ],
        cwd=REPO_ROOT,
    )
    return root / compute_run_name(tag, track)


def ensure_expected_track(worktree: Path, expected_track: str) -> None:
    track_path = worktree / "training" / "autoresearch_v3" / "runs" / infer_tag_from_branch(worktree) / "track.txt"
    if not track_path.exists():
        return
    actual_track = track_path.read_text().strip()
    if actual_track and actual_track != expected_track:
        raise RuntimeError(
            f"Worktree track mismatch: expected {expected_track}, found {actual_track}"
        )


def current_branch(worktree: Path) -> str:
    return git(worktree, ["rev-parse", "--abbrev-ref", "HEAD"])


def infer_tag_from_branch(worktree: Path) -> str:
    branch = current_branch(worktree)
    if branch.startswith("autoresearch-"):
        return branch[len("autoresearch-"):]
    raise RuntimeError("Refusing to operate outside an autoresearch-* branch")


def commit_baseline_if_needed(worktree: Path, track: str) -> bool:
    # Use subprocess directly to preserve raw porcelain format (leading spaces matter)
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(worktree),
        capture_output=True,
        text=True,
        check=True,
    )
    status_output = result.stdout
    changed_files = parse_changed_files(status_output)
    if not changed_files:
        return False

    auto_dir_rel = "training/autoresearch_v3/"
    unexpected = [path for path in changed_files if not path.startswith(auto_dir_rel)]
    if unexpected:
        raise RuntimeError(
            "Worktree has unexpected uncommitted changes:\n" + "\n".join(unexpected)
        )

    git(worktree, ["add", "training/autoresearch_v3"])
    git(
        worktree,
        ["commit", "-m", str(TRACK_DEFAULTS[track]["baseline_commit_message"])],
    )
    return True


def build_controller_command(
    *,
    worktree: Path,
    track: str,
    args: argparse.Namespace,
) -> List[str]:
    defaults = TRACK_DEFAULTS[track]
    initial_description = args.initial_description or str(defaults["initial_description"])
    time_budget_s = args.time_budget_s
    if time_budget_s is None:
        time_budget_s = float(defaults["time_budget_s"])
    eval_interval_s = args.eval_interval_s
    if eval_interval_s is None:
        eval_interval_s = float(defaults["eval_interval_s"])

    cmd = [
        sys.executable,
        str(THIS_DIR / "llm_controller.py"),
        "--worktree",
        str(worktree),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--initial-description",
        initial_description,
        "--poll-interval-s",
        str(args.poll_interval_s),
        "--time-budget-s",
        str(time_budget_s),
        "--eval-interval-s",
        str(eval_interval_s),
    ]
    if args.max_iterations is not None:
        cmd.extend(["--max-iterations", str(args.max_iterations)])
    if args.resume:
        cmd.extend(["--resume", str(Path(args.resume).expanduser().resolve())])
    if args.reset_on_discard:
        cmd.append("--reset-on-discard")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and launch an autoresearch v3 loop")
    parser.add_argument(
        "--track",
        choices=sorted(TRACK_DEFAULTS.keys()),
        default="unet_dual_head",
        help="Track to initialize or launch",
    )
    parser.add_argument("--tag", default=None, help="Run tag; defaults to the current UTC date")
    parser.add_argument("--worktree", default=None, help="Existing autoresearch worktree to reuse")
    parser.add_argument(
        "--worktree-root",
        default=str(DEFAULT_WORKTREE_ROOT),
        help="Parent directory used when creating a fresh worktree",
    )
    parser.add_argument("--backend", default="claude", choices=["claude", "codex"], help="LLM backend to use")
    parser.add_argument("--model", default="sonnet", help="Model to use (e.g. sonnet, opus, gpt-5.4)")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high", "max"],
        help="LLM reasoning effort",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Optional controller iteration cap")
    parser.add_argument("--poll-interval-s", type=float, default=30.0, help="Controller poll interval")
    parser.add_argument("--initial-description", default=None, help="Initial benchmark description")
    parser.add_argument("--resume", default=None, help="Optional resume checkpoint for run_commit.py")
    parser.add_argument("--time-budget-s", type=float, default=None, help="Per-run training wall-clock budget")
    parser.add_argument("--eval-interval-s", type=float, default=None, help="Per-run evaluation interval")
    parser.add_argument("--reset-on-discard", action="store_true", default=False, help="Reset HEAD^ after discard")
    parser.add_argument("--prepare-only", action="store_true", default=False, help="Initialize and baseline-commit only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or default_tag()

    if args.worktree:
        worktree = worktree_from_inputs(
            worktree=args.worktree,
            worktree_root=args.worktree_root,
            tag=tag,
            track=args.track,
        )
        if not worktree.exists():
            raise RuntimeError(f"Worktree does not exist: {worktree}")
    else:
        worktree = init_worktree(args.track, tag, args.worktree_root)

    if not (worktree / "training" / "autoresearch_v3").exists():
        raise RuntimeError(f"Not an autoresearch v3 worktree: {worktree}")

    ensure_expected_track(worktree, args.track)
    created_baseline_commit = commit_baseline_if_needed(worktree, args.track)
    controller_cmd = build_controller_command(worktree=worktree, track=args.track, args=args)

    print(f"track: {args.track}")
    print(f"worktree: {worktree}")
    if created_baseline_commit:
        print("baseline commit: created")
    else:
        print("baseline commit: unchanged")
    print("controller command:")
    print("  " + " ".join(controller_cmd))

    if args.prepare_only:
        return

    ensure_llm_available(args.backend)
    subprocess.run(controller_cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
