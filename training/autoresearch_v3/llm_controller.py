#!/usr/bin/env python3
"""
LLM-driven autoresearch controller for a v3 worktree.

The controller:
- waits for any active benchmark in the target worktree to finish
- reads the latest experiment history and current candidate
- asks the local `codex` CLI to edit candidate.py in the target worktree
- commits the new candidate
- runs the benchmark
- repeats until interrupted or max iterations is reached
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MAX_RESULTS_CONTEXT = 8
MAX_EXPERIMENTS_CONTEXT = 4
MAX_LOG_TAIL_LINES = 40
MAX_FAMILY_HISTORY = 16
MAX_PROPOSAL_ATTEMPTS = 3
SAME_FAMILY_IMPROVEMENT_THRESHOLD = 0.0003
PRIMARY_PRIORITY_FAMILIES = [
    "learning_rate",
    "batch_size",
    "optimizer",
    "target_parameterization",
]
SECONDARY_PRIORITY_FAMILIES = [
    "loss_shaping",
    "weight_decay",
]
KNOWN_FAMILIES = [
    "learning_rate",
    "batch_size",
    "optimizer",
    "target_parameterization",
    "loss_shaping",
    "weight_decay",
    "jpeg_augmentation",
    "non_jpeg_augmentation",
    "decode_calibration",
    "model_head",
    "input_representation",
    "other",
]
FAMILY_DISPLAY_NAMES = {
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "optimizer": "optimizer",
    "weight_decay": "weight decay",
    "jpeg_augmentation": "JPEG augmentation",
    "non_jpeg_augmentation": "non-JPEG augmentation",
    "decode_calibration": "decode calibration",
    "loss_shaping": "loss shaping",
    "model_head": "model/head",
    "input_representation": "input representation",
    "target_parameterization": "target parameterization",
    "other": "other",
}


def family_sort_key(family: str) -> Tuple[int, int, str]:
    if family in PRIMARY_PRIORITY_FAMILIES:
        return (0, PRIMARY_PRIORITY_FAMILIES.index(family), family)
    if family in SECONDARY_PRIORITY_FAMILIES:
        return (1, SECONDARY_PRIORITY_FAMILIES.index(family), family)
    if family == "other":
        return (3, 0, family)
    return (2, KNOWN_FAMILIES.index(family), family)


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


def current_branch(repo_root: Path) -> str:
    return git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])


def current_commit(repo_root: Path) -> str:
    return git(repo_root, ["rev-parse", "--short=7", "HEAD"])


def infer_tag(branch_name: str) -> str:
    if branch_name.startswith("autoresearch-"):
        return branch_name[len("autoresearch-"):]
    return branch_name.replace("/", "_")


def ensure_clean_worktree(repo_root: Path) -> None:
    raw = git(repo_root, ["status", "--porcelain"])
    if raw.strip():
        raise RuntimeError(
            "Target worktree is dirty. The controller only runs from a clean worktree."
        )


def active_harness_commands(repo_root: Path) -> List[str]:
    repo_str = str(repo_root.resolve())
    output = run(["ps", "-axo", "pid=,command="], cwd=repo_root)
    commands = []
    for line in output.splitlines():
        if repo_str not in line:
            continue
        if "training/autoresearch_v3/fixed_harness.py" not in line:
            continue
        commands.append(line.strip())
    return commands


def wait_for_active_benchmark(repo_root: Path, poll_interval_s: float) -> None:
    while True:
        commands = active_harness_commands(repo_root)
        if not commands:
            return
        print("benchmark already active in target worktree; waiting...", flush=True)
        for command in commands:
            print(f"  {command}", flush=True)
        time.sleep(poll_interval_s)


def read_tsv_tail(path: Path, max_lines: int) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    if len(lines) <= max_lines:
        return lines
    return [lines[0], *lines[-(max_lines - 1):]]


def load_experiment_cards(path: Path, max_items: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    cards = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cards.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return cards[-max_items:]


def summarize_card(card: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "commit": card.get("commit"),
        "parent_commit": card.get("parent_commit"),
        "description": card.get("description"),
        "status": card.get("status"),
    }
    metrics = card.get("metrics")
    if isinstance(metrics, dict):
        split_metrics = metrics.get("split_metrics", {})
        summary["primary_metric"] = metrics.get("primary_metric")
        for split_name in ["combined", "chess_dataset_recovered:val", "chessred2k:val"]:
            if split_name in split_metrics:
                summary[split_name] = {
                    "mean_dist": split_metrics[split_name].get("mean_dist"),
                    "max_dist": split_metrics[split_name].get("max_dist"),
                    "p95_dist": split_metrics[split_name].get("p95_dist"),
                }
    return summary


def extract_primary_metric(card: Dict[str, Any]) -> Optional[float]:
    metrics = card.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("primary_metric")
        if isinstance(value, (int, float)):
            return float(value)
    value = card.get("primary_metric")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def infer_experiment_family(*parts: str) -> str:
    text = " ".join(part for part in parts if part).lower()
    if any(
        token in text
        for token in [
            "batch size",
            "batchsize",
            "accumulation",
            "gradient accumulation",
            "microbatch",
            "micro-batch",
        ]
    ):
        return "batch_size"
    if any(
        token in text
        for token in [
            "optimizer",
            "adamw",
            "adam",
            "sgd",
            "rmsprop",
            "lion",
            "momentum",
            "betas",
        ]
    ):
        return "optimizer"
    if any(
        token in text
        for token in [
            "weight decay",
            "weight_decay",
            "wd=",
            "wd ",
            "decoupled decay",
            "l2 regularization",
            "l2 penalty",
        ]
    ):
        return "weight_decay"
    if any(
        token in text
        for token in [
            "learning rate",
            "resume lr",
            "base lr",
            "lr scale",
            "lr=",
            " lr ",
            "warmup",
            "scheduler",
            "cosine",
            "anneal",
            "annealing",
            "t_max",
        ]
    ):
        return "learning_rate"
    if any(token in text for token in ["jpeg", "quality floor", "quality ceiling", "q62", "compression"]):
        return "jpeg_augmentation"
    if any(token in text for token in ["blur", "shadow", "resize", "clahe", "gamma", "contrast", "brightness"]):
        return "non_jpeg_augmentation"
    if any(token in text for token in ["decode", "outward", "quadrilateral", "quad", "calibration", "expand"]):
        return "decode_calibration"
    if any(token in text for token in ["loss", "smooth_l1", "huber", "weighting"]):
        return "loss_shaping"
    if any(token in text for token in ["dropout", "head", "backbone", "resnet", "fc", "hidden dim"]):
        return "model_head"
    if any(token in text for token in ["gray_edges", "hybrid", "heatmap input", "input mode", "edge"]):
        return "input_representation"
    if any(token in text for token in ["target", "parameterization", "parametrization", "center+offset"]):
        return "target_parameterization"
    return "other"


def build_family_policy(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = []
    for card in cards:
        description = card.get("description")
        if not isinstance(description, str) or not description.strip():
            continue
        usable.append(
            {
                "commit": card.get("commit"),
                "description": description,
                "status": card.get("status"),
                "family": infer_experiment_family(description),
                "primary_metric": extract_primary_metric(card),
            }
        )

    if not usable:
        return {
            "recent_families": [],
            "current_family": None,
            "current_streak": 0,
            "streak_improvement": None,
            "soft_avoid_family": None,
            "forced_switch_family": None,
            "suggested_families": [family for family in KNOWN_FAMILIES if family != "other"][:4],
        }

    current_family = usable[-1]["family"]
    streak = []
    for item in reversed(usable):
        if item["family"] != current_family:
            break
        streak.append(item)
    streak.reverse()

    prior_metric = None
    for item in reversed(usable[:-len(streak)] if len(streak) < len(usable) else []):
        metric = item["primary_metric"]
        if isinstance(metric, float) and metric > 0.0:
            prior_metric = metric
            break

    best_in_streak = None
    for item in streak:
        metric = item["primary_metric"]
        if isinstance(metric, float) and metric > 0.0:
            best_in_streak = metric if best_in_streak is None else min(best_in_streak, metric)

    streak_improvement = None
    if prior_metric is not None and best_in_streak is not None:
        streak_improvement = prior_metric - best_in_streak

    forced_switch_family = current_family if len(streak) >= 3 else None
    soft_avoid_family = None
    if forced_switch_family is None and len(streak) >= 2:
        if streak_improvement is None or streak_improvement <= SAME_FAMILY_IMPROVEMENT_THRESHOLD:
            soft_avoid_family = current_family

    recent_unique = []
    for item in reversed(usable[-6:]):
        family = item["family"]
        if family not in recent_unique:
            recent_unique.append(family)
    suggested_families = [
        family
        for family in KNOWN_FAMILIES
        if family not in recent_unique and family != current_family and family != "other"
    ]
    if not suggested_families:
        suggested_families = [
            family
            for family in KNOWN_FAMILIES
            if family != current_family and family != "other"
        ]
    suggested_families = sorted(suggested_families, key=family_sort_key)

    return {
        "recent_families": usable[-8:],
        "current_family": current_family,
        "current_streak": len(streak),
        "streak_improvement": streak_improvement,
        "soft_avoid_family": soft_avoid_family,
        "forced_switch_family": forced_switch_family,
        "suggested_families": suggested_families[:4],
    }


def format_family_policy(policy: Dict[str, Any]) -> str:
    recent = policy["recent_families"]
    if not recent:
        return "- no recent family history yet"

    labels = [
        f"{item['commit']}={FAMILY_DISPLAY_NAMES.get(item['family'], item['family'])}"
        for item in recent
        if item.get("commit")
    ]
    lines = [
        f"- recent families (oldest->newest): {', '.join(labels)}",
        (
            f"- current family streak: "
            f"{FAMILY_DISPLAY_NAMES.get(policy['current_family'], policy['current_family'])} "
            f"x{policy['current_streak']}"
        ),
    ]
    improvement = policy["streak_improvement"]
    if improvement is None:
        lines.append(
            f"- streak improvement vs prior baseline: unavailable "
            f"(threshold {SAME_FAMILY_IMPROVEMENT_THRESHOLD:.6f})"
        )
    else:
        lines.append(
            f"- streak improvement vs prior baseline: {improvement:.6f} "
            f"(threshold {SAME_FAMILY_IMPROVEMENT_THRESHOLD:.6f})"
        )
    if policy["forced_switch_family"]:
        lines.append(
            "- hard rule this iteration: choose a different family than "
            f"{FAMILY_DISPLAY_NAMES.get(policy['forced_switch_family'], policy['forced_switch_family'])}"
        )
    elif policy["soft_avoid_family"]:
        lines.append(
            "- strong preference this iteration: avoid another "
            f"{FAMILY_DISPLAY_NAMES.get(policy['soft_avoid_family'], policy['soft_avoid_family'])} "
            "experiment unless the evidence is much stronger than usual"
        )
    suggested = [
        FAMILY_DISPLAY_NAMES.get(family, family)
        for family in policy["suggested_families"]
    ]
    if suggested:
        lines.append(f"- suggested alternative families: {', '.join(suggested)}")
    return "\n".join(lines)


def family_policy_violation(policy: Dict[str, Any], proposal_family: str) -> Optional[str]:
    forced = policy.get("forced_switch_family")
    if forced and proposal_family == forced:
        return (
            "the last three experiments were already in the "
            f"{FAMILY_DISPLAY_NAMES.get(forced, forced)} family, so the next one must switch"
        )
    return None


def latest_log_tail(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return ""
    latest = cards[-1]
    log_path_raw = latest.get("log_path")
    if not log_path_raw:
        return ""
    log_path = Path(log_path_raw)
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    signal_prefixes = (
        "eval ",
        "  chessred2k:val:",
        "  chess_dataset_recovered:val:",
        "  combined:",
        "  -> New best",
        "primary_mean_dist:",
        "combined:",
        "chessred2k:val:",
        "chess_dataset_recovered:val:",
        "FATAL:",
        "Early stop:",
    )
    selected = [line for line in lines if line.startswith(signal_prefixes)]
    if not selected:
        selected = lines[-MAX_LOG_TAIL_LINES:]
    else:
        selected = selected[-MAX_LOG_TAIL_LINES:]
    return "\n".join(selected)


def head_touches_candidate(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD^", "HEAD", "--", "training/autoresearch_v3/candidate.py"],
        cwd=str(repo_root),
    )
    return result.returncode == 1


def load_benchmarked_commits(experiments_path: Path) -> set:
    commits = set()
    for card in load_experiment_cards(experiments_path, 1000000):
        commit = card.get("commit")
        if commit:
            commits.add(commit)
    return commits


def write_controller_artifacts(
    controller_dir: Path,
    stem: str,
    prompt: str,
    raw_response: Dict[str, Any],
    stderr_text: str,
) -> None:
    controller_dir.mkdir(parents=True, exist_ok=True)
    (controller_dir / f"{stem}.prompt.txt").write_text(prompt)
    (controller_dir / f"{stem}.response.json").write_text(json.dumps(raw_response, indent=2) + "\n")
    (controller_dir / f"{stem}.stderr.txt").write_text(stderr_text)


def build_prompt(
    *,
    branch: str,
    track: str,
    current_commit_id: str,
    results_tail: List[str],
    cards: List[Dict[str, Any]],
    latest_log: str,
    run_dir: Path,
    family_policy: Dict[str, Any],
    retry_feedback: Optional[str],
) -> str:
    experiment_summaries = [summarize_card(card) for card in cards]
    retry_block = ""
    if retry_feedback:
        retry_block = f"""
Controller feedback from the last rejected proposal:
{retry_feedback}
"""
    prompt = f"""You are driving an autonomous research loop for a chess corner model.

You must obey these repository rules:
- Edit only training/autoresearch_v3/candidate.py
- Do not change fixed_harness.py, validation logic, metrics, data, or decision rules
- Prefer one coherent hypothesis per iteration
- Prefer smaller changes when evidence is weak
- Optimize combined validation mean corner distance without materially regressing chess_dataset_recovered:val

Checkpoint policy guidance:
- Assume the controller may resume training from the current best checkpoint for local refinements.
- Resume is appropriate for small same-family changes such as mild augmentation, scheduler, loss-weight, or decode-calibration tweaks.
- Be cautious about proposing architecture, representation, or target changes that really need a from-scratch run to be evaluated fairly.
- If a hypothesis seems to require from-scratch training to be meaningful, prefer a different experiment family unless the evidence for that larger change is strong.

Target branch: {branch}
Track: {track}
Current commit: {current_commit_id}
Run directory: {run_dir}

Recent results.tsv:
{chr(10).join(results_tail) if results_tail else "(none yet)"}

Recent experiment summaries:
{json.dumps(experiment_summaries, indent=2)}

Latest run log tail:
{latest_log if latest_log else "(no log tail available)"}

Experiment-family diversity policy:
{format_family_policy(family_policy)}
{retry_block}

Experiment priority guidance:
- Prefer the next experiments in these primary families: learning rate, batch size, optimizer, target parameterization.
- Secondary families are: loss design and weight decay.
- Deprioritize further JPEG, augmentation, and decode-calibration fiddling unless recent evidence is unusually strong.
- When evidence is mixed, choose one of the primary families above instead of another augmentation-style micro-tweak.
- Treat architecture/model-head changes as lower priority than the primary families unless they directly support a target-parameterization hypothesis.

CRITICAL — Prior agent findings (22 experiments, best combined_mean = 0.00537):
A previous research agent ran 22 experiments and found a winning recipe that achieved
0.00537 combined mean distance. You MUST incorporate ALL of these changes:

1. ExponentialLR scheduler (gamma calibrated so LR decays to ~2% by end of training).
   This was the SINGLE BIGGEST improvement, dropping from 0.00655 to 0.00537.
   Do NOT use CosineAnnealing or warmup+cosine — use ExponentialLR.
2. batch_size in range 16–32 (sweet spot; 20–32 all worked well)
3. lr = 3e-4 (do not go lower to 2e-4, that hurt)
4. weight_decay = 0.03 (3x the old default of 0.01; reducing back to 0.01 hurt)
5. coord_loss_weight = 2.0 (doubled from 1.0; strengthens coordinate regression signal)
6. AdamW with amsgrad=True

Things that HURT and must be avoided:
- Linear warmup + cosine decay — worse than ExponentialLR
- Lowering lr to 2e-4 — worse
- soft_argmax_beta=40 — worse
- heatmap_sigma=2.0 (sharper targets) — worse
- Reducing weight_decay back to 0.01 — worse

Start by applying ALL of the winning recipe above as the baseline, then explore
refinements from there. Do not regress any of these settings without strong evidence.

Your task:
- use the prompt-provided history as the default source of truth
- read training/autoresearch_v3/program.md only if the prompt history is insufficient
- read training/autoresearch_v3/EXPERIMENT_PLAN.md only if the prompt history is insufficient
- read training/autoresearch_v3/candidate.py
- inspect additional files in {run_dir} only if needed to resolve a concrete uncertainty
- avoid opening image/png files or large historical artifacts unless absolutely necessary
- edit only training/autoresearch_v3/candidate.py
- do not commit
- do not run the benchmark

Return exactly one JSON object with these keys after you finish editing:
- description: short benchmark description for run_commit.py
- commit_message: short git commit message
- rationale: short explanation of the single hypothesis

Hard constraints:
- only training/autoresearch_v3/candidate.py may be modified
- keep imports reasonable and preserve needed interfaces used by fixed_harness.py
- do not mention any file other than candidate.py in the returned JSON
- do not wrap the JSON in markdown fences
"""
    return prompt


def validate_candidate_source(candidate_source: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(candidate_source)
    try:
        run([sys.executable, "-m", "py_compile", str(temp_path)], cwd=temp_path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def codex_schema_path() -> Path:
    schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "commit_message": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["description", "commit_message", "rationale"],
        "additionalProperties": False,
    }
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    path = Path(handle.name)
    with handle:
        json.dump(schema, handle)
    return path


def stream_lines_to_stdout(pipe, sink: List[str], prefix: str) -> None:
    if pipe is None:
        return
    try:
        for line in pipe:
            sink.append(line)
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()
    finally:
        pipe.close()


def run_codex_exec(repo_root: Path, prompt: str, *, model: str, reasoning_effort: str) -> Tuple[Dict[str, Any], str]:
    schema_path = codex_schema_path()
    try:
        cmd = [
            "codex",
            "exec",
            "-C",
            str(repo_root),
            "--sandbox",
            "workspace-write",
            "--output-schema",
            str(schema_path),
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            prompt,
        ]
        print("starting codex experiment proposal...", flush=True)
        print(f"codex cwd: {repo_root}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines: List[str] = []
        stderr_thread = threading.Thread(
            target=stream_lines_to_stdout,
            args=(proc.stderr, stderr_lines, "[codex] "),
            daemon=True,
        )
        stderr_thread.start()
        stdout_text = proc.stdout.read() if proc.stdout is not None else ""
        if proc.stdout is not None:
            proc.stdout.close()
        returncode = proc.wait()
        stderr_thread.join()
        stderr_text = "".join(stderr_lines)
        if returncode != 0:
            raise RuntimeError(
                f"codex exec failed with exit code {returncode}\n"
                f"stdout:\n{stdout_text}\n"
                f"stderr:\n{stderr_text}"
            )
        try:
            payload = json.loads(stdout_text.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"codex exec returned non-JSON stdout:\n{stdout_text}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("codex exec returned a non-object JSON payload")
        print("codex proposal complete", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
        return payload, stderr_text
    finally:
        schema_path.unlink(missing_ok=True)


def run_claude_exec(repo_root: Path, prompt: str, *, model: str, effort: str) -> Tuple[Dict[str, Any], str]:
    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--tools", "Read,Edit",
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--no-session-persistence",
        prompt,
    ]
    print("starting claude experiment proposal...", flush=True)
    print(f"claude cwd: {repo_root}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: List[str] = []
    stderr_thread = threading.Thread(
        target=stream_lines_to_stdout,
        args=(proc.stderr, stderr_lines, "[claude] "),
        daemon=True,
    )
    stderr_thread.start()
    stdout_text = proc.stdout.read() if proc.stdout is not None else ""
    if proc.stdout is not None:
        proc.stdout.close()
    returncode = proc.wait()
    stderr_thread.join()
    stderr_text = "".join(stderr_lines)
    if returncode != 0:
        raise RuntimeError(
            f"claude exec failed with exit code {returncode}\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )
    try:
        envelope = json.loads(stdout_text.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"claude returned non-JSON stdout:\n{stdout_text}"
        ) from exc

    # claude --output-format json wraps the result in an envelope
    if isinstance(envelope, dict) and envelope.get("is_error"):
        raise RuntimeError(f"claude returned an error: {envelope.get('result', envelope)}")

    # Extract the payload from claude's JSON envelope
    payload = None
    if isinstance(envelope, dict) and "result" in envelope:
        result_text = envelope["result"]
        # The result text should contain a JSON object; extract it
        if isinstance(result_text, str):
            # Try to find JSON in the text (claude may wrap it in markdown or text)
            import re
            # Try raw parse first
            try:
                payload = json.loads(result_text.strip())
            except (json.JSONDecodeError, TypeError):
                pass
            # Try extracting from markdown code fences
            if payload is None:
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', result_text, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(1).strip())
                    except json.JSONDecodeError:
                        pass
            # Try finding a JSON object in the text
            if payload is None:
                match = re.search(r'\{[^{}]*"description"[^{}]*\}', result_text, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
    if payload is None:
        payload = envelope

    if not isinstance(payload, dict):
        raise RuntimeError(f"claude returned a non-object JSON payload: {stdout_text[:500]}")
    print("claude proposal complete", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return payload, stderr_text


def run_llm_exec(repo_root: Path, prompt: str, *, model: str, reasoning_effort: str, backend: str) -> Tuple[Dict[str, Any], str]:
    if backend == "claude":
        return run_claude_exec(repo_root, prompt, model=model, effort=reasoning_effort)
    else:
        return run_codex_exec(repo_root, prompt, model=model, reasoning_effort=reasoning_effort)


def changed_files(repo_root: Path) -> List[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    raw = result.stdout
    files = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        files.append(line[3:])
    return files


def restore_candidate_from_head(repo_root: Path) -> None:
    candidate_path = repo_root / "training" / "autoresearch_v3" / "candidate.py"
    source = git(repo_root, ["show", "HEAD:training/autoresearch_v3/candidate.py"])
    candidate_path.write_text(source)


def commit_candidate(repo_root: Path, commit_message: str) -> None:
    git(repo_root, ["add", "training/autoresearch_v3/candidate.py"])
    git(repo_root, ["commit", "-m", commit_message])


def find_best_checkpoint(experiments_path: Path) -> Optional[str]:
    """Find the checkpoint path from the best 'keep' experiment."""
    if not experiments_path.exists():
        return None
    best_metric = None
    best_checkpoint = None
    for line in experiments_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            card = json.loads(line)
        except json.JSONDecodeError:
            continue
        if card.get("status") != "keep":
            continue
        checkpoint = card.get("checkpoint_path")
        if not checkpoint or not Path(checkpoint).exists():
            continue
        metrics = card.get("metrics")
        if not isinstance(metrics, dict):
            continue
        primary = metrics.get("primary_metric")
        if not isinstance(primary, (int, float)):
            continue
        if best_metric is None or primary < best_metric:
            best_metric = primary
            best_checkpoint = checkpoint
    if best_checkpoint:
        print(f"auto-resume: best keep checkpoint={best_checkpoint} (primary={best_metric:.6f})", flush=True)
    return best_checkpoint


def latest_result_status(results_path: Path, commit_id: str) -> Optional[str]:
    if not results_path.exists():
        return None
    for line in reversed(results_path.read_text().splitlines()[1:]):
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == commit_id:
            return parts[2]
    return None


def run_benchmark(
    repo_root: Path,
    description: str,
    *,
    resume: Optional[str],
    time_budget_s: Optional[float],
    eval_interval_s: Optional[float],
) -> None:
    cmd = [sys.executable, "training/autoresearch_v3/run_commit.py", "--description", description]
    if resume:
        cmd.extend(["--resume", str(Path(resume).expanduser().resolve())])
    if time_budget_s is not None:
        cmd.extend(["--time-budget-s", str(time_budget_s)])
    if eval_interval_s is not None:
        cmd.extend(["--eval-interval-s", str(eval_interval_s)])
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def maybe_reset_discard(repo_root: Path, results_path: Path, commit_id: str, enabled: bool) -> None:
    if not enabled:
        return
    status = latest_result_status(results_path, commit_id)
    if status == "discard":
        print(f"latest benchmark for {commit_id} was discard; resetting branch to parent", flush=True)
        git(repo_root, ["reset", "--hard", "HEAD^"])


def load_track(run_dir: Path) -> str:
    track_path = run_dir / "track.txt"
    if track_path.exists():
        return track_path.read_text().strip()
    return "unknown"


def controller_iteration(
    *,
    repo_root: Path,
    run_dir: Path,
    branch: str,
    track: str,
    args: argparse.Namespace,
) -> None:
    auto_dir = repo_root / "training" / "autoresearch_v3"
    candidate_path = auto_dir / "candidate.py"
    results_path = run_dir / "results.tsv"
    experiments_path = run_dir / "experiments.jsonl"
    controller_dir = run_dir / "controller"

    ensure_clean_worktree(repo_root)
    current_id = current_commit(repo_root)
    benchmarked_commits = load_benchmarked_commits(experiments_path)
    if current_id not in benchmarked_commits and head_touches_candidate(repo_root):
        print(f"current commit {current_id} has not been benchmarked yet; running it first", flush=True)
        resume = args.resume or find_best_checkpoint(experiments_path)
        run_benchmark(
            repo_root,
            args.initial_description,
            resume=resume,
            time_budget_s=args.time_budget_s,
            eval_interval_s=args.eval_interval_s,
        )
        maybe_reset_discard(repo_root, results_path, current_id, args.reset_on_discard)
        return

    results_tail = read_tsv_tail(results_path, MAX_RESULTS_CONTEXT)
    cards = load_experiment_cards(experiments_path, MAX_EXPERIMENTS_CONTEXT)
    family_cards = load_experiment_cards(experiments_path, MAX_FAMILY_HISTORY)
    latest_log = latest_log_tail(cards)
    family_policy = build_family_policy(family_cards)

    proposal: Optional[Dict[str, Any]] = None
    retry_feedback: Optional[str] = None
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        prompt = build_prompt(
            branch=branch,
            track=track,
            current_commit_id=current_id,
            results_tail=results_tail,
            cards=cards,
            latest_log=latest_log,
            run_dir=run_dir,
            family_policy=family_policy,
            retry_feedback=retry_feedback,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_response, stderr_text = run_llm_exec(
            repo_root,
            prompt,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            backend=args.backend,
        )
        write_controller_artifacts(
            controller_dir,
            f"{timestamp}.attempt{attempt}",
            prompt,
            raw_response,
            stderr_text,
        )

        proposal = raw_response
        for key in ["description", "commit_message", "rationale"]:
            if key not in proposal or not isinstance(proposal[key], str) or not proposal[key].strip():
                raise RuntimeError(f"Model response missing required string field: {key}")

        dirty_files = changed_files(repo_root)
        if dirty_files != ["training/autoresearch_v3/candidate.py"]:
            raise RuntimeError(
                "Controller changed unexpected files:\n" + "\n".join(dirty_files)
            )

        proposal_family = infer_experiment_family(
            proposal["description"],
            proposal["commit_message"],
            proposal["rationale"],
        )
        violation = family_policy_violation(family_policy, proposal_family)
        if not violation:
            break

        print(
            f"proposal rejected: {violation} "
            f"(family={FAMILY_DISPLAY_NAMES.get(proposal_family, proposal_family)})",
            flush=True,
        )
        restore_candidate_from_head(repo_root)
        retry_feedback = (
            f"The last proposal was classified as "
            f"{FAMILY_DISPLAY_NAMES.get(proposal_family, proposal_family)} and was rejected because {violation}. "
            "Propose a materially different family this time."
        )
    else:
        raise RuntimeError("Failed to obtain a family-diverse proposal after repeated attempts")

    assert proposal is not None

    validate_candidate_source(candidate_path.read_text())

    print(f"hypothesis: {proposal['rationale']}", flush=True)
    print(f"commit: {proposal['commit_message']}", flush=True)
    print(f"benchmark description: {proposal['description']}", flush=True)
    commit_candidate(repo_root, proposal["commit_message"])

    commit_id = current_commit(repo_root)
    resume = args.resume or find_best_checkpoint(experiments_path)
    run_benchmark(
        repo_root,
        proposal["description"],
        resume=resume,
        time_budget_s=args.time_budget_s,
        eval_interval_s=args.eval_interval_s,
    )
    maybe_reset_discard(repo_root, results_path, commit_id, args.reset_on_discard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven autoresearch controller for a v3 worktree")
    parser.add_argument("--worktree", required=True, help="Path to the autoresearch worktree root")
    parser.add_argument("--backend", default="claude", choices=["claude", "codex"], help="LLM backend to use")
    parser.add_argument("--model", default="sonnet", help="Model to use (e.g. sonnet, opus, gpt-5.4)")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high", "max"],
        help="LLM reasoning effort",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after this many controller iterations")
    parser.add_argument("--poll-interval-s", type=float, default=30.0, help="Polling interval while waiting for active runs")
    parser.add_argument("--initial-description", default="baseline", help="Description to use if the current candidate commit still needs its first benchmark")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path passed through to run_commit.py --resume")
    parser.add_argument("--time-budget-s", type=float, default=None, help="Optional override passed through to run_commit.py")
    parser.add_argument("--eval-interval-s", type=float, default=None, help="Optional override passed through to run_commit.py")
    parser.add_argument("--reset-on-discard", action="store_true", default=False, help="Hard-reset the worktree to HEAD^ after a discard result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.worktree).expanduser().resolve()
    auto_dir = repo_root / "training" / "autoresearch_v3"
    if not auto_dir.exists():
        raise RuntimeError(f"Not an autoresearch v3 worktree: {repo_root}")

    if args.backend == "codex":
        status = subprocess.run(["codex", "login", "status"], capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError(
                f"codex login status failed:\nstdout:\n{status.stdout}\nstderr:\n{status.stderr}"
            )
        login_text = f"{status.stdout}\n{status.stderr}"
        if "Logged in" not in login_text:
            raise RuntimeError("codex is not logged in")
    else:
        # claude CLI handles auth via API key or OAuth; verify it's reachable
        status = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError("claude CLI not found or not working")

    branch = current_branch(repo_root)
    if not branch.startswith("autoresearch-"):
        raise RuntimeError("Refusing to run outside an autoresearch-* branch")

    tag = infer_tag(branch)
    run_dir = auto_dir / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    track = load_track(run_dir)

    print(f"controller backend: {args.backend}", flush=True)
    print(f"controller model: {args.model}", flush=True)
    print(f"controller reasoning effort: {args.reasoning_effort}", flush=True)
    print(f"target worktree: {repo_root}", flush=True)
    print(f"branch: {branch}", flush=True)
    print(f"track: {track}", flush=True)

    iteration = 0
    while args.max_iterations is None or iteration < args.max_iterations:
        wait_for_active_benchmark(repo_root, args.poll_interval_s)
        try:
            controller_iteration(
                repo_root=repo_root,
                run_dir=run_dir,
                branch=branch,
                track=track,
                args=args,
            )
            iteration += 1
            print(f"completed controller iteration {iteration}", flush=True)
        except RuntimeError as exc:
            error_msg = str(exc)
            # Detect rate limits / usage caps and retry after a delay
            if any(phrase in error_msg.lower() for phrase in [
                "out of extra usage", "rate limit", "overloaded", "capacity",
                "too many requests", "429", "quota", "resets",
            ]):
                retry_minutes = 10
                print(
                    f"rate limit / usage cap hit: {error_msg[:200]}",
                    flush=True,
                )
                print(f"sleeping {retry_minutes} minutes before retrying...", flush=True)
                # Restore candidate.py if it was dirtied
                try:
                    restore_candidate_from_head(repo_root)
                except Exception:
                    pass
                time.sleep(retry_minutes * 60)
                continue
            raise


if __name__ == "__main__":
    main()
