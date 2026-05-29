# ImagetoFEN (pngtopgn)

React Native / Expo app that turns a photo of a chess position into FEN / PGN. Designed to run fully on-device via ONNX Runtime.

> **Snapshot as of 2026-05-29.** The training pipeline reaches **99.09% combined val cell accuracy**. The mobile app (`src/ml/inference.ts`) still ships the older per-square ONNX pipeline; the two-stage production model has not been exported / wired into the app yet. See [Status & open work](#status--open-work).

## Pipeline (training)

```
photo
  → corner detector (U-Net dual-head, frozen)
  → perspective warp to 256×256
  → whole-board classifier (ResNet-34 + cell-attention transformer)
  → piece-only refinement on non-empty cells (gated)
  → 8×8 grid of class predictions
  → FEN
  → PGN / Lichess URL
```

- **Combined val cell accuracy: 99.09%** (chessred2k 98.54%, chess_dataset_recovered 99.87%, synthetic 99.68%)
- Whole-board model standalone: 97.10% (2.90% cell error)
- Production combiner mode: `gated` — fall back to the piece classifier only when the whole-board model's piece-class confidence < 0.999

## Repo layout

```
app/                          Expo Router screens (camera, result)
src/
  ml/inference.ts             on-device ONNX inference (CURRENT PROD — old pipeline)
  chess/{fenBuilder,pgnExporter}.ts
  components/{ChessBoard,PieceEditor}.tsx
training/                     Python training & eval
  autoresearch_v3/            autonomous experiment loop (Claude/Codex driven)
  two_stage/                  piece-only refinement model (production combiner)
  wb_384/                     384px whole-board retrain (DEAD END, kept for reference)
  detect_board_v5.py          CV fallback corner detector (reference only)
  train_corners_*.py          older corner-detector training paths
  visualize_worst_predictions.py / analyze_failures.py  failure analysis
  checkpoints/                checked-in model weights (see below)
assets/models/                ONNX exports (gitignored — would be built from training/checkpoints/)
```

[`CLAUDE.md`](CLAUDE.md) is the authoritative state-of-pipeline doc — read it first if you're picking this up cold.

## Checked-in model weights

These live under `training/checkpoints/` so a fresh clone can reproduce evaluation without re-training:

| File | Size | What it is |
|---|---|---|
| `corner_detector_7e15e8e.pt` | 55 MB | U-Net dual-head corner detector, autoresearch winner from track `20260331-unet_dual_head` commit `7e15e8e`. Frozen. |
| `corner_detector_7e15e8e.candidate.py` | 14 KB | Immutable source snapshot for the corner detector above. Required to load the checkpoint. |
| `whole_board_0399b00.pt` | 99 MB | Whole-board classifier (ResNet-34 + cell-attention transformer + 2-layer MLP head), autoresearch winner from track `20260423-whole_board_classifier` commit `0399b00`. Frozen at 97.10% combined. |
| `whole_board_0399b00.candidate.py` | 12 KB | Immutable source snapshot for the whole-board model above. Required to load. |
| `corner_detector_results.tsv` | 22 KB | Full autoresearch experiment log for corner-detector track. |
| `whole_board_results.tsv` | 12 KB | Full autoresearch experiment log for whole-board track. |

The piece-only refinement model lives at its training run path:

| File | Size | What it is |
|---|---|---|
| `training/two_stage/run2/best.pt` | 43 MB | **Production piece classifier (v2)**, val piece-acc 96.84%. Used by `eval_two_stage.py` in `gated` mode. |
| `training/two_stage/run3/best.pt` | 43 MB | Experimental v3 with color + type auxiliary heads. Targets piece-classifier residual confusions. |
| `training/two_stage/run{2,3}/confusion.npy` | small | Confusion matrices for failure analysis. |

`annotations.json` (24 MB) is also checked in — primary labels for all splits.

## What's NOT in the repo

| Item | Local path on dev machine | Size | How to rebuild |
|---|---|---|---|
| ChessReD2K dataset | `chessred2k/` | 4.3 GB | <https://chessred.github.io/> |
| chess-dataset (samryan18, vinyl board) | `chess-dataset/` | 4.4 GB | <https://github.com/samryan18/chess-dataset> |
| Synthetic renders | `training/data/` | varies | `training/prepare_all_data.py` |
| Warp cache for piece classifier | `training/two_stage/warp_cache/` | ~1.7 GB | `.venv/bin/python training/two_stage/build_warp_cache.py` (needs `annotations.json` corners) |
| 384 px experiment outputs | `training/wb_384/run1/` | ~99 MB | Re-run from `training/wb_384/candidate.py` (dead-end, low priority) |
| Per-square / two-stage / failure-viz dumps | `training/{recovered_audit,crop_test_output,worst_predictions,viz_*.png}` | varies | Re-run the corresponding scripts |
| User photos (5 imgs) | inside `annotations.json` group `user:train` (metadata only — actual JPEGs are local) | small | Re-take, re-annotate via `training/annotate_corners.py` |

The truly hard loss if the dev machine is wiped is the **raw dataset images** (8.7 GB combined). Annotations and checkpoints are checked in here. Datasets are public — re-download from the links above.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Last published reference state (older corner-detector era). |
| `whole-board-classifier` | **Primary development branch.** Whole-board + piece-refinement pipeline, autoresearch v3, current production accuracy. |
| `autoresearch-20260331-unet_dual_head` | Worktree branch — autoresearch experiment commits for the corner detector. Each commit is one experiment. Final winner: `7e15e8e`. |
| `autoresearch-20260423-whole_board_classifier` | Worktree branch — autoresearch experiment commits for the whole-board classifier. Each commit is one experiment. Final winner for primary metric: `0399b00`. Later experiments (`62ede1b` label-smoothing line) hit 0.02861 standalone but did not beat the combined-metric selection. |

## Autoresearch v3

The loop in `training/autoresearch_v3/` is the engine behind both the corner and whole-board models. Architecture:

- `llm_controller.py` drives candidate edits via the Claude CLI (sonnet, high reasoning) against a program file (`program_whole_board.md`) and an experiment plan (`experiment_plan_whole_board.md`).
- Each iteration: LLM edits `candidate.py` → commit → `run_commit.py` benchmarks via `fixed_harness.py` → controller decides keep / discard based on combined cell error → fine-tune from previous `best.pt` via `--resume`.
- Driven by `start_loop.py`. **Must be launched with the project venv** (cv2 is imported at module level in `fixed_harness.py`):

```bash
/home/johann/ImagetoFEN/.venv/bin/python training/autoresearch_v3/start_loop.py \
  --track whole_board_classifier \
  --worktree /home/johann/autoresearch/20260423-whole_board_classifier \
  --backend claude --model sonnet --reasoning-effort high \
  --time-budget-s 21600 --eval-interval-s 1800 --resume
```

- Results land in `training/autoresearch_v3/runs/<track>/results.tsv` (gitignored on worktree branches; snapshots committed to `training/checkpoints/`).
- Per-iteration artifacts (immutable `candidate.py`, `best.pt`, `metrics.json`, `run.log`) land in `runs/<track>/artifacts/<timestamp>_<sha>/`.

## Useful commands

```bash
# Reproduce production metrics (combined ~99.09%)
.venv/bin/python training/two_stage/eval_two_stage.py

# Top-3 worst per val split, with overlay
.venv/bin/python training/visualize_worst_predictions.py

# Per-class recall / confusion / per-row error
.venv/bin/python training/analyze_failures.py

# Retrain piece classifier (~1 hr on a GTX 1070)
.venv/bin/python training/two_stage/train_piece_v3.py --time-budget-s 3600
```

## Status & open work

### Out-of-distribution failure on user photos
The 5 manually annotated user images (`user:train` split, held out from training) score **cell_err 0.36–0.50** vs. ~0.045 on chessred2k. The model is essentially broken on them. Worst is the starting position at 0.50. No autoresearch iteration to date has moved this needle. The most likely lever — adding more user-like training data and/or stronger domain-shift augmentation — has not been tried.

### Residual error (in-distribution)
Dominant remaining errors come from the **piece classifier**, not the whole-board model. From v2's confusion matrix: white K↔Q (118 + 24), black B↔P/N (50+ together), N↔R confusions. v3 (color + type auxiliary heads) targets these; comparison to v2 not yet decisively measured.

### Whole-board model is plateaued
14 autoresearch keep-iterations moved combined cell_err from 3.13% → 2.90%. The label-smoothing experiment (`62ede1b`) hit 0.02861 but did not beat `0399b00` (0.0290) in the held-out combined metric used for selection. **Don't expect large gains from further `candidate.py` edits.**

### Deferred / out of scope
- **384 px whole-board retrain.** Tried with resume on 2026-04-27, regressed −1.4 pp (overfit). A from-scratch ImageNet-init retry is plausible but low expected payoff vs. the piece-classifier lever.
- **Production ONNX export of the two-stage pipeline.** `src/ml/inference.ts` still ships the old per-square pipeline. Two-stage productionization = two ONNX models + on-device orchestration. Not started.
- **Piece-classifier autoresearch track.** Would need a parallel harness. Defer until v3 lands and shows a real ceiling.
- **TTA on the whole-board model.** Training aug has no rotations or flips; naïve TTA regressed. Needs hflip aug in training first.

## Key technical decisions

1. **Two-stage pipeline** (detect corners → classify squares) — each stage independently improvable.
2. **3-channel hybrid input** for the older corner detector — grayscale + Canny edges + square-center heatmap.
3. **On-device only** — no cloud APIs. ONNX Runtime + pure-JS jpeg-js + JS homography.
4. **Chesscog-style square cropping** for the piece classifier — variable height/width margins, horizontal flip for left columns, bottom-aligned padding.

## License

No license file. Treat as private / unlicensed unless one is added.
