# Whole-Board Classifier Autoresearch

Autonomous research loop for the whole-board chess piece classifier.

## Rules

- Edit only `training/autoresearch_v3/candidate.py`.
- Do not edit `fixed_harness.py`, validation logic, result schema, or promotion rules.
- Do not change validation data or metrics.
- Do not retrain or modify the corner detector — it is frozen. Training uses GT corners from `annotations.json`.
- **Out of scope**: the production pipeline now uses a two-stage refinement (whole-board + piece-only ResNet-18 on per-cell crops; see `/home/johann/ImagetoFEN/training/two_stage/`). The piece classifier is *not* part of this autoresearch track. The loop optimizes only the whole-board candidate.
- Prefer simple changes when gains are small.

## Goal

- Minimize combined validation `mean_dist`. In this track `mean_dist` is the **per-cell error rate** (fraction of the 64 cells per board that are predicted incorrectly), averaged across the combined val set. Lower is better.
- Secondary signal: `per_corner[1]` = **per-board error rate** (fraction of boards with any wrong cell).
- Do not materially regress `chess_dataset_recovered:val` (top-down, clean domain) beyond ~+0.005 cell error.
- Do not materially regress `chessred2k:val` beyond ~+0.01 cell error.
- Do not materially worsen `max_dist` (worst-board cell error) or `p95_dist`.

## Task

The model takes a 256×256 RGB **warped** board (warped upstream by `BoardDataset` using GT corners) and outputs a `[B, 13, 8, 8]` classification tensor — 13 classes per cell. The 13 classes and their IDs must match `src/ml/inference.ts:CLASS_TO_PIECE` so the existing FEN builder and on-device inference path work unchanged.

## Current best

ResNet-34 backbone + 2-layer cell-attention TransformerEncoder + `Conv2d(512, 256, 1) → GELU → Conv2d(256, 13, 1)` head. Combined cell_err **0.029** (commit `0399b00`, 2026-04-25). Resume from this checkpoint by default.

## Loop

1. Read the latest `results.tsv` and `experiments.jsonl`.
2. Form one concrete hypothesis.
3. Edit only `candidate.py`.
4. Commit the change.
5. Run:

```bash
python3 training/autoresearch_v3/run_commit.py --description "short hypothesis"
```

6. Keep advancing on `keep`.
7. Treat `promising` as a near-miss worth iterating on, especially for architecture changes.
8. Reset on `discard` if needed.
9. Continue until interrupted.

## What's already been tried

These are recorded in `experiments.jsonl` and `results.tsv` but called out here so the controller doesn't re-propose them:

**Already proven (in the current candidate, do not undo)**
- ResNet-18 → ResNet-34 backbone: **+0.25pp** (commit `0399b00`).
- 1×1 conv head → 2-layer MLP head: **kept** (commit `974644f`).
- 2-layer TransformerEncoder cell-attention block on the 8×8 feature map: **kept** (commit `974644f`, descended from `68e3ba0`).
- ExponentialLR + amsgrad + weight_decay 0.03: **kept** (commit `b9c740d`).
- batch_size 24 (vs 16): **kept** (commit `1420a40`).

**Already discarded — do not retry without a substantive change**
- Label smoothing ε ∈ {0.05, 0.1}: discarded twice (commits `66f23ed`, `9c2e03c`).
- Focal loss γ = 2.0: discarded (commit `43812fd` reverted at `6811cfb`; also `0316565`).
- ResNet-50 backbone: hard regression, discarded (`92f0513`).
- Higher input resolution 320 / 384 px (with adaptive pool): hard regression, discarded twice (`65db0b4`, `d0712fb`). 384 was also tried by manual training resume on 2026-04-27 outside the loop (overfit, −1.4pp).

**Untried (prioritize)**
- **Top margin on the warp** (warp to a slightly larger destination quad and crop, so piece tops on the back rank aren't clipped). Plausible because back-rank piece overflow was an early hypothesis but no candidate has actually tried it.
- EfficientNet-B0 backbone (similar parameter count to ResNet-34 with different inductive biases).
- Heavier color/exposure augmentation (the current `augment_image` is light).

## Heuristics

- **Architectural changes > hyperparameter tuning.** Hyperparam sweeps already plateaued (see discarded list). Look for changes that affect what features the backbone can learn.
- Use split metrics, not just combined mean. The three val splits have very different difficulty: `chess_dataset_recovered` is near-top-down and easy (99.6%+), `chessred2k` is varied angle (95.6%, the bottleneck), `synthetic` is essentially solved (99.9%).
- Empty squares are ~74% of cells in the combined val. Plain CE handles this fine — see discarded class-balanced losses above.
- Augmentation should focus on color/exposure and single-cell random erasing. **No geometric augmentation** — the board is already warped, and rotations/flips are out-of-distribution for this model. (TTA was tested and regressed; would require retraining with hflip aug.)
- Use longer budgets for architecture changes than for small tuning tweaks.
- Don't break the output shape contract: the model must return `{"logits": [B, 13, 8, 8]}` and `decode_coords` must return `[B, 64]` int64 row-major predictions.
- `make_targets` receives a `pieces` dict (not `corners` — the `corners` arg is None here because the board is already warped upstream).

## Known failure modes (current best, 2026-04-27 analysis)

- **Bishop & knight shape confusion under perspective** is the dominant residual on chessred2k. Per-class recall: B 0.58, b 0.55, N 0.63, n 0.75 vs P/p at 0.94/0.98 and empty at 0.997. Top off-diagonals: B→P 72, b→r 70, b→p 52, N→R 41.
- Errors are higher in middle ranks (rows 4–6) than back ranks. The "tall pieces overflow into back rank" hypothesis from the original program is contradicted by the data — it's a uniform shape-recognition problem, not a position-overflow problem.
- All 5 user images (real phone photos, angled) score ~46% cell error on the current best. Out-of-distribution for what's in train.

## Context

- Corner detection is solved (combined mean normalized distance ≈ 0.25% from track `unet_dual_head`, commit `7e15e8e`). Do not touch it.
- Dataset labeling convention: the `corners` dict in `annotations.json` uses **board-space** labels (`top_left` = a8 regardless of image rotation). `BoardDataset` warps using these labels directly without re-sorting, so row 0 = rank 8 and col 0 = file a always.
- For synthetic images, piece labels come from FEN in `training/data/data.json` (no `annotations.pieces` entries).
- `chess_dataset_recovered:val` was cleaned 2026-04-27 (3 corner-rotation-bug images dropped: ids 12875, 12844, 12986). The split now has 97 samples instead of 100. Train splits are unchanged.
