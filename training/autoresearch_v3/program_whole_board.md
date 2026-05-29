# Whole-Board Classifier Autoresearch

Autonomous research loop for the whole-board chess piece classifier.

## Rules

- Edit only `training/autoresearch_v3/candidate.py`.
- Do not edit `fixed_harness.py`, validation logic, result schema, or promotion rules.
- Do not change validation data or metrics.
- Do not retrain or modify the corner detector — it is frozen. Training uses GT corners from `annotations.json`.
- Prefer simple changes when gains are small.

## Goal

- Minimize combined validation `mean_dist`. In this track `mean_dist` is the **per-cell error rate** (fraction of the 64 cells per board that are predicted incorrectly), averaged across the combined val set. Lower is better. The schema key is reused from the corner track so the controller, `run_commit.py`, and `results.tsv` keep working unchanged — the semantics differ, but the direction (lower = better) is the same.
- Secondary signal: `per_corner[1]` = **per-board error rate** (fraction of boards with any wrong cell). Also lower-is-better.
- Do not materially regress `chess_dataset_recovered:val` (top-down, clean domain) beyond ~+0.005 cell error.
- Do not materially regress `chessred2k:val` beyond ~+0.01 cell error.
- Do not materially worsen `max_dist` (worst-board cell error) or `p95_dist`.

## Task

The model takes a 256×256 RGB **warped** board (warped upstream by `BoardDataset` using GT corners) and outputs a `[B, 13, 8, 8]` classification tensor — 13 classes per cell. The 13 classes and their IDs must match `src/ml/inference.ts:CLASS_TO_PIECE` so the existing FEN builder and on-device inference path work unchanged.

Baseline architecture: ImageNet-pretrained ResNet-18 + a `Conv2d(512, 13, 1)` head on the natural 8×8 feature map from `layer4` at input 256. Baseline loss: plain per-cell cross-entropy.

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

## Heuristics

- **Prioritize architectural changes over hyperparameter tuning.** The first 10 iterations showed that optimizer/loss/LR tweaks plateau around 3% combined cell error with only marginal differences. The next gains will come from:
  1. **Larger backbone** (ResNet-34, ResNet-50, EfficientNet-B0) — more capacity to distinguish similar pieces.
  2. **MLP head** — replace the 1×1 conv with `Conv2d(512, 256, 1) → ReLU → Conv2d(256, 13, 1)` so each cell has more classification capacity.
  3. **Higher input resolution** (320 or 384 instead of 256) with `adaptive_avg_pool2d(x, (8, 8))` before the head — more pixels per cell helps on chessred2k where pieces are small.
  4. **Top margin on the warp** — the back rank gets clipped when pieces extend above the board boundary after warping. Try warping to a slightly larger canvas and cropping, or padding the destination quad with a top margin.
  5. **Attention between neighboring cells** — a small decoder block before the classification head so the model can resolve ambiguity when a piece visually overlaps into a neighbor's cell.
- Do not spend more iterations on optimizer/LR/loss tuning unless an architectural change specifically requires it.
- Use split metrics, not just combined mean. The three val splits have very different difficulty profiles: `chess_dataset_recovered` is near-top-down and easy, `chessred2k` is varied angle, `synthetic` is the largest but with random camera poses.
- Empty squares dominate ~50% of cells; watch for class collapse toward "empty" and use per-class recall to diagnose.
- Augmentation should focus on color/exposure and single-cell random erasing. No geometric augmentation — the board is already warped.
- Use longer budgets for architecture changes than for small tuning tweaks.
- Don't break the output shape contract: the model must return `{"logits": [B, 13, 8, 8]}` and `decode_coords` must return `[B, 64]` int64 row-major predictions.
- `make_targets` receives a `pieces` dict (not `corners` — the `corners` arg is None here because the board is already warped upstream).

## Known failure modes (from baseline analysis)

- ~5% of boards across all datasets are catastrophically wrong (80-97% cells wrong). These boards predict mostly `n`/`K` for every cell, suggesting the model completely fails on certain lighting/angle conditions. Architectural improvements are more likely to fix this than loss tuning.
- All 5 user images (real phone photos, angled) score 78-91% cell error. The user split is eval-only and represents the real deployment scenario.
- The worst chessred2k boards are concentrated in group G006 — a specific capture setup/lighting condition.

## Context

- Corner detection is already solved (combined mean normalized distance ≈ 0.25% from track `unet_dual_head`, commit `7e15e8e`). Do not touch it.
- The existing per-square classifier (`training/models/best_square_classifier.pt`) is the comparison baseline. We expect the whole-board model to match or beat its per-cell accuracy and to enable ≥50% fully-correct-board rate — a bar per-square struggles with on steep angles because tall pieces occupy multiple squares.
- Dataset labeling convention: the `corners` dict in `annotations.json` uses **board-space** labels (`top_left` = a8 regardless of image rotation). `BoardDataset` warps using these labels directly without re-sorting, so row 0 = rank 8 and col 0 = file a always.
- For synthetic images, piece labels come from FEN in `training/data/data.json` (no `annotations.pieces` entries).
