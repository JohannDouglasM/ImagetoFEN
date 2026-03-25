# Chess Corner Autoresearch v3

Autonomous research loop for the chess corner model.

## Rules

- Edit only `training/autoresearch_v3/candidate.py`.
- Do not edit `fixed_harness.py`, validation logic, result schema, or promotion rules.
- Do not change validation data or metrics.
- Prefer simple changes when gains are small.

## Goal

- Minimize combined validation mean corner distance.
- Do not materially regress `chess_dataset_recovered:val`.
- Do not materially worsen worst-case or p95 error.

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
7. Treat `promising` as a near-miss worth iterating on, especially for new architectures.
8. Reset on `discard` if needed.
9. Continue until interrupted.

## Heuristics

- Start with small, coherent changes.
- Use split metrics, not only combined mean.
- Use longer budgets for new architectures than for small tuning changes.
- If recovered green-board keeps regressing, pivot.
- Discard ugly complexity for tiny gains.
- Keep simplifications that match or improve metrics.

## Context

- `gray_edges` is the current strongest baseline family.
- The recovered green-board split is the fragile domain.
- Robustness-oriented preprocessing, augmentation, modest architecture changes, and short-budget optimizer tuning are the main search directions.
- For larger shifts like U-Net heatmap models, do not discard too early just because the first run is only close to the best baseline.
