# Whole-Board Classifier: Experiment Roadmap

> **Status note (2026-04-28)**: this roadmap was the seed for the loop's first iterations. Outcomes from the 32 commits to date are inlined per section so the controller doesn't re-propose discarded experiments. The current best is commit `0399b00` (combined cell_err 0.029, ResNet-34 + cell-attention transformer + 2-layer MLP head). For the canonical "what to try next" list, see `program_whole_board.md`.

## 1. Baseline ✅ (superseded)

Original baseline: ResNet-18 + `Conv2d(512, 13, 1)`. **Now obsolete** — current best uses ResNet-34 + cell-attention + 2-layer MLP head (commit `0399b00`).

## 2. Augmentation sweep — partial

Color jitter, HSV, blur, single-cell erasing are all in `augment_image`. No explicit standalone sweep was logged as a keep; behavior is folded into the current candidate. **Heavier augmentation** (wider alpha/beta ranges, larger erasing patches) is still untried.

## 3. Class-balanced loss ❌ (all variants discarded)

- Plain CE: kept as the loss in current best.
- **Label smoothing ε ∈ {0.05, 0.1}**: ❌ discarded (`66f23ed`, `9c2e03c`).
- **Focal loss γ ∈ {1.0, 2.0}**: ❌ discarded (`43812fd` reverted at `6811cfb`; `0316565`).
- **Inverse-frequency class weights**: not tried, but the dominant residual is *piece-vs-piece* confusion (bishop/knight shape ambiguity), not piece-vs-empty class imbalance — class weighting would address the wrong problem.

## 4. Larger backbone — partial

- **ResNet-34**: ✅ kept as current best (`0399b00`, +0.25pp over ResNet-18).
- **ResNet-50**: ❌ hard regression, discarded (`92f0513`). Don't retry without architecture changes (e.g. proper channel reducer for the 2048-dim layer4 output).
- **EfficientNet-B0**: untried. Plausible candidate.

## 5. Head capacity ✅ (in current best)

- **2-layer MLP head** (`Conv2d(512, 256, 1) → GELU → Conv2d(256, 13, 1)`): ✅ kept (`974644f`).
- **2-layer TransformerEncoder cell-attention** before the MLP head: ✅ kept (`974644f`, descended from `68e3ba0`).
- **Don't split this further**: cell-attn + MLP head is the winning recipe; adding a third stage on top hasn't been tested but expectations are low.

## 6. Input margin — UNTRIED, prioritize

Re-warp to a slightly larger destination quad (e.g. add 5–10% margin on top), crop or pad to 256², so back-rank piece tops aren't clipped. This is the most-promising untried item from the original roadmap. The 2026-04-27 failure analysis found errors are not concentrated in the back rank, but pieces in middle ranks may still be partially clipped near the warp boundary on extreme angles.

## 7. Resolution scaling ❌ (discarded twice)

- **320 px**: ❌ hard regression (`65db0b4`).
- **384 px** (within autoresearch loop): ❌ hard regression (`d0712fb`).
- **384 px** (manual train, resume from 256 best, 2026-04-27): ❌ overfit, −1.4pp combined.
- Conclusion: don't retry without changing the input pipeline (e.g. progressive resize, train at higher res from ImageNet init with longer schedule).

## Loop diversity

- The controller tends to over-focus on a family once it finds a small improvement. Spread across new directions when you can. From here:
  - Top margin (§6) is the top untried.
  - Heavier augmentation (§2) is cheap to test.
  - EfficientNet-B0 (§4) is the only remaining promising backbone variant.
- Promotion policy in `run_commit.py` is unchanged from the corner track — trust it.
