# Whole-Board Classifier: Experiment Roadmap

Seeded experiments for the autoresearch LLM controller to consider first. Each entry is a hypothesis, not a prescription — the controller is free to pick a different direction once the baseline results land.

## 1. Baseline

ResNet-18 (ImageNet) + `Conv2d(512, 13, 1)` on the natural `[B, 512, 8, 8]` feature map. Per-cell cross-entropy, AdamW, cosine LR. This is the template candidate; first commit runs it as-is to get a number on the board.

## 2. Augmentation sweep

Color jitter strength (alpha/beta ranges in `augment_image`), HSV hue/saturation jitter, gaussian blur probability, and single-cell random erasing probability. No geometric augmentation — the board is already warped.

## 3. Class-balanced loss

Empty squares are ~50% of cells. Plain CE biases toward predicting empty. Try:
- CE with class weights inversely proportional to training-set frequency.
- Focal loss (γ ∈ {1.0, 2.0}).
- Label smoothing (ε ∈ {0.05, 0.1}).

## 4. Larger backbone

If ResNet-18 plateaus, try ResNet-34 or EfficientNet-B0. Watch inference-time budget: the whole-board win is partly about going from 64 forward passes to 1, so a big backbone that's slower than 64× ResNet-18 would erase the gain.

## 5. Head capacity

Replace the 1×1 conv head with:
- 2-layer MLP on each cell (`Conv2d(512, 256, 1) → ReLU → Conv2d(256, 13, 1)`).
- A small decoder: one conv block that attends to neighboring cells before classifying, so a queen overflowing into a neighbor's square can still be resolved correctly.

## 6. Input margin

If back-rank accuracy lags (pieces whose tops extend above the board plane after warping get clipped), rewarp with a small top-margin and input size 320 or 384 instead of 256. The baseline uses no margin because the global receptive field should see piece tops anyway, but this is worth checking experimentally.

## 7. Resolution scaling

Input size 320 or 384 with `adaptive_avg_pool2d(x, (8, 8))` before the head. Gives the backbone more pixels per cell — useful for the `chessred2k` split where pieces are small relative to the full board.

## Loop diversity

From the corner-detection run: the controller tends to over-focus on one experiment family once it finds a small improvement. Prefer a spread across these first few directions before iterating within one family. The promotion policy in `run_commit.py` is the same as the corner track; trust it.
