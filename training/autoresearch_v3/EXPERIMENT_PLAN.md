# Multi-Track Plan

Run two tracks in parallel and compare them over time under the same harness:
- `resnet_coords`: current coordinate-regression family
- `unet_dual_head`: segmentation + heatmaps + decoded coordinates

Each track should live in its own worktree and branch. Judge long-term progress by each track's `results.tsv`, not by a single early run.

## U-Net Dual-Head Plan

Goal: test whether a spatial model can beat the current ResNet coordinate baseline on the recovered green-board split without blowing up runtime or max error.

## Proposed model

- Input: keep `gray_edges` first
- Encoder: ResNet-18 style encoder or equivalent residual downsampling path
- Decoder: U-Net style upsampling with skip connections
- Outputs:
  - `board_mask_logits`: 1 channel
  - `corner_heatmaps`: 4 channels in TL/TR/BR/BL order
  - `corner_coords`: 8 normalized floats decoded from soft-argmax heatmaps or decoder features

## Initial targets

- Train board mask from the quadrilateral filled interior of the labeled corners
- Train four Gaussian corner heatmaps
- Evaluate using the final coordinate output under the existing corner-distance metric

## Initial losses

- `mask_loss = 0.5 * weighted_bce(mask) + 0.5 * dice_loss(mask)`
- `heatmap_loss = weighted_mse(sigmoid(corner_heatmaps), target_heatmaps)`
- `coord_loss = SmoothL1(pred_coords, target_coords)`
- `total_loss = 1.0 * heatmap_loss + 0.5 * mask_loss + 0.2 * coord_loss`

## Initial geometry choices

- Decoder output resolution: `96 x 96`
- Heatmap sigma: start near `2.5` to `3.5` pixels at heatmap scale
- Decode coordinates with soft-argmax first
- Add a direct coordinate head only if soft-argmax alone is unstable

## Budget policy

- Small tuning changes: `20-30 min`
- New architecture baselines: `45-60 min`
- If a new architecture lands close to best and split guards are acceptable, mark it `promising` and keep iterating instead of discarding immediately

## First 5 experiments

1. Baseline dual-head U-Net
   - `gray_edges`
   - segmentation + 4 corner heatmaps + soft-argmax coordinates
   - `45 min`

2. Heatmap sigma sweep
   - one smaller sigma and one larger sigma
   - same architecture
   - `30-45 min`

3. Loss rebalance
   - raise heatmap weight or coord weight slightly
   - keep mask branch unchanged
   - `30 min`

4. Video-oriented augmentation
   - add motion blur and stronger shadow/brightness augmentation
   - `30 min`

5. Decoder refinement
   - stronger skip fusion or a slightly wider decoder block
   - keep encoder fixed
   - `45 min`

## If the first branch works

Next directions:
- UNet++ style skip fusion
- attention gates on skip connections
- mask-guided feature gating
- RGB vs `gray_edges`
- hard negatives that look board-like but are not boards

## Promotion criteria

- combined mean improves enough to matter
- recovered split does not materially regress
- max error does not spike badly
- complexity stays justified by the gain
