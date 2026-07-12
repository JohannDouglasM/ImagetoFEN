# CLAUDE.md

## Current state (as of 2026-07-12)

Production pipeline: **corner detection → warp → whole-board classifier → piece-only refinement → FEN build**.

Combined val cell accuracy: **99.09%** (up from autoresearch best of 97.10%). Per split: chessred2k 98.54%, chess_dataset_recovered 99.87%, synthetic 99.68%. **Caveat discovered 2026-07-12: these numbers use GT-corner warps.** See "Mobile app" below for detected-corner numbers.

## Mobile app (two-stage on-device, added 2026-07-12)

`src/ml/inference.ts` now runs corner detector + whole-board classifier on-device via ONNX Runtime (the old per-square pipeline is gone). Piece-only refinement is NOT on-device yet.

- **Export**: `.venv/bin/python training/export_onnx_two_stage.py` → `assets/models/{corner_unet,whole_board}.onnx` (58 + 103 MB, fp32, gitignored — regenerate from `training/checkpoints/`). Parity vs PyTorch ~1e-5.
- **Key discovery — corner order**: the corner detector localizes corners superbly (~0.15% of diagonal) but does **not** reliably predict board-space order (which corner is a8) — wrong on ~60% of chessred2k val angles. Training eval never caught this because warps used GT corners. The app therefore sorts corners visually (convex, angular sort), warps once, runs the whole-board model on **all 4 rotations** of the warp, and keeps the orientation with the highest mean per-cell softmax confidence. A "rotate board" button in the result screen is the manual backstop (picker is right on ~13/16 val images). A chess-plausibility score (pawns on back ranks, king counts) was tested and performed WORSE than raw confidence — don't retry naively.
- **Honest end-to-end numbers** (detected corners, confidence-picked rotation, 1024px working image): chessred2k val sample ~91.3% cell acc (oracle rotation 95.0%); the 5 user photos ~55.6%. Reproduce with `node scripts/validate_pipeline.ts` (needs `valsample/` images extracted from chessred2k.zip) and `node scripts/test_user_photos.ts`.
- **Shared TS core**: all preprocessing (bilinear resize, cv2-faithful Gaussian blur + Canny 80/200, homography warp, rot90) lives in `src/ml/pipelineCore.ts`, imported by both the app and the Node harnesses. Change it only with the harness re-run.
- **Correction flywheel**: verified/corrected positions are saved via `src/ml/corrections.ts` to `<documents>/corrections/` as photo (FEN in filename, `:` for `/`) + JSON sidecar (corners, dims). This is the future user-domain training data.
- **Build**: managed workflow + expo-dev-client (must be `~55.x`, the SDK-matched line); `android/` is generated (prebuild). `npx expo run:android` / `run:ios`. No Xcode on the dev Mac; Android SDK + one AVD exist. Use Android Studio's bundled JDK: `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`.
- **Build gotchas (all fixed in-repo, 2026-07-12)**:
  1. `patches/` (applied via patch-package postinstall): foojay-resolver bumped to 1.0.0 in `@react-native/gradle-plugin` (Gradle 9 removed `JvmVendorSpec.IBM_SEMERU`), and a Gradle-9-incompatible `VersionNumber` check removed from `onnxruntime-react-native`.
  2. `react-native.config.js` at repo root: onnxruntime-react-native carries a leftover Expo-module config, so SDK 55 autolinking silently skips it → `NativeModules.Onnxruntime` null at runtime. The explicit config forces RN linking. Verify with `npx expo-modules-autolinking react-native-config --platform android --json`.
  3. `react-dom` must be installed (Expo's dev LogBox imports it); npm needs `--legacy-peer-deps` for installs in this project.
- **Verified on-device 2026-07-12** (Android emulator, debug build): full flow photo → detected position → FEN → correction UI works; ~2 min inference on emulator (JS JPEG decode + 4 whole-board passes dominate). Known follow-ups: quantize models (58+103 MB fp32), speed up (batch the 4 rotations, native decode), piece-refinement stage on-device.

### Components

- **Corner detector** (frozen). U-Net dual-head, ~0.25% mean normalized corner distance. Checkpoint: `/home/johann/autoresearch/20260331-unet_dual_head/training/autoresearch_v3/runs/20260331-unet_dual_head/artifacts/20260406T122113Z_7e15e8e/best.pt`. **Do not touch.**

- **Whole-board classifier** (frozen at autoresearch best). ResNet-34 + cell-attention transformer + 2-layer MLP head, 256×256 RGB warped board → `[B, 13, 8, 8]`. Combined cell_err 2.90% standalone. Checkpoint: `/home/johann/autoresearch/20260423-whole_board_classifier/training/autoresearch_v3/runs/20260423-whole_board_classifier/artifacts/20260425T091625Z_0399b00/best.pt` (commit `0399b00`). Source: same artifact dir, `candidate.py`.

- **Piece-only refinement** (production). ResNet-18 trained on 96×96 per-cell crops (with chesscog-style upward height extension). Checkpoint: `/home/johann/ImagetoFEN/training/two_stage/run2/best.pt` (v2, val piece-acc 96.84%). v3 training run with auxiliary color/type heads is in `training/two_stage/run3/`. Source: `training/two_stage/{piece_model.py, train_piece_v2.py, piece_dataset.py}`.

- **Eval & inference combiner**. `training/two_stage/eval_two_stage.py`. Three modes:
  - `unconditional`: replace whole-board prediction on every non-empty cell.
  - `gated`: replace only when whole-board's piece-class confidence < 0.999. **This is the production mode** (combined 99.09%).
  - `ensemble`: average whole-board piece-class softmax with piece-only softmax, argmax. Useful when piece classifier is weaker (was best in v1; gating wins now).

### Data state

- `annotations.json` — primary labels. Three corner-rotation-bug images (12875, 12844, 12986) dropped from `chess_dataset_recovered:val` on 2026-04-27. Pre-cleanup state in `annotations.json.bak_20260427`.
- `training/two_stage/warp_cache/` — 1.7 GB of pre-warped 512×512 PNGs for the piece classifier. Rebuild via `training/two_stage/build_warp_cache.py` if `annotations.json` corner data changes.
- Synthetic piece labels come from FEN in `training/data/data.json` (no `annotations.pieces` entries).

### Failure modes (residual)

- Dominant remaining error is in the **piece classifier**, not the whole-board model. From v2's confusion matrix: white k↔q (118 + 24), black B↔P/N (50+ together), N↔R confusions. v3 with color + type aux heads targets these.
- chessred2k has the highest residual (1.46% cell_err). The other two splits are essentially solved.
- Whole-board model is plateaued — 14 autoresearch keep-iterations moved it 3.13% → 2.90%. Don't expect big additional gains from candidate.py edits.

### Useful commands

```bash
# Reproduce production metrics (combined ~99.09%)
.venv/bin/python training/two_stage/eval_two_stage.py

# Failure visualization (top-3 worst per val split)
.venv/bin/python training/visualize_worst_predictions.py

# Aggregate failure stats (per-class recall, confusion, per-row error)
.venv/bin/python training/analyze_failures.py

# Re-train piece classifier (~1hr GPU on GTX 1070)
.venv/bin/python training/two_stage/train_piece_v3.py --time-budget-s 3600
```

### Out of scope / deferred

- **Whole-board retrain at 384px from scratch.** Tried with resume on 2026-04-27, regressed −1.4pp (overfit hard). A from-scratch ImageNet-init retry is plausible but low expected payoff vs. the piece-classifier lever.
- **Production ONNX export.** `src/ml/inference.ts` still exports the old per-square pipeline. Two-stage productionization (two ONNX models + on-device orchestration) is a separate effort.
- **Piece-classifier autoresearch track.** Would need a parallel harness — meaningful infra work. Defer until manual v3 lands and shows a real ceiling.
- **TTA on the whole-board model.** Training augmentation has no rotations or flips; TTA was tested and regressed. Would require retraining whole-board with hflip aug first.
