# pngtopgn

React Native Expo app that recognizes chess positions from top-down photos and converts them to FEN/PGN. Everything runs on-device (no API calls).

## Pipeline

Photo → **Corner Detection** (find 4 board corners) → **Perspective Warp** → **Square Classification** (13 classes per square) → FEN → PGN/Lichess

## Project Structure

```
app/                     # Expo Router screens
  _layout.tsx            # Root layout
  index.tsx              # Camera/photo picker screen
  result.tsx             # Result display + FEN editing

src/
  ml/inference.ts        # On-device inference (ONNX Runtime)
                         #   - Corner detection model → 4 corner coords
                         #   - JS homography → warp + crop 64 squares
                         #   - Square classifier → piece per square
  chess/fenBuilder.ts    # Board array → FEN string
  chess/pgnExporter.ts   # FEN → PGN/Lichess URL
  components/
    ChessBoard.tsx       # Visual board display
    PieceEditor.tsx      # Manual piece correction UI

training/                # Python training & evaluation scripts
  train_corners_hybrid.py  # Train hybrid corner detector
                           #   - Input ablations: hybrid / gray_edges / gray
                           #   - ResNet-18 → Linear(512, 8) → Sigmoid
                           #   - SmoothL1Loss, 2-phase training (head-only → fine-tune)
  train_corners_heatmap.py # Train heatmap-output corner detector
                           #   - ResNet-18 encoder + decoder → 4 corner heatmaps
                           #   - Argmax / soft-argmax decoding to corner coords
  autoresearch.py          # Autonomous hyperparameter search loop
                           #   - Patches train script → runs experiment → keeps/discards
                           #   - Resumes from best checkpoint, logs to TSV
  detect_board_v5.py       # CV-based board detection (fallback/reference)
                           #   - Multi-method binary images + contour finding
                           #   - Checkerboard validation + Sobel edge grid refinement
  train_squares.py         # Train ResNet-18 square classifier (13 classes)
  prepare_squares.py       # Extract per-square crops (chesscog-style)
  prepare_all_data.py      # Download + prepare all datasets
  eval_gt_corners.py       # Evaluate corner predictions against ground truth
  annotate_corners.py      # Manual corner annotation tool
  add_chess_dataset.py     # Auto-merge samryan18/chess-dataset via CV corners
  add_user_images.py       # Add user photos to training data

assets/                  # App assets (images, model files)
```

## Current State

### Corner Detection — ACTIVE DEVELOPMENT
- **Hybrid ML model**: ResNet-18 with 3-channel input (grayscale + Canny edges + square center heatmap)
- **Input ablation path**: `train_corners_hybrid.py --input-mode gray_edges` tests dropping the square heatmap channel
- **Heatmap output path**: `train_corners_heatmap.py` predicts 4 corner heatmaps instead of 8 direct coordinates
- **Best val_dist: 0.0198** (~60px mean error on 3072px images, ~2% diagonal)
- Trained on ChessReD2K (1,447 train / 330 val)
- Generalizes well to user's own photos (0.0200 norm_dist on 5 test images)
- Autoresearch loop exploring hyperparameters (lr, dropout, head architecture, loss, augmentation, etc.)

### CV Board Detection (detect_board_v5.py) — REFERENCE
- 100% detection rate on 600 images across multiple datasets
- Pure OpenCV, no ML model needed
- Used as reference/fallback, being superseded by ML corner detector

### Square Classifier — NEEDS WORK
- ResNet-18 trained on ChessReD2K tournament board squares
- 13 classes: 6 white pieces + 6 black pieces + empty
- ~92% accuracy on ChessReD2K validation set
- ~65% accuracy on other chess sets (domain gap problem)

### On-Device Inference (src/ml/inference.ts)
- Uses onnxruntime-react-native for model execution
- jpeg-js for pure-JS pixel extraction
- JS homography implementation (no native OpenCV dependency)
- Requires Expo dev client build (not Expo Go)

## Models

Exported as ONNX (~43MB each), stored in `assets/models/` (gitignored):
- **Corner model**: ResNet-18 → 8 floats (4 normalized corner coordinates). Input: 384x384 x 3ch
- **Square classifier**: ResNet-18 → 13 classes. Input: 100x200 (WxH)

## Training Data

All in `training/data/` (gitignored):
- **ChessReD2K**: 1,447 train / 330 val / 306 test images with corner + piece annotations
- **chess-dataset**: 500 real photos of green/white vinyl board with FEN labels (originals + preprocessed)
  - `training/add_chess_dataset.py` can reverse-engineer board corners with `detect_board_v5.py` and merge them into `annotations.json`
- **User images**: 5 manually annotated board photos

## Key Technical Decisions

1. **Two-stage pipeline** (detect corners → classify squares) — each stage independently improvable
2. **3-channel hybrid input** for corner detection — grayscale for appearance, Canny for edges, heatmap for board structure
3. **On-device only** — no cloud APIs, all inference via ONNX Runtime
4. **Chesscog-style square cropping** — variable height/width margins, horizontal flip for left columns, bottom-aligned padding

## Next Steps

See [TODO.md](TODO.md)
