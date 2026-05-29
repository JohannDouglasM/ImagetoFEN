#!/usr/bin/env python3
"""Two-stage eval: whole-board classifier + piece-only refinement.

Pipeline per image:
  1. Whole-board model -> 8x8 piece logits (13 classes incl empty)
  2. For every cell whose argmax is non-empty, crop 96x96 around that cell
     from a 512px re-warp and feed to the piece-only classifier.
  3. Replace the whole-board prediction for that cell with the piece-only
     prediction (mapped back to the 13-class space).

The empty-vs-piece decision still comes from the whole-board model (which is
already 99.7% accurate on that decision). The piece classifier only fixes
piece-type confusions (B↔P, b↔r, N↔R, ...) which dominate residual error.
"""

import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
WORKTREE = Path("/home/johann/autoresearch/20260423-whole_board_classifier")
HARNESS_DIR = WORKTREE / "training" / "autoresearch_v3"
ANNOTATIONS = "/home/johann/ImagetoFEN/annotations.json"
IMAGES_ROOT = "/home/johann/ImagetoFEN"

# Whole-board model defaults (use the 384 run if it exists, else fall back to current best)
WB_384 = Path("/home/johann/ImagetoFEN/training/wb_384/run1/best.pt")
WB_256 = Path("/home/johann/autoresearch/20260423-whole_board_classifier/training/autoresearch_v3/runs/20260423-whole_board_classifier/artifacts/20260425T091625Z_0399b00/best.pt")
PIECE_CKPT_V1 = THIS_DIR / "run1" / "best.pt"
PIECE_CKPT_V2 = THIS_DIR / "run2" / "best.pt"
PIECE_CKPT_V3 = THIS_DIR / "run3" / "best.pt"
# v3 (run3) tried color + type aux heads — net slightly worse on combined
# gated than v2 (99.05% vs 99.09%). Keeping v2 as the production model.
# Set PIECE_CKPT_OVERRIDE=run3 to load v3 explicitly.
import os as _os
_override = _os.environ.get("PIECE_CKPT_OVERRIDE", "").lower()
if _override == "run3" and PIECE_CKPT_V3.exists():
    PIECE_CKPT = PIECE_CKPT_V3
    PIECE_VARIANT = "aux"
elif PIECE_CKPT_V2.exists():
    PIECE_CKPT = PIECE_CKPT_V2
    PIECE_VARIANT = "single"
else:
    PIECE_CKPT = PIECE_CKPT_V1
    PIECE_VARIANT = "single"
PIECE_TTA = True  # hflip avg; safe because v2/v3 trained with hflip aug

LETTER = ["b", "k", "n", "p", "q", "r", ".", "B", "K", "N", "P", "Q", "R"]
EMPTY = 6

VAL_SPLITS = [
    ("chessred2k", "val"),
    ("chess_dataset_recovered", "val"),
    ("synthetic", "val"),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    sys.path.insert(0, str(HARNESS_DIR))
    candidate = load_module("autoresearch_candidate", HARNESS_DIR / "candidate.py")
    harness = load_module("fixed_harness_board", HARNESS_DIR / "fixed_harness_board.py")
    from piece_dataset import warp_board, crop_cell, patch_to_tensor, PIECE_TO_WHOLE, WARP_BOARD_PX
    from piece_model import build_piece_model, build_piece_model_aux

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WB_384 regressed (-1.4pp combined). Pin to the original 256 best.
    wb_ckpt_path = WB_256
    wb_model = candidate.build_model(input_channels=3)
    ckpt = torch.load(str(wb_ckpt_path), map_location="cpu", weights_only=True)
    candidate.load_checkpoint(wb_model, ckpt.get("model_state_dict", ckpt))
    wb_model.eval().to(device)
    print(f"whole-board: {wb_ckpt_path}")

    if not PIECE_CKPT.exists():
        print(f"NOTE: piece checkpoint not found at {PIECE_CKPT} — running stage-1 only")
        piece_model = None
    else:
        if PIECE_VARIANT == "aux":
            piece_model = build_piece_model_aux()
        else:
            piece_model = build_piece_model()
        ck = torch.load(str(PIECE_CKPT), map_location="cpu", weights_only=True)
        piece_model.load_state_dict(ck["model_state_dict"])
        piece_model.eval().to(device)
        print(f"piece-only:  {PIECE_CKPT}  variant={PIECE_VARIANT}  "
              f"(trained val_acc={ck.get('val_acc', 'n/a')})")

    defaults = candidate.get_defaults()

    print(f"\n{'split':<35} {'mode':>22} {'cell_acc':>9} {'mean_err':>9} {'max_err':>8}")
    print("-" * 95)
    combined = {"stage1": [], "two_stage_unconditional": [], "two_stage_gated": [], "ensemble": []}
    for group, split in VAL_SPLITS:
        bd = harness.BoardDataset(
            ANNOTATIONS, IMAGES_ROOT, group=group, split=split,
            candidate=candidate, candidate_defaults=defaults,
            input_mode=defaults["input_mode"], img_size=defaults["img_size"],
            augment=False, seed=1337,
        )
        loader = DataLoader(bd, batch_size=8, shuffle=False, num_workers=0)
        results = run_split(wb_model, piece_model, bd, loader, device)
        for mode, errs in results.items():
            errs = np.array(errs)
            print(f"{group}:{split:<24} {mode:>22} {1-errs.mean():>9.4f} {errs.mean():>9.4f} {errs.max():>8.4f}")
            combined[mode].extend(errs.tolist())
        print()

    print("-" * 95)
    for mode in ("stage1", "two_stage_unconditional", "two_stage_gated", "ensemble"):
        if not combined[mode]:
            continue
        e = np.array(combined[mode])
        print(f"COMBINED  {mode:>22}  cell_acc={1-e.mean():.4f}  mean_err={e.mean():.4f}  "
              f"max={e.max():.4f}  p95={np.quantile(e, 0.95):.4f}")


def _piece_logits(piece_model, patches):
    """Extract 12-class piece logits, accepting either a single-head or
    aux-head model variant transparently."""
    out = piece_model(patches)
    return out["piece"] if isinstance(out, dict) else out


def run_split(wb_model, piece_model, bd, loader, device, confidence_threshold=0.999):
    """Two-stage refinement gated on whole-board confidence.

    For non-empty cells where whole-board's max softmax over the 12 piece
    classes (renormalized over non-empty classes) is BELOW the threshold,
    we replace its prediction with the piece classifier's. Above the
    threshold, we trust whole-board.

    Threshold 0.999 was empirically best vs unconditional swap on combined
    val: 99.09% gated vs 99.00% unconditional. It captures the chessred2k
    win (+2.95pp) without hurting synthetic where wb is already perfect.
    """
    from piece_dataset import warp_board, crop_cell, patch_to_tensor, PIECE_TO_WHOLE, WHOLE_TO_PIECE, WARP_BOARD_PX
    import torch.nn.functional as F

    results = {"stage1": [], "two_stage_unconditional": [], "two_stage_gated": [], "ensemble": []}
    if piece_model is None:
        results = {"stage1": []}

    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            labels = targets["labels"].to(device)
            wb_logits = wb_model(images)["logits"]  # [B, 13, 8, 8]
            wb_preds = wb_logits.argmax(dim=1)
            # Confidence over non-empty classes only
            piece_class_idx = torch.tensor(list(WHOLE_TO_PIECE.keys()), device=device, dtype=torch.long)
            wb_piece_logits = wb_logits.index_select(1, piece_class_idx)  # [B, 12, 8, 8]
            wb_piece_softmax = F.softmax(wb_piece_logits, dim=1)
            wb_piece_conf = wb_piece_softmax.max(dim=1).values  # [B, 8, 8]

            for b in range(wb_preds.size(0)):
                stage1 = wb_preds[b].cpu().numpy()
                gt = labels[b].cpu().numpy()
                results["stage1"].append((stage1 != gt).mean())

                if piece_model is None:
                    continue

                image_id = int(meta["image_id"][b])
                item = next((it for it in bd.items if it.image_id == image_id), None)
                if item is None or (image := cv2.imread(str(item.image_path))) is None:
                    for k in ("two_stage_unconditional", "two_stage_gated", "ensemble"):
                        results[k].append(results["stage1"][-1])
                    continue
                warped = warp_board(image, item.corners, out_size=WARP_BOARD_PX)

                cells = [(r, c) for r in range(8) for c in range(8) if stage1[r, c] != EMPTY]
                if not cells:
                    for k in ("two_stage_unconditional", "two_stage_gated", "ensemble"):
                        results[k].append(results["stage1"][-1])
                    continue

                patches = torch.stack([patch_to_tensor(crop_cell(warped, r, c)) for r, c in cells]).to(device)
                piece_logits = _piece_logits(piece_model, patches)
                if PIECE_TTA:
                    piece_logits_flip = _piece_logits(piece_model, torch.flip(patches, dims=(3,)))
                    piece_logits = (piece_logits + piece_logits_flip) / 2.0
                piece_softmax = F.softmax(piece_logits, dim=1).cpu().numpy()
                piece_preds = piece_logits.argmax(dim=1).cpu().numpy()

                refined_unc = stage1.copy()
                refined_gated = stage1.copy()
                refined_ens = stage1.copy()
                for idx, (r, c) in enumerate(cells):
                    pp = int(piece_preds[idx])
                    refined_unc[r, c] = PIECE_TO_WHOLE[pp]
                    if float(wb_piece_conf[b, r, c].item()) < confidence_threshold:
                        refined_gated[r, c] = PIECE_TO_WHOLE[pp]
                    # Ensemble: average wb_piece_softmax + piece_softmax over piece classes,
                    # pick argmax in piece-class space.
                    wb_p = wb_piece_softmax[b, :, r, c].cpu().numpy()
                    avg = (wb_p + piece_softmax[idx]) / 2.0
                    refined_ens[r, c] = PIECE_TO_WHOLE[int(avg.argmax())]

                results["two_stage_unconditional"].append((refined_unc != gt).mean())
                results["two_stage_gated"].append((refined_gated != gt).mean())
                results["ensemble"].append((refined_ens != gt).mean())

    return results


if __name__ == "__main__":
    main()
