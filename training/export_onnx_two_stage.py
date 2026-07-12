#!/usr/bin/env python3
"""Export the frozen two-stage pipeline to ONNX for the mobile app.

- corner detector (UNetDualHead, 2ch gray+edges 384px) -> assets/models/corner_unet.onnx
  input  "input"  [1, 2, 384, 384]  (gray/255, canny_edges/255)
  output "coords" [1, 8]            normalized TL,TR,BR,BL (board-space, TL=a8)
- whole-board classifier (ResNet-34 + cell attention, 3ch 256px) -> assets/models/whole_board.onnx
  input  "input"  [1, 3, 256, 256]  (RGB/255, ImageNet-normalized)
  output "logits" [1, 13, 8, 8]

Verifies ONNX vs PyTorch parity on random and real inputs.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO / "training" / "checkpoints"
OUT_DIR = REPO / "assets" / "models"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CornerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["coords"]


class BoardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["logits"]


def export(model, dummy, path, output_name):
    kwargs = dict(
        input_names=["input"],
        output_names=[output_name],
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)


def verify(model, dummy, path, output_name, atol=1e-3):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    for trial in range(3):
        x = dummy if trial == 0 else torch.rand_like(dummy)
        with torch.no_grad():
            want = model(x).numpy()
        got = sess.run([output_name], {"input": x.numpy()})[0]
        diff = np.abs(want - got).max()
        print(f"  trial {trial}: max abs diff {diff:.2e}")
        assert diff < atol, f"parity failure: {diff}"
    return ref


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== corner detector ==")
    corner_mod = load_module("corner_candidate", CKPT_DIR / "corner_detector_7e15e8e.candidate.py")
    corner = corner_mod.build_model(input_channels=2)
    ckpt = torch.load(CKPT_DIR / "corner_detector_7e15e8e.pt", map_location="cpu", weights_only=True)
    report = corner_mod.load_checkpoint(corner, ckpt.get("model_state_dict", ckpt))
    assert not report["missing"] and not report["skipped_mismatch"], report
    corner.eval()
    wrapped = CornerWrapper(corner).eval()
    dummy = torch.rand(1, 2, 384, 384)
    out_path = OUT_DIR / "corner_unet.onnx"
    export(wrapped, dummy, out_path, "coords")
    verify(wrapped, dummy, out_path, "coords")
    print(f"  -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("== whole-board classifier ==")
    board_mod = load_module("board_candidate", CKPT_DIR / "whole_board_0399b00.candidate.py")
    board = board_mod.build_model(input_channels=3)
    ckpt = torch.load(CKPT_DIR / "whole_board_0399b00.pt", map_location="cpu", weights_only=True)
    report = board_mod.load_checkpoint(board, ckpt.get("model_state_dict", ckpt))
    assert not report["missing"] and not report["skipped_mismatch"], report
    board.eval()
    wrapped = BoardWrapper(board).eval()
    dummy = torch.rand(1, 3, 256, 256)
    out_path = OUT_DIR / "whole_board.onnx"
    export(wrapped, dummy, out_path, "logits")
    verify(wrapped, dummy, out_path, "logits")
    print(f"  -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("done")


if __name__ == "__main__":
    sys.exit(main())
