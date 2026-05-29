"""Piece-only classifier for the two-stage pipeline.

ResNet-18 backbone (ImageNet pretrained). Two head variants:
- `build_piece_model()`: single 12-way classification head (v1 / v2).
- `build_piece_model_aux()`: 12-way main + 2-way color (black/white) +
  6-way type (king/queen/rook/bishop/knight/pawn) auxiliary heads, sharing
  the avgpool feature. Used by v3 to push down the dominant residual
  k↔q confusion.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

NUM_PIECE_CLASSES = 12

# Piece-class layout (in piece-only space, post-WHOLE_TO_PIECE):
#  0 b  1 k  2 n  3 p  4 q  5 r   6 B  7 K  8 N  9 P  10 Q  11 R
PIECE_TO_COLOR = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]            # 0 black, 1 white
PIECE_TO_TYPE = [2, 0, 3, 4, 1, 5,  2, 0, 3, 4, 1, 5]            # 0 king, 1 queen, 2 bishop, 3 knight, 4 pawn, 5 rook


def _backbone():
    backbone = models.resnet18(weights=None)
    cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if cached.exists():
        state = torch.load(cached, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    else:
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            pass
    return backbone


def build_piece_model():
    """Single-head 12-class model (used by v1 / v2 checkpoints)."""
    backbone = _backbone()
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_PIECE_CLASSES)
    return backbone


class PieceModelAux(nn.Module):
    """Backbone + main piece head + color/type auxiliary heads.

    forward() returns a dict so the training loop can compute the joint
    loss; eval-time argmax uses logits["piece"].
    """

    def __init__(self):
        super().__init__()
        backbone = _backbone()
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_piece = nn.Linear(feat_dim, NUM_PIECE_CLASSES)
        self.head_color = nn.Linear(feat_dim, 2)
        self.head_type = nn.Linear(feat_dim, 6)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "piece": self.head_piece(feat),
            "color": self.head_color(feat),
            "type": self.head_type(feat),
        }


def build_piece_model_aux():
    return PieceModelAux()
