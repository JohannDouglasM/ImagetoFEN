#!/usr/bin/env python3
"""
Default mutable candidate for autoresearch v3.

This file starts as the ResNet coordinate baseline. Track-specific worktrees can
replace it with other templates, for example the U-Net dual-head version.
"""

import importlib.util
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "resnet_coords.py"
_SPEC = importlib.util.spec_from_file_location("autoresearch_default_candidate", _TEMPLATE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load default candidate template from {_TEMPLATE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MODULE, _name)
