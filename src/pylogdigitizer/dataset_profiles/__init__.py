# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""dataset_profiles: load/validate dataset profile YAMLs (scan conventions, tracks, scales, models).

Workflow: register profile -> collect labels (paired LAS + user corrections) -> train/fine-tune ->
validate on held-out wells (split by well) -> batch-apply with confidence -> review -> retrain.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import os
_REPO = Path(__file__).resolve().parents[3] / "dataset_profiles"
PROFILE_DIR = Path(os.environ.get("PYLOGDIGITIZER_PROFILES", _REPO if _REPO.exists() else Path.cwd() / "dataset_profiles"))
REQUIRED_KEYS = ("name", "description", "scan", "depth", "tracks")

def list_profiles(directory: Path = PROFILE_DIR) -> list[str]:
    return sorted(p.stem for p in Path(directory).glob("*.yaml"))

def load_profile(name_or_path: str, directory: Path = PROFILE_DIR) -> dict[str, Any]:
    import yaml
    p = Path(name_or_path)
    if not p.exists():
        p = Path(directory) / f"{name_or_path}.yaml"
    prof = yaml.safe_load(p.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in prof]
    if missing:
        raise ValueError(f"profile {p} missing keys: {missing}")
    return prof
