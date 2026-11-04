# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Dependency trim: in-house Sauvola matches the formula; dropped packages are not imported by the package."""
import pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sauvola_matches_bruteforce():
    from pylogdigitizer.scan_cleanup import threshold_sauvola, binarize
    rng = np.random.default_rng(0)
    img = (rng.random((40, 50)) * 255).astype(np.uint8); img[10:14] = 5
    w = 7; T = threshold_sauvola(img, w)
    pad = np.pad(img.astype(float), w // 2, mode="reflect"); r = 0.5 * (img.max() - img.min())
    for y, x in [(0, 0), (12, 20), (39, 49), (25, 3)]:
        win = pad[y:y + w, x:x + w]
        assert abs(T[y, x] - win.mean() * (1 + 0.2 * (win.std() / r - 1))) < 1e-6
    assert binarize(img, "sauvola")[11].mean() > 0.9


def test_dropped_dependencies_not_imported():
    deps = re.search(r"dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.S).group(1)
    for name in ("scipy", "scikit-image", "tifffile"):
        assert f'"{name}"' not in deps
    pat = re.compile(r"^\s*(import|from)\s+(scipy|skimage|tifffile)\b", re.M)
    hits = [str(p) for p in (ROOT / "src").rglob("*.py") if pat.search(p.read_text())]
    assert not hits, hits
