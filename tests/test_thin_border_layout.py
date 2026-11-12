# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Auto layout on digital plots whose grid and thin (3 px) track borders are mid-grey
(e.g. Eli / Gemini triple-combo prints), plus the shipped KGS STONE 5-19 colour crop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pylogdigitizer import track_grid_detection as tgd
from pylogdigitizer.scan_io import load_ink
from pylogdigitizer.workflows import analyze_layout

ROOT = Path(__file__).resolve().parents[1]
STONE = ROOT / "example_data" / "STONE_5-19_15-055-22611"
GREY = (128, 128, 128)
BORDERS = (0, 479, 639, 1118, 1597)          # left x of each 3 px border


def _eli_like(h: int = 3600, w: int = 1600) -> np.ndarray:
    """White sheet, 3 tracks + depth column (linear / log / linear grids), grey 3 px borders
    and grey grid, two coloured curves."""
    rgb = np.full((h, w, 3), 255, np.uint8)
    spans = [(3, 479), (642, 1118), (1121, 1597)]
    for a, b in spans:                       # horizontal grid every 10 px, heavier every 50 px
        for y in range(0, h, 10):
            rgb[y:y + (2 if y % 50 == 0 else 1), a:b] = GREY
    for a, b in (spans[0], spans[2]):        # linear tracks: 10 divisions
        for x in np.linspace(a, b, 11)[1:-1].astype(int):
            rgb[:, x] = GREY
    a, b = spans[1]                          # resistivity track: 3-decade log grid
    for dec in range(3):
        for m in range(1, 10):
            x = int(a + (b - a) * (dec + np.log10(m)) / 3)
            if a < x < b:
                rgb[:, x] = GREY
    for x in BORDERS:
        rgb[:, x:x + 3] = GREY
    y = np.arange(h)
    for (a, b), col in zip(spans[:2], ((0, 8, 252), (245, 28, 2))):
        xc = (a + (b - a) * (0.5 + 0.3 * np.sin(y / 97.0))).astype(int)
        for d in (-1, 0, 1):
            rgb[y, xc + d] = col
    return rgb


@pytest.fixture(scope="module")
def eli_tif(tmp_path_factory):
    p = tmp_path_factory.mktemp("eli") / "eli_like.tif"
    Image.fromarray(_eli_like()).save(p, compression="tiff_lzw")
    return p


def test_grey_grid_is_dropped_by_default_threshold(eli_tif):
    ink = load_ink(eli_tif)
    assert ink[:, 479:482].mean() < 0.01            # grey border is not ink at 128
    with pytest.raises(tgd.LayoutError):
        tgd.find_probe(ink, 800)


def test_eli_like_layout_found_without_positions(eli_tif):
    L = analyze_layout(eli_tif)
    assert L.layout_mode == "grey_thin_borders"
    assert L.track_names == ["track1", "track2", "track3"]
    kinds = [t["kind"] for t in L.tracks]
    assert kinds == ["track", "depth", "track", "track"]
    depth = L.tracks[1]
    assert abs(depth["left"] - 482) <= 3 and abs(depth["right"] - 639) <= 3
    for n, (a, b) in zip(L.track_names, [(1, 480), (640, 1119), (1119, 1598)]):
        li, ri = L.edge_names.index(n + ".left"), L.edge_names.index(n + ".right")
        assert abs(np.median(L.edges[:, li]) - a) <= 3 and abs(np.median(L.edges[:, ri]) - b) <= 3


def test_thin_border_detector_needs_opt_in(eli_tif):
    ink = load_ink(eli_tif, threshold=200)[800:1600]
    thin = tgd.detect_layout(ink, thin_borders=True)
    assert len([t for t in thin if t["kind"] == "track"]) == 3
    assert thin[0]["detector"] == "detect_tracks_thin"


stone_crop = STONE / "triple_combo_color_crop.tif"
needs_stone = pytest.mark.skipif(not stone_crop.exists(), reason="STONE example crop not present")


@needs_stone
def test_stone_crop_auto_layout_and_depth_labels():
    L = analyze_layout(stone_crop)
    assert L.layout_mode == "grey_thin_borders"
    assert L.track_names == ["track1", "track2", "track3"]
    assert len(L.labels) >= 9                         # 4400 ... 4850 ft every 50 ft
    rows = L.label_rows
    assert np.allclose(np.diff(rows), 500, atol=3)   # 10 px/ft


@needs_stone
def test_stone_working_image_keeps_three_tracks(tmp_path):
    """1-bit per-curve working image: the grey grid becomes black, so the default pass can
    succeed through a fallback detector that merges tracks 2 and 3; the thin-border layout
    (more tracks, same depth column) must win."""
    from pylogdigitizer.curve_ink import make_base_ink, make_curve_ink, save_ink_tif
    from pylogdigitizer.scan_color import classify_colors, suggest_palette
    from pylogdigitizer.scan_io import load_rgb
    rgb = load_rgb(stone_crop)
    labels, names = classify_colors(rgb, suggest_palette(rgb))
    body = (414, rgb.shape[0])
    base = make_base_ink(labels, names, body)
    assert base[body[0]:, 479:482].mean() > 0.9       # grey border kept as ink
    ink = make_curve_ink(base, labels, names, "blue", (1, 480), body)
    L = analyze_layout(save_ink_tif(ink, tmp_path / "working_GR.tif"))
    assert L.layout_mode == "thin_borders"
    assert L.track_names == ["track1", "track2", "track3"]
    assert len(L.labels) >= 9
