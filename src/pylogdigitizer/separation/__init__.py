# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""separation: per-pixel grid / curve / other labels for a (straightened) track panel, by rules.

- grid: long thin straight runs (vertical or horizontal) no wider than the grid-line width;
- other: text (rows of small blobs side by side: depth numbers, remarks, handwriting) and stamps /
  blocks (large components that are densely filled), i.e. ink that is neither grid nor curve;
- curve: the remaining ink.
``other`` is what the tracer must ignore; ``grid`` is crossed but not followed (curve_tracing
already treats straight lines as neutral). A trained segmentation model can replace these rules
where it is confident (see ml_seg)."""
from __future__ import annotations
import numpy as np

PAPER, GRID, CURVE, OTHER = 0, 1, 2, 3


def grid_mask(ink: np.ndarray, length: int = 61, max_width: int = 3) -> np.ndarray:
    """Long straight thin lines (grid). Thick straight runs (a flat curve) are not grid."""
    import cv2
    from pylogdigitizer.grid_rectify import line_response
    ink = np.asarray(ink, bool)
    g = line_response(ink, "v", length) | line_response(ink, "h", length)
    # too wide to be a grid line: survives an opening wider than max_width across the line
    wv = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max_width + 1), np.uint8)).astype(bool)
    wh = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max_width + 1, 1), np.uint8)).astype(bool)
    vg = line_response(ink, "v", length) & ~wv
    hg = line_response(ink, "h", length) & ~wh
    return (vg | hg) & g


def other_mask(ink: np.ndarray, grid: np.ndarray | None = None, char_h: tuple[int, int] = (6, 45),
               min_chars: int = 3, gap: float = 1.2, stamp_min: int = 80, stamp_fill: float = 0.3) -> np.ndarray:
    """Text lines (>= ``min_chars`` character-sized blobs side by side on one baseline, gaps <=
    ``gap`` x height) and stamps (components >= ``stamp_min`` px both ways with fill >= ``stamp_fill``)."""
    import cv2
    ink = np.asarray(ink, bool)
    if grid is None:
        grid = grid_mask(ink)
    rest = (ink & ~grid).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(rest, connectivity=8)
    out = np.zeros(ink.shape, bool)
    x, y, w, h, a = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    keep = np.zeros(n, bool)
    ch = np.flatnonzero((h >= char_h[0]) & (h <= char_h[1]) & (w <= 3 * h) & (w >= 2))
    ch = ch[ch > 0]
    if len(ch):
        o = ch[np.argsort(x[ch])]
        used = set()
        for i in o:   # greedy chains to the right on the same baseline
            if i in used:
                continue
            chain = [i]
            while True:
                j0 = chain[-1]; best = None
                for j in o:
                    if j in used or j in chain:
                        continue
                    dx = x[j] - (x[j0] + w[j0])
                    if -2 <= dx <= gap * max(h[j0], h[j]) and abs((y[j] + h[j]) - (y[j0] + h[j0])) <= 0.35 * max(h[j0], h[j]) \
                            and 0.6 <= h[j] / h[j0] <= 1.6:
                        if best is None or dx < best[0]:
                            best = (dx, j)
                if best is None:
                    break
                chain.append(best[1])
            if len(chain) >= min_chars:
                keep[chain] = True; used.update(chain)
    out = keep[lab] & rest.astype(bool)
    # stamps / filled blocks: what survives a square opening far wider than any curve stroke
    k = max(9, stamp_min // 6)
    solid = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if solid.any():
        m, sl, ss, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        ok = np.zeros(m, bool)
        ok[1:] = (ss[1:, 2] >= stamp_min) & (ss[1:, 3] >= stamp_min) & (ss[1:, 4] >= stamp_fill * ss[1:, 2] * ss[1:, 3])
        out |= cv2.dilate(ok[sl].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ink
    return out


def classify(ink: np.ndarray, **kw) -> np.ndarray:
    """uint8 labels PAPER / GRID / CURVE / OTHER."""
    ink = np.asarray(ink, bool)
    g = grid_mask(ink, **{k: v for k, v in kw.items() if k in ("length", "max_width")})
    o = other_mask(ink, g)
    lab = np.zeros(ink.shape, np.uint8)
    lab[ink] = CURVE; lab[g] = GRID; lab[o] = OTHER
    return lab


def curve_layer(ink: np.ndarray, **kw) -> np.ndarray:
    """Ink with 'other' removed (grid kept: the tracer neutralises it and bridges across it)."""
    ink = np.asarray(ink, bool)
    return ink & ~other_mask(ink, grid_mask(ink))
