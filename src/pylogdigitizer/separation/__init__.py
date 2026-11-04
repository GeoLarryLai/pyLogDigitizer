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
    out = (vg | hg) & g
    # full-width horizontal bands of any thickness (thick depth lines are > max_width) are grid,
    # except where a curve stroke crosses them (ink right above and below the band)
    bands = hline_bands(ink)
    for a, e in bands:
        up = ink[max(0, a - 3):a].any(axis=0) if a > 0 else np.zeros(ink.shape[1], bool)
        dn = ink[e:e + 3].any(axis=0)
        cross = cv2.dilate((up & dn).astype(np.uint8)[None], np.ones((1, 5), np.uint8))[0].astype(bool)
        out[a:e] |= ink[a:e] & ~cross[None, :]
    return out


def hline_bands(ink: np.ndarray, frac: float = 0.6, max_h: int = 40, min_run: float = 0.25) -> list[tuple[int, int]]:
    """Full-track-width horizontal line bands (thin grid rows and thick depth lines, e.g. every
    100 ft), as (first_row, end_row). A row belongs to a band when >= ``frac`` of it is ink in
    horizontal runs of >= ``min_run`` x width (a curve, even a flat one, never fills a row like that);
    bands taller than ``max_h`` rows (stamps, header blocks) are dropped."""
    ink = np.asarray(ink, bool)
    h, w = ink.shape
    if not h or not w:
        return []
    import cv2
    L = max(15, int(min_run * w))
    runs = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, L), np.uint8)).astype(bool)
    rows = np.flatnonzero(runs.mean(axis=1) >= frac)
    if not len(rows):
        return []
    out = []
    for b in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
        a, e = int(b[0]), int(b[-1]) + 1
        # include the faded fringe rows of the band (half coverage) on both sides
        cov = runs.mean(axis=1)
        while a > 0 and cov[a - 1] >= frac / 2:
            a -= 1
        while e < h and cov[e] >= frac / 2:
            e += 1
        if e - a <= max_h and (not out or a > out[-1][1]):
            out.append((a, e))
    return out


def band_rows_mask(h: int, bands) -> np.ndarray:
    m = np.zeros(h, bool)
    for a, e in bands:
        m[a:e] = True
    return m


def thick_band_anchors(bands, min_h: int | None = None) -> list[float]:
    """Centre rows of the thicker bands (heavier than the median grid row: the 50/100-ft depth
    lines). Periodic in depth, so usable as depth anchors / QC of the label lattice."""
    if not bands:
        return []
    hs = np.array([e - a for a, e in bands])
    thr = min_h if min_h is not None else max(np.median(hs) * 1.6, np.median(hs) + 2)
    return [0.5 * (a + e - 1) for (a, e), hh in zip(bands, hs) if hh >= thr]


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
        o = ch[np.argsort(x[ch], kind="stable")]
        xo = x[o]
        hmax = int(h[o].max())
        used = set()
        for i in o:   # greedy chains to the right on the same baseline
            if i in used:
                continue
            chain = [i]
            while True:
                j0 = chain[-1]; best = None
                lo = np.searchsorted(xo, x[j0] + w[j0] - 2, "left")
                hi = np.searchsorted(xo, x[j0] + w[j0] + gap * max(h[j0], hmax), "right")
                for j in o[lo:hi]:
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
