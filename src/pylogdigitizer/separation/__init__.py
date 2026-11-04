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


def _runs(row: np.ndarray) -> list:
    """(start, end-exclusive-1) of True runs in a 1-D bool row."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(p), int(q) - 1) for p, q in zip(st, en)]


def _bridge_along(rest: np.ndarray, cand: np.ndarray, reach: int, sigma: float = 6.0) -> np.ndarray:
    """Candidate pixels that have ``rest`` ink on BOTH sides along the local stroke direction
    (structure-tensor orientation of the blurred remaining ink) within ``reach`` px: a gap the
    curve itself runs through, not a cross-direction closing."""
    import cv2
    f = cv2.GaussianBlur(rest.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma); jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    near = cv2.dilate(rest.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)).astype(bool)
    ys, xs = np.nonzero(cand & near)
    out = np.zeros_like(rest)
    if not len(ys):
        return out
    th = 0.5 * np.arctan2(2 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs])   # gradient orientation
    ty, tx = np.cos(th), -np.sin(th)          # stroke direction = perpendicular to gradient
    Hh, Ww = rest.shape
    hit = []
    for sgn in (1, -1):
        h = np.zeros(len(ys), bool)
        for k in range(1, reach + 1):
            yy = np.clip(np.rint(ys + sgn * k * ty).astype(int), 0, Hh - 1)
            xx = np.clip(np.rint(xs + sgn * k * tx).astype(int), 0, Ww - 1)
            h |= rest[yy, xx]
        hit.append(h)
    ok = hit[0] & hit[1]
    out[ys[ok], xs[ok]] = True
    return out


_RG_CACHE: dict = {}


def _ink_key(ink: np.ndarray, *extra) -> tuple:
    import hashlib
    a = np.ascontiguousarray(np.asarray(ink, bool))
    return (a.shape, hashlib.blake2b(np.packbits(a).tobytes(), digest_size=16).hexdigest()) + extra


def remove_grid_cached(ink: np.ndarray, **kw) -> np.ndarray:
    """:func:`remove_grid` memoised on the ink content (+ kwargs); identical output. Small LRU (4)."""
    key = _ink_key(ink, tuple(sorted(kw.items())))
    hit = _RG_CACHE.pop(key, None)
    if hit is None:
        hit = remove_grid(ink, **kw)
        hit.setflags(write=False)
    _RG_CACHE[key] = hit
    while len(_RG_CACHE) > 4:
        _RG_CACHE.pop(next(iter(_RG_CACHE)))
    return hit.copy()


def remove_grid(ink: np.ndarray, length: int = 61, max_width: int = 3, ref_length: int = 301,
                max_bridge: int = 41, return_masks: bool = False, col_frac: float = 0.3,
                use_seg: bool = False, short_length: int = 21, curve_thr: float = 0.5, prob=None):
    """Grid-removed ink: what is left for the tracer once the grid lattice is erased.

    Erased: thin vertical / horizontal grid lines (:func:`grid_mask` rules), full-width horizontal
    bands of any thickness (heavy depth lines, :func:`hline_bands`) and long straight vertical runs
    of any width (thick reference / bit-size lines, >= ``ref_length`` px). Only pixels whose local
    stroke runs along the line direction are erased: a curve crossing a line keeps its pixels
    there, because the gap left in it is bridged by a closing ACROSS the line (vertical closing
    for horizontal lines, horizontal closing for vertical lines) whose result is put back only on
    erased pixels. Gaps up to ``max_bridge`` px (thickest band + 1) are bridged.
    ``return_masks``: also return (h_mask, v_mask, restored)."""
    import cv2
    from pylogdigitizer.grid_rectify import line_response
    ink = np.asarray(ink, bool)
    u8 = ink.astype(np.uint8)
    wv = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, max_width + 1), np.uint8)).astype(bool)
    wh = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((max_width + 1, 1), np.uint8)).astype(bool)
    H = line_response(ink, "h", length) & ~wh
    V = line_response(ink, "v", length) & ~wv
    bands = hline_bands(ink)
    hb = 1
    for a, e in bands:
        H[a:e] |= ink[a:e]
        hb = max(hb, e - a)
    if ref_length and ink.shape[0] >= ref_length:
        V |= line_response(wv, "v", ref_length, tilt=0) & ink   # thick straight vertical runs only
    # detected grid columns (column ink fraction, robust to dashed / broken lines): erase every
    # pixel inside the column whose horizontal ink run is no wider than the column (a curve
    # crossing it is wider and is kept; narrow steep crossings are restored by closing below)
    from pylogdigitizer.track_grid_detection import vertical_lines
    cols = []
    for l in vertical_lines(ink, col_frac):
        x0, x1 = max(0, l["x0"] - 1), min(ink.shape[1], l["x1"] + 1)
        cols.append((x0, x1, l["width"]))
    if cols:
        d = np.diff(np.pad(ink.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        runlen = np.zeros(ink.shape, np.int32)
        for yy, xs in enumerate(d):
            st, en = np.flatnonzero(xs == 1), np.flatnonzero(xs == -1)
            for p_, q_ in zip(st, en):
                runlen[yy, p_:q_] = q_ - p_
        for x0, x1, w in cols:
            V[:, x0:x1] |= ink[:, x0:x1] & (runlen[:, x0:x1] <= w + 1)
    if use_seg or prob is not None:
        # v3: short partial-width horizontal dashes too, then let the seg model protect curve
        # pixels (on / along grid columns, beside thick reference lines) from the line masks
        H |= line_response(ink, "h", short_length, tilt=1) & ~wh
        if prob is None:
            from pylogdigitizer import segmentation as sgm
            prob = sgm.predict_cached(ink)
            pc, pg = prob[sgm.CURVE], prob[sgm.GRID]
        else:
            pc, pg = prob
        protect = ink & (pc >= curve_thr) & (pc > pg)
        H &= ~protect
        V &= ~protect
        # thick partial-width horizontal lines (heavy depth lines broken by the track edges,
        # missed by hline_bands): straight runs >= 30% of the width, any thickness <= max_bridge.
        # Geometry wins over the seg model here; crossings are bridged below.
        L = max(length, int(0.3 * ink.shape[1]))
        th = line_response(wh, "h", L, tilt=0) & ink & ~protect
        if th.any():
            tall = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max_bridge, 1), np.uint8)).astype(bool)
            H |= th & ~tall
        rest = ink & ~H & ~V
        cand = (H | V) & ink
        restored = np.zeros_like(rest)
        for _ in range(3):          # grow the bridge from both stroke ends inward
            add = _bridge_along(rest | restored, cand & ~restored, max_bridge // 2)
            if not add.any():
                break
            restored |= add
        out = rest | restored
        if return_masks:
            return out, H, V, restored
        return out
    vw = 1
    if V.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(V.astype(np.uint8), connectivity=8)
        if n > 1:
            vw = int(min(max_bridge - 1, np.percentile(st[1:, 2], 90)))
    rest = ink & ~H & ~V
    kh = min(max_bridge, hb + 2) | 1
    kv = min(max_bridge, vw + 2) | 1
    close_v = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kh, 1), np.uint8)).astype(bool)
    close_h = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, kv), np.uint8)).astype(bool)
    restored = (H & close_v) | (V & close_h)
    # oblique crossings of erased horizontal strips: join stroke runs just above and below the
    # strip along the curve direction (straight segment, mean run width)
    rest2 = rest | restored
    hrow = H.mean(1) > 0.3
    y = 0
    n_rows, W = ink.shape
    while y < n_rows:
        if not hrow[y]:
            y += 1
            continue
        a = y
        while y < n_rows and hrow[y]:
            y += 1
        e = y
        if a == 0 or e >= n_rows or e - a > max_bridge:
            continue
        up, dn = _runs(rest2[a - 1]), _runs(rest2[e])
        if not up or not dn:
            continue
        h = e - a + 1
        dcen = np.array([(p + q) / 2 for p, q in dn])
        for p, q in up:
            c = (p + q) / 2
            j = int(np.argmin(np.abs(dcen - c)))
            if abs(dcen[j] - c) > 2 * h + 6:
                continue
            p2, q2 = dn[j]
            w = max(q - p, q2 - p2) / 2
            for k, yy in enumerate(range(a, e)):
                t = (k + 1) / h
                cc = c + t * (dcen[j] - c)
                x0, x1 = int(max(0, np.floor(cc - w))), int(min(W, np.ceil(cc + w) + 1))
                restored[yy, x0:x1] |= ink[yy, x0:x1]
    # a crossing pixel of both families (curve through a grid node) also counts once both closings agree
    out = rest | restored
    if return_masks:
        return out, H, V, restored
    return out


def curve_layer(ink: np.ndarray, **kw) -> np.ndarray:
    """Ink with 'other' removed (grid kept: the tracer neutralises it and bridges across it)."""
    ink = np.asarray(ink, bool)
    return ink & ~other_mask(ink, grid_mask(ink))
