# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Overlay QA image for a raster digitization: the deskewed scan with the followed track edges,
the detected grid lines and every traced curve drawn on top, so a reviewer can see at a glance
whether the trace sits on the ink. Output is a PNG (OpenCV only, no extra dependencies)."""
from __future__ import annotations

import numpy as np

COLORS = [(0, 0, 230), (230, 90, 0), (0, 150, 0), (180, 0, 180)]   # BGR: picked curve first
EDGE_C, GRID_C = (40, 170, 40), (230, 200, 0)


def _edge_x(L, name: str, rows: np.ndarray) -> np.ndarray:
    k = L.edge_names.index(name)
    return np.interp(rows, L.centers, L.edges[:, k])


def grid_lines(panel: np.ndarray, h_frac: float = 0.6, v_frac: float = 0.5):
    """Rows / columns of a track panel that are grid lines (ink across most of the panel)."""
    p = np.asarray(panel, bool)
    return np.flatnonzero(p.mean(1) > h_frac), np.flatnonzero(p.mean(0) > v_frac)


def overlay_scan(R: dict, track: str, out_path: str | None = None, rows: tuple[int, int] | None = None,
                 max_height: int = 6000) -> np.ndarray:
    """Draw ``R`` (result of :func:`workflows.digitize_curve` for ``track``) on its deskewed scan strip.

    Track edges green, grid lines yellow-cyan, the picked curve red and other traced candidates in
    other colours; samples with a null value are not drawn. ``rows`` limits the drawn strip rows
    (default: the traced panel). Returns the BGR image (and writes it when ``out_path`` is given)."""
    import cv2
    L, T = R["layout"], R["trace"]
    r0, r1 = R["panel_rows"] if rows is None else rows
    a, b = max(0, int(r0)), min(L.ink.shape[0], int(r1))
    img = cv2.cvtColor(np.where(L.ink[a:b], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = (img * 0.55 + 255 * 0.45).astype(np.uint8)                   # fade the ink under the overlay
    yy = np.arange(a, b)
    xl, xr = _edge_x(L, track + ".left", yy), _edge_x(L, track + ".right", yy)
    pan = np.asarray(R["panel"], bool); pr0 = int(R["panel_rows"][0]); W = pan.shape[1]
    hr, vc = grid_lines(pan)
    for r in hr:                                                         # horizontal grid
        y = pr0 + r - a
        if 0 <= y < len(yy):
            cv2.line(img, (int(xl[y]), y), (int(xr[y]), y), GRID_C, 1)
    for c in vc:                                                         # vertical grid
        xs = xl + (xr - xl) * c / max(W - 1, 1)
        img[np.arange(len(yy)), np.clip(xs.astype(int), 0, img.shape[1] - 1)] = GRID_C
    for e in (xl, xr):                                                   # track edges
        pts = np.c_[e, np.arange(len(yy))].astype(np.int32)
        cv2.polylines(img, [pts], False, EDGE_C, 2)
    cands = T.get("candidates") or [T]
    pick = T.get("picked", 0) if T.get("candidates") else 0
    order = [pick] + [i for i in range(len(cands)) if i != pick]
    val = np.asarray(R["value"], float)
    for k, i in enumerate(order):
        t = cands[i]
        rr = np.asarray(t["rows"], float); x = np.asarray(t["x"], float)
        ok = np.isfinite(x) & (rr >= a) & (rr < b)
        if i == pick:
            ok &= np.isfinite(val[:len(x)]) if len(val) >= len(x) else ok
        y = (rr - a).astype(int); y = np.clip(y, 0, len(yy) - 1)
        xs = xl[y] + (xr[y] - xl[y]) * x / max(W - 1, 1)
        seg = np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1)
        for s in seg:
            if len(s) > 1:
                cv2.polylines(img, [np.c_[xs[s], y[s]].astype(np.int32)], False, COLORS[k % len(COLORS)], 2 if k == 0 else 1)
    if img.shape[0] > max_height:
        f = max_height / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * f)), max_height), interpolation=cv2.INTER_AREA)
    if out_path:
        cv2.imwrite(str(out_path), img)
    return img
