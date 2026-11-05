"""Appearance statistics of a 1-bit log-scan crop (used to fit the synthetic tile generator and to
compare synthetic against real): grid line widths / spacing, curve stroke width, edge roughness,
speckle and pin-hole dropouts."""
from __future__ import annotations

import numpy as np


def _bands(m):
    p = np.r_[0, m.astype(np.int8), 0]; d = np.diff(p)
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def scan_stats(ink: np.ndarray, v_cover: float = 0.6, h_cover: float = 0.5) -> dict:
    """``ink``: bool crop (True = ink). Returns a flat dict of scalar statistics."""
    import cv2
    ink = np.asarray(ink, bool); H, W = ink.shape
    a, b = _bands(ink.mean(0) >= v_cover); vw = b - a
    gy = np.zeros(W, bool)
    for s, e in zip(a, b):
        gy[s:e] = True
    hc = ink[:, ~gy].mean(1) if (~gy).any() else np.zeros(H)
    c, d = _bands(hc >= h_cover); hw = d - c
    hcent = (c + d) / 2
    gr = np.zeros(H, bool)
    for s, e in zip(c, d):
        gr[s:e] = True
    cur = ink.copy(); cur[:, gy] = False; cur[gr] = False
    u = cur.astype(np.uint8)
    dt = cv2.distanceTransform(np.pad(u, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    ridge = cur & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (dt >= 1)
    sw = 2 * dt[ridge] if ridge.any() else np.array([0.0])
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    speck = int(((st[1:, cv2.CC_STAT_AREA] <= 4)).sum())
    n2, _, st2, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    holes = int((st2[1:, cv2.CC_STAT_AREA] <= 4).sum())
    er = cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge = (ink & ~er).sum() / max(ink.sum(), 1)
    mp = H * W / 1e6
    return dict(ink_frac=float(ink.mean()), v_lines=len(vw), v_width_med=float(np.median(vw)) if len(vw) else 0.0,
                v_width_max=float(vw.max()) if len(vw) else 0.0, h_lines=len(hw),
                h_width_med=float(np.median(hw)) if len(hw) else 0.0, h_width_max=float(hw.max()) if len(hw) else 0.0,
                h_spacing=float(np.median(np.diff(hcent))) if len(hcent) > 2 else 0.0,
                stroke_med=float(np.median(sw)), stroke_p90=float(np.percentile(sw, 90)),
                curve_frac=float(cur.mean()), edge_ratio=float(edge),
                speckle_per_mpx=speck / mp, holes_per_mpx=holes / mp)
