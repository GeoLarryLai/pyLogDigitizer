#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Render an editor-like view (sheet crop + trace overlay + anchors) of an AssistedSession to PNG
with matplotlib - used for headless checks."""
from __future__ import annotations
import numpy as np


def render(s, depth_center: float, half_ft: float, out: str, title: str = "", ref: bool = True, ax=None):
    from matplotlib.figure import Figure
    dm = s.dm; r0, r1 = int(dm.inverse(depth_center - half_ft)), int(dm.inverse(depth_center + half_ft))
    r0, r1 = max(0, r0), min(s.ink.shape[0], r1)
    xl, xr = s._edges(np.array([r0, r1]))
    x0, x1 = int(max(0, min(xl) - 60)), int(min(s.ink.shape[1], max(xr) + 60))
    own = ax is None
    if own:
        fig = Figure(figsize=(5, 7)); ax = fig.subplots()
    ax.imshow(~s.ink[r0:r1, x0:x1], cmap="gray", extent=(x0, x1, r1, r0), aspect="auto", interpolation="antialiased")
    ov = s.overlay((r0, r1), 20000)
    if ov["trace"]:
        t = np.array(ov["trace"]); ax.plot(t[:, 1], t[:, 0], "r-", lw=1.2, alpha=0.8, label="trace")
    if ref and s.reference() is not None:
        from pylogdigitizer import scale_calibration as scm
        d, v = s.reference(); m = (d >= depth_center - half_ft) & (d <= depth_center + half_ft)
        sm = (scm.log_scale if s.scale["type"] == "log" else scm.linear_scale)(0, s.width, s.scale["left"], s.scale["right"])
        rr = dm.inverse(d[m]); xp = sm.inverse(v[m]); xs = s.panel_to_sheet(rr, np.clip(xp, -5, s.width + 5))
        ax.plot(xs, rr, "c--", lw=1, alpha=0.9, label="paired LAS (main scale)")
    for a0, a1 in ov.get("backup_intervals", []):
        if a1 >= r0 and a0 <= r1:
            ax.axhspan(max(a0, r0), min(a1, r1), color="orange", alpha=0.15, lw=0)
    for a in ov["anchors"]:
        if r0 <= a[0] <= r1:
            ax.plot(a[1], a[0], "o", ms=9, mfc="none", mec="lime", mew=2.5)
    for xx in (xl, xr):
        ax.plot(xx, [r0, r1], "b:", lw=0.8)
    yt = np.linspace(r0, r1, 5); ax.set_yticks(yt); ax.set_yticklabels([f"{dm(y):.0f}" for y in yt])
    ax.set_ylabel("depth (ft)"); ax.set_xlabel("sheet x (px)"); ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    if own:
        fig.tight_layout(); fig.savefig(out, dpi=100)
    return out
