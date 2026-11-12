# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Draw a vector PDF page as a picture (for the app's viewer and figures), with matplotlib.

The page is drawn from what :mod:`pylogdigitizer.pdf_vector.extract` reads: every stroked path in
its colour, width and dash, the text runs, and the placed pictures. Filled areas are not drawn,
so the picture is a faithful view of the lines and text, not a print-perfect copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _image_in_frame(arr: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Turn/flip an image's pixel array so it shows as placed in the frame (``A``: unit square of the
    image to the frame, 2x2 part; only quarter turns and flips are handled)."""
    if abs(A[0, 1]) < 1e-9 and abs(A[1, 0]) < 1e-9:          # upright or flipped
        if A[0, 0] < 0:
            arr = arr[:, ::-1]
        if A[1, 1] < 0:
            arr = arr[::-1]
        return arr
    arr = arr.transpose(1, 0, 2)                                # turned a quarter
    if A[0, 1] > 0:
        arr = arr[:, ::-1]
    if A[1, 0] > 0:
        arr = arr[::-1]
    return arr


def render_page(path: str | Path, page: int = 0, scale: float = 3.0, rotation: int = 0) -> np.ndarray:
    """RGB uint8 array (H, W, 3) of the page, ``scale`` pixels per PDF point (3 = 216 dpi).
    ``rotation`` (0/90/180/270, clockwise) draws the page turned, in the log frame of
    :func:`~pylogdigitizer.pdf_vector.vector_log.read_vector_page` (``vp.rotation``).
    Pixel row 0 is the top; ``x_px = x * scale``, ``row = (height - y) * scale`` in that frame."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from pypdf import PdfReader
    from pylogdigitizer.pdf_vector.extract import extract_strokes, extract_text_runs, frame_transform, apply_frame
    from pylogdigitizer.input_kind import _image_placements
    del matplotlib
    rotation = int(rotation) % 360
    pg = PdfReader(str(path)).pages[page]
    b = pg.mediabox
    box = (float(b.left), float(b.bottom), float(b.right), float(b.top))
    M = frame_transform(box, rotation)
    W, H = box[2] - box[0], box[3] - box[1]
    if rotation % 180:
        W, H = H, W
    w_px, h_px = int(round(W * scale)), int(round(H * scale))
    fig = Figure(figsize=(w_px / 100.0, h_px / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    pt = scale * 72.0 / 100.0                     # one PDF point in matplotlib points at this size
    for _x0, _y0, _x1, _y1, m, xo in _image_placements(pg):
        if xo is None:
            continue
        try:
            arr = np.asarray(xo.decode_as_image().convert("RGB"))
            A = M[:2, :2] @ np.asarray(m, float)[:2, :2]
            c = apply_frame(M, apply_frame(np.asarray(m, float), [[0, 0], [1, 0], [0, 1], [1, 1]]))
            ax.imshow(_image_in_frame(arr, A), extent=(c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()),
                      aspect="auto", zorder=0, interpolation="nearest")
        except Exception:
            continue
    strokes = extract_strokes(pg)
    if strokes:
        segs = [apply_frame(M, s.pts) for s in strokes]
        lc = LineCollection(segs, colors=[tuple(np.clip(s.color, 0, 1)) for s in strokes],
                            linewidths=[max(0.5, s.width * pt) for s in strokes], zorder=2)
        lc.set_linestyles([(0, tuple(d * pt for d in s.dash)) if s.dash and any(s.dash) else "solid" for s in strokes])
        ax.add_collection(lc)
    for t in extract_text_runs(pg):
        if t.size <= 0:
            continue
        x, y = apply_frame(M, [[t.x, t.y]])[0]
        ang = ((t.angle if t.angle >= 0 else 0) - rotation) % 360
        ax.text(x, y, t.text, fontsize=t.size * pt * 0.92, va="baseline", ha="left", zorder=3, rotation=ang,
                rotation_mode="anchor", family="DejaVu Sans", color="black")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return rgb[:h_px, :w_px]
