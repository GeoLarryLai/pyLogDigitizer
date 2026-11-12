# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scan_color: palette classification of colour raster well-log scans.

A colour scan prints each curve in its own ink colour. Labelling every pixel with
its nearest palette colour lets later steps build one working image per curve
(see :mod:`pylogdigitizer.curve_ink`) so the tracer sees a single line.

``white`` (paper) and ``black`` (grid, depth numbers, header text) are required
palette entries. Other names (red, green, blue, ...) are the curve / shading
colours. :func:`suggest_palette` proposes a palette by clustering in CIELAB
(ignoring near-white paper and merging near-duplicates); the caller confirms or
edits it. :func:`classify_colors` then labels the sheet.
"""
from __future__ import annotations

import colorsys

import numpy as np

from pylogdigitizer.scan_color.plots import plot_color_autodetect, plot_color_trace_pairs

__all__ = [
    "COLOR_TRACE_PRESET", "require_palette", "suggest_palette",
    "classify_colors", "find_log_body", "legend_color", "is_color_rgb",
    "plot_color_autodetect", "plot_color_trace_pairs",
]

# Tracer settings matched to a thickened per-curve working image (gap-closed,
# ~2 + 2*thicken_px wide). Do not change the global ``trace_track`` defaults.
COLOR_TRACE_PRESET = dict(min_width=5, straight_len=0, grid_frac=1.01,
                          center_weight=0.3, min_area=0, line_frac=0.9)

_NAME_HUES = [
    (0.00, "red"), (0.08, "orange"), (0.15, "yellow"), (0.35, "green"),
    (0.50, "cyan"), (0.66, "blue"), (0.78, "purple"), (0.90, "magenta"), (1.00, "red"),
]


def require_palette(palette: dict) -> list[str]:
    """Return palette names; raise if ``white`` or ``black`` is missing."""
    names = list(palette)
    for need in ("white", "black"):
        if need not in names:
            raise ValueError(f"palette needs a {need!r} entry")
    return names


def is_color_rgb(rgb: np.ndarray, min_share: float = 0.002, spread: int = 18) -> bool:
    """True when a noticeable share of pixels is chromatic (not gray/paper)."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        return False
    s = a[::4, ::4, :3].astype(np.int16)
    chroma = np.max(s, axis=-1) - np.min(s, axis=-1)
    return float((chroma >= spread).mean()) >= min_share


def _lab(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB).astype(np.float32)


def _name_rgb(rgb: tuple[int, int, int], used: set[str]) -> str:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.10:
        base = "gray" if mx > 0.18 else "black"
    else:
        h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
        base = _NAME_HUES[-1][1]
        # pick the closer of the two neighbours around the wrap
        for i, (hh, nm) in enumerate(_NAME_HUES[:-1]):
            nxt = _NAME_HUES[i + 1]
            if hh <= h <= nxt[0]:
                base = nm if (h - hh) <= (nxt[0] - h) else nxt[1]
                break
    name = base
    k = 2
    while name in used:
        name = f"{base}_{k}"
        k += 1
    used.add(name)
    return name


def suggest_palette(rgb: np.ndarray, max_colors: int = 8, min_share: float = 0.0005,
                    merge_lab: float = 18.0, sample: int = 3) -> dict:
    """Propose a palette from a colour scan (CIELAB clustering).

    Near-white, low-chroma paper is ignored. Near-black ink is kept as ``black``.
    Remaining chromatic pixels are k-means clustered in Lab; centres closer than
    ``merge_lab`` are merged. Names are hue-based (red, green, blue, ...) with
    ``_2``, ``_3`` suffixes if needed. Always includes ``white`` and ``black``.
    """
    import cv2
    rgb = np.asarray(rgb)
    a = rgb[::sample, ::sample, :3]
    lab = _lab(a)
    L, A, B = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    chroma = np.sqrt(A * A + B * B)
    paper = (L > 230) & (chroma < 12)
    black = (L < 42) & (chroma < 16)
    color = ~(paper | black)
    out = {"white": (255, 255, 255), "black": (0, 0, 0)}
    used = {"white", "black"}
    pix = lab[color].reshape(-1, 3)
    if len(pix) < 20:
        return out
    # k-means: try up to max_colors-2 chromatic clusters
    k = int(np.clip(max_colors - 2, 1, 8))
    k = min(k, max(1, len(pix) // 30))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _compact, labels, centers = cv2.kmeans(
        pix.astype(np.float32), k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    # merge near-duplicate centres
    centers = centers.astype(np.float64)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
    alive = [True] * len(centers)
    for i in range(len(centers)):
        if not alive[i]:
            continue
        for j in range(i + 1, len(centers)):
            if not alive[j]:
                continue
            if np.linalg.norm(centers[i] - centers[j]) < merge_lab:
                w = counts[i] + counts[j]
                centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / max(w, 1)
                counts[i] = w
                alive[j] = False
    # back to RGB via a Lab 1x1 image
    order = [i for i in np.argsort(-counts) if alive[i]]
    total = float(a.shape[0] * a.shape[1])
    for i in order:
        if counts[i] / total < min_share:
            continue
        lab_px = np.clip(centers[i], [0, 0, 0], [255, 255, 255]).reshape(1, 1, 3).astype(np.uint8)
        rgb_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2RGB)[0, 0]
        rgb_t = (int(rgb_px[0]), int(rgb_px[1]), int(rgb_px[2]))
        # skip anything that collapsed back to paper / black
        if max(rgb_t) - min(rgb_t) < 18 and min(rgb_t) > 200:
            continue
        if max(rgb_t) < 40:
            continue
        out[_name_rgb(rgb_t, used)] = rgb_t
        if len(out) >= max_colors:
            break
    return out


def classify_colors(rgb: np.ndarray, palette: dict, max_dist: float = 110, chunk: int = 500):
    """Label every pixel with its nearest palette colour.

    Pixels farther than ``max_dist`` (RGB Euclidean distance) from every palette
    colour count as paper (``white``). Returns ``(labels uint8, names)``.
    """
    names = require_palette(palette)
    ref = np.array([palette[n] for n in names], np.int32)
    out = np.empty(rgb.shape[:2], np.uint8)
    w = names.index("white")
    rgb = np.asarray(rgb)
    for r0 in range(0, rgb.shape[0], chunk):
        blk = rgb[r0:r0 + chunk, ..., :3].astype(np.int32)
        d = ((blk[:, :, None, :] - ref[None, None]) ** 2).sum(-1)
        k = d.argmin(-1).astype(np.uint8)
        if max_dist:
            k[np.take_along_axis(d, k[..., None].astype(np.intp), -1)[..., 0] > max_dist ** 2] = w
        out[r0:r0 + chunk] = k
    return out, names


def find_log_body(labels: np.ndarray, names: list) -> tuple[int, int]:
    """Rows of the log body = the longest unbroken vertical ink run (a track border)."""
    inked = labels != names.index("white")
    best, best_len = (0, labels.shape[0]), 0
    for x in range(labels.shape[1]):
        d = np.diff(np.r_[0, inked[:, x].astype(np.int8), 0])
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(a):
            k = int(np.argmax(b - a))
            if b[k] - a[k] > best_len:
                best_len, best = int(b[k] - a[k]), (int(a[k]), int(b[k]))
    return best


def legend_color(labels: np.ndarray, names: list, y0: float, y1: float, x_range,
                 below: int = 45, above: int = 8, min_cover: float = 0.2) -> str:
    """Colour of the legend line printed under a curve name (or ``black`` if none)."""
    x0, x1 = int(x_range[0]), int(x_range[1])
    sub = labels[max(0, int(y0) - above):int(y1) + below, x0:x1]
    best, best_c = "black", 0.0
    for ci, n in enumerate(names):
        if n in ("white", "black"):
            continue
        c = float((sub == ci).mean(1).max()) if sub.size else 0.0
        if c >= min_cover and c > best_c:
            best, best_c = n, c
    return best

