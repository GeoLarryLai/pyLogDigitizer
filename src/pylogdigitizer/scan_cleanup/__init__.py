# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scan_cleanup: binarize, deskew (projection-profile search), tiling, grid-line masks.

Conventions: ``ink`` arrays are boolean, True = ink (dark), shape (rows, cols).
"""
from __future__ import annotations
from typing import Iterator
import numpy as np


def binarize(image: np.ndarray, method: str = "otsu", threshold: int | None = None) -> np.ndarray:
    """Grayscale/bilevel image -> boolean ink mask (True = dark).

    method: "otsu" (global), "fixed" (``threshold``, default 128) or "sauvola" (local).
    Bilevel (bool) input is returned as ``~image`` when it is a PIL '1' array (True = white).
    """
    if image.dtype == bool:
        return ~image
    img = image.astype(np.uint8)
    if method == "fixed":
        return img < (128 if threshold is None else threshold)
    if method == "otsu":
        import cv2
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img <= t  # OpenCV puts values <= t in the dark class
    if method == "sauvola":
        return img < threshold_sauvola(img, window_size=31)
    raise ValueError(method)


def threshold_sauvola(image: np.ndarray, window_size: int = 15, k: float = 0.2, r: float | None = None) -> np.ndarray:
    """Sauvola & Pietikainen (2000) local threshold T = m * (1 + k * (s / r - 1)), m / s = local mean /
    std over a ``window_size`` square (reflected borders), r = half the image's value range by
    default. Written from the paper's formula (box filters), matching the usual implementation."""
    import cv2
    img = np.asarray(image, np.float64)
    w = (int(window_size), int(window_size))
    m = cv2.boxFilter(img, cv2.CV_64F, w, borderType=cv2.BORDER_REFLECT_101)
    m2 = cv2.boxFilter(img * img, cv2.CV_64F, w, borderType=cv2.BORDER_REFLECT_101)
    sd = np.sqrt(np.clip(m2 - m * m, 0, None))
    if r is None:
        r = 0.5 * (img.max() - img.min())
    return m * (1 + k * ((sd / r) - 1))


def estimate_skew(ink: np.ndarray, max_angle: float = 2.0, step: float = 0.05, sample_rows: int = 4000) -> float:
    """Skew angle in degrees (positive = rows rise to the right) by maximizing the variance of the
    horizontal projection profile over sheared versions of a sample window (cheap Hough-like search).
    """
    h = ink.shape[0]
    y0 = max(0, h // 2 - sample_rows // 2)
    win = ink[y0:y0 + sample_rows]
    ys, xs = np.nonzero(win)
    if len(ys) == 0:
        return 0.0
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_angle, max_angle + step / 2, step):
        yy = np.round(ys + xs * np.tan(np.deg2rad(a))).astype(np.int64)
        yy -= yy.min()
        prof = np.bincount(yy)
        v = float((prof.astype(np.float64) ** 2).sum())
        if v > best:
            best, best_a = v, float(a)
    return best_a


def deskew(image: np.ndarray, angle: float | None = None, max_angle: float = 2.0,
           sample: tuple[slice, slice] | None = None) -> tuple[np.ndarray, float]:
    """Remove skew so horizontal grid lines become horizontal. Returns (image, angle_used).

    Uses a vertical shear (each column shifted by round(x * tan(angle))), which equals a rotation
    for the small angles of scanned logs, is exact for bilevel data and has no size limit (OpenCV
    warps fail above 32767 px, typical log scans are 100k+ rows). ``sample`` = (rows, cols) slices
    used for angle estimation (default: whole image)."""
    is_bool = image.dtype == bool
    if angle is None:
        win = image if sample is None else image[sample]
        angle = estimate_skew(win if is_bool else binarize(win), max_angle=max_angle)
    if abs(angle) < 0.02:
        return image, 0.0
    h, w = image.shape
    t = np.tan(np.deg2rad(angle))
    shifts = -np.round(np.arange(w) * t).astype(int)
    shifts -= int(round(np.median(shifts)))
    out = np.zeros_like(image) if is_bool else np.full_like(image, 255)
    for x in range(w):
        s = shifts[x]
        if s > 0:
            out[:h - s, x] = image[s:, x]
        elif s < 0:
            out[-s:, x] = image[:h + s, x]
        else:
            out[:, x] = image[:, x]
    return out, float(angle)


def iter_tiles(shape: tuple[int, int], tile: int = 2048, overlap: int = 128) -> Iterator[tuple[slice, slice]]:
    """Yield (row_slice, col_slice) covering an image with overlapping tiles."""
    h, w = shape
    step = max(1, tile - overlap)
    for y in range(0, max(h - overlap, 1), step):
        for x in range(0, max(w - overlap, 1), step):
            yield slice(y, min(y + tile, h)), slice(x, min(x + tile, w))


def line_masks(ink: np.ndarray, min_len: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """Masks of straight horizontal and vertical runs >= ``min_len`` px (grid lines)."""
    import cv2
    u = ink.astype(np.uint8)
    hm = cv2.morphologyEx(u, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))).astype(bool)
    vm = cv2.morphologyEx(u, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))).astype(bool)
    return hm, vm


def remove_grid(binary: np.ndarray, min_len: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """Return (ink_without_lines, line_mask)."""
    hm, vm = line_masks(binary, min_len)
    m = hm | vm
    return binary & ~m, m


def thick_strokes(ink: np.ndarray, min_width: int = 3) -> np.ndarray:
    """Keep strokes at least ``min_width`` px thick (erode then dilate). Drops thin grid lines."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, min_width))
    return cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)


def split_sections(label_rows: np.ndarray, spacing_tol: float = 0.05) -> list[tuple[int, int]]:
    """Split a sequence of depth-label rows into runs with consistent spacing (one run per log
    section / scale). Returns (start_index, end_index_exclusive) into ``label_rows``.
    A gap that is ~k times the median spacing (missing labels) does not split a run."""
    r = np.asarray(label_rows, float)
    if len(r) < 3:
        return [(0, len(r))]
    d = np.diff(r)
    runs, s = [], 0
    for i, g in enumerate(d):
        ref = np.median(d[max(s, i - 5):i]) if i - s >= 2 else None
        if ref is None:
            if i - s == 1 and abs(g - d[i - 1]) > spacing_tol * d[i - 1]:
                s = i
            continue
        k = round(g / ref)
        if k < 1 or abs(g - k * ref) > spacing_tol * ref * k or k > 4:
            runs.append((s, i + 1)); s = i + 1
    runs.append((s, len(r)))
    return [x for x in runs if x[1] - x[0] >= 3]



try:
    from pylogdigitizer import curve_tracing_rs as _rs
    if not hasattr(_rs, "deskew_blocks"):     # older build without the deskew kernel
        _rs = None
except ImportError:
    _rs = None

numba, HAVE_NUMBA = None, False
if _rs is None:     # optional fallback only (numba is not a dependency)
    try:
        import numba
        HAVE_NUMBA = True
    except ImportError:
        pass


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _deskew_blocks_nb(ink: np.ndarray, block: int, c0: int, c1: int,
                          angles: np.ndarray, smooth: int) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - compiled
        h, w = ink.shape
        n_blocks = (h + block - 1) // block
        n_ang = len(angles)
        raw_ang = np.zeros(n_blocks, dtype=np.float64)
        max_s = int(np.ceil(w * np.tan(np.deg2rad(np.max(np.abs(angles)))))) + 2
        for b in numba.prange(n_blocks):
            s = b * block
            e = min(h, s + block)
            bh = e - s
            n = 0
            for y in range(s, e):
                for x in range(c0, c1):
                    if ink[y, x]:
                        n += 1
            if n == 0:
                raw_ang[b] = 0.0
                continue
            ys = np.empty(n, dtype=np.int32)
            xs = np.empty(n, dtype=np.int32)
            idx = 0
            for y in range(bh):
                row = ink[s + y]
                for x in range(c0, c1):
                    if row[x]:
                        ys[idx] = y
                        xs[idx] = x - c0
                        idx += 1
            best = -1.0
            best_a = 0.0
            prof = np.zeros(bh + 2 * max_s + 2, dtype=np.int64)
            shifts = np.empty(c1 - c0, dtype=np.int32)
            for ai in range(n_ang):
                a = angles[ai]
                t = np.tan(np.deg2rad(a))
                for x in range(c1 - c0):
                    shifts[x] = int(np.rint(x * t)) + max_s
                prof[:] = 0
                for i in range(n):
                    prof[ys[i] + shifts[xs[i]]] += 1
                v = 0.0
                for k in range(len(prof)):
                    pk = float(prof[k])
                    v += pk * pk
                if v > best:
                    best = v
                    best_a = a
            raw_ang[b] = best_a
        ang = np.empty(n_blocks, dtype=np.float64)
        if smooth and n_blocks > 2:
            for i in range(n_blocks):
                lo = max(0, i - smooth)
                hi = min(n_blocks, i + smooth + 1)
                ang[i] = np.median(raw_ang[lo:hi])
        else:
            ang[:] = raw_ang
        out = np.zeros_like(ink)
        xc = (c0 + c1) / 2.0
        for b in numba.prange(n_blocks):
            s = b * block
            e = min(h, s + block)
            bh = e - s
            t = np.tan(np.deg2rad(ang[b]))
            b_shifts = np.empty(w, dtype=np.int32)
            for x in range(w):
                b_shifts[x] = -int(np.rint((x - xc) * t))
            for y in range(bh):
                dst_y = s + y
                for x in range(w):
                    src_y = dst_y + b_shifts[x]
                    if 0 <= src_y < h:
                        out[dst_y, x] = ink[src_y, x]
        return out, ang


def deskew_blocks(ink: np.ndarray, block: int = 2000, sample_cols: slice | None = None,
                  max_angle: float = 1.0, step: float = 0.05, smooth: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise deskew for long scans whose skew drifts along the sheet (fan-fold paper feed).
    The skew is estimated per ``block`` rows (on ``sample_cols``), median-smoothed over
    ``smooth`` neighbours, and removed by a vertical shear about the sample centre.
    Returns (deskewed ink, per-block angles in degrees)."""
    h, w = ink.shape
    if _rs is not None and ink.dtype == bool and h > 0 and w > 0:      # Rust (parallel, no numba needed)
        c0, c1, _ = (sample_cols or slice(0, w)).indices(w)
        angles = np.arange(-max_angle, max_angle + step / 2, step, dtype=np.float64)
        out, ang = _rs.deskew_blocks(np.ascontiguousarray(ink).view(np.uint8), int(block), int(c0), int(c1),
                                     angles, int(smooth))
        return out.view(bool), ang
    if HAVE_NUMBA and ink.dtype == bool and h > 0 and w > 0:
        c0, c1, _ = (sample_cols or slice(0, w)).indices(w)
        angles = np.arange(-max_angle, max_angle + step / 2, step, dtype=np.float64)
        return _deskew_blocks_nb(np.ascontiguousarray(ink), int(block), int(c0), int(c1), angles, int(smooth))
    sc = sample_cols or slice(0, w)
    starts = list(range(0, h, block))
    ang = np.array([estimate_skew(ink[s:s + block, sc], max_angle, step, block) for s in starts])
    if smooth and len(ang) > 2:
        ang = np.array([np.median(ang[max(0, i - smooth):i + smooth + 1]) for i in range(len(ang))])
    out = np.zeros_like(ink)
    xc = ((sc.start or 0) + (sc.stop or w)) / 2
    xs = np.arange(w)
    for s, a in zip(starts, ang):
        e = min(h, s + block)
        shifts = -np.round((xs - xc) * np.tan(np.deg2rad(a))).astype(int)
        for x in range(w):
            k = shifts[x]
            src0, src1 = s + k, e + k
            d0 = max(0, -src0); d1 = (e - s) - max(0, src1 - h)
            if d1 > d0:
                out[s + d0:s + d1, x] = ink[src0 + d0:src0 + d1, x]
    return out, ang
