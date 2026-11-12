# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""curve_ink: build a black-on-white working image for ONE colour curve.

The tracer follows whatever dark ink it is given. On a colour scan that means
isolating one curve's palette colour, closing dash gaps and thickening the
stroke so it beats thin grid lines, then combining it with the black grid and
depth numbers so track / depth detection still works.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pylogdigitizer.track_grid_detection import _runs

__all__ = ["save_ink_tif", "make_base_ink", "make_curve_ink", "black_curve_only",
           "thin_vertical_lines"]

Image.MAX_IMAGE_PIXELS = None


def save_ink_tif(ink: np.ndarray, path) -> Path:
    """Write a True=ink mask as a black-on-white 1-bit TIFF (what the tracer reads)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(~np.asarray(ink, bool)).save(str(path), compression="group4")
    return path


def _grey_names(names: list) -> list:
    return [n for n in names if str(n).lower().startswith(("gray", "grey"))]


def make_base_ink(labels: np.ndarray, names: list, body_rows=None, blank_below_body: bool = True,
                  extra_borders_px=(), border_width: int = 7, grey_grid: bool = True,
                  grid_run_px: int = 61) -> np.ndarray:
    """Black ink (grid, depth numbers, header text) plus a grey printed grid.

    ``grey_grid``: digital plots often draw the grid and the track borders in mid-grey
    (palette names starting with ``gray`` / ``grey``). Their straight runs (vertical or
    horizontal, >= ``grid_run_px`` px) are added so track and depth detection still see
    the grid on every working image; grey *curves* (dashed, not straight) are left out.
    Optionally blank the footer below the log body and draw extra thick track
    borders (helps the detector split tracks that share a border).
    """
    if "black" not in names or "white" not in names:
        raise ValueError("names must include 'white' and 'black'")
    ink = labels == names.index("black")
    if grey_grid:
        for g in _grey_names(names):
            ink |= _line_mask(labels == names.index(g), grid_run_px)
    r0, r1 = body_rows or (0, ink.shape[0])
    if blank_below_body and body_rows:
        ink[r1:] = False
    h = border_width // 2
    for x in extra_borders_px or ():
        ink[r0:r1, max(0, int(x) - h):int(x) + h + 1] = True
    return ink


def _line_mask(m: np.ndarray, n: int = 61):
    import cv2
    u8 = m.astype(np.uint8)
    hl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1))).astype(bool)
    vl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n))).astype(bool)
    return hl | vl


def _grow1d(mask: np.ndarray, n: int = 1) -> np.ndarray:
    return np.convolve(np.asarray(mask, float), np.ones(2 * n + 1), "same") > 0


def black_curve_only(lab: np.ndarray, names: list) -> np.ndarray:
    """Black ink of a track window with the grid taken out.

    Grid lines are found three ways: long straight runs (on all ink, so a line is
    not cut where a colour curve crosses it), columns that are black over
    >= 25 % of the log (dashed / broken grid lines) and full-width rows.
    """
    black = lab == names.index("black")
    inked = lab != names.index("white")
    cols = _grow1d(black.mean(axis=0) >= 0.25)
    rows = _grow1d(inked.mean(axis=1) >= 0.4)
    return black & ~_line_mask(inked) & ~cols[None, :] & ~rows[:, None]


def thin_vertical_lines(block: np.ndarray, keep_px: int = 2, frac: float = 0.5) -> np.ndarray:
    """Vertical grid lines inside a track, drawn at most ``keep_px`` wide.

    Heavy lines (and two lines printed side by side) would otherwise survive the
    tracer's thin-line filter (``min_width``) and pull the trace onto a straight
    line. The lines stay visible for track detection.
    """
    import cv2
    cf = block.mean(axis=0)
    on = cv2.morphologyEx((cf >= frac).astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]
    out = block.copy()
    for a, b in _runs(on.astype(bool)):
        if b - a <= keep_px:
            continue
        while a > 0 and cf[a - 1] >= frac / 4:
            a -= 1
        while b < len(cf) and cf[b] >= frac / 4:
            b += 1
        c = (a + b - keep_px) // 2
        keep = out[:, c:c + keep_px].copy()
        out[:, a:b] = False
        out[:, c:c + keep_px] = keep
    return out


def make_curve_ink(base: np.ndarray, labels: np.ndarray, names: list, color: str, x_range,
                   body_rows=None, gap_close_px: int = 9, thicken_px: int = 3, margin: int = 4,
                   erase_path=None, erase_px: int = 6, thin_grid_px: int = 2) -> np.ndarray:
    """Working image for ONE curve: black grid + this curve's colour inside its track.

    Dash gaps are closed (``gap_close_px``) and the line is thickened by
    ``thicken_px`` on each side so the tracer sees one clear continuous line.
    ``color='black'``: black ink with the grid taken out is used as the curve.
    ``erase_path`` = (sheet_rows, sheet_x, side): a traced companion curve to erase from this image.
    ``thin_grid_px``: vertical grid lines inside the track are drawn at most this
    wide (0 = as scanned).
    """
    import cv2
    if color not in names:
        raise ValueError(f"colour {color!r} is not in palette {names}")
    x0, x1 = int(round(x_range[0])) + margin, int(round(x_range[1])) - margin
    r0, r1 = body_rows or (0, base.shape[0])
    lab = labels[r0:r1, x0:x1]
    sub = black_curve_only(lab, names) if color == "black" else lab == names.index(color)
    if erase_path is not None:
        pr, px = np.asarray(erase_path[0], float), np.asarray(erase_path[1], float)
        side = erase_path[2] if len(erase_path) > 2 else None
        xo = np.interp(np.arange(r0, r1), pr, px, left=np.nan, right=np.nan)[:, None]
        cols = np.arange(x0, x1)[None, :]
        with np.errstate(invalid="ignore"):
            gone = (cols <= xo + erase_px) if side == "left" else (cols >= xo - erase_px) if side == "right" \
                else (np.abs(cols - xo) <= erase_px)
        sub &= ~gone
    u8 = sub.astype(np.uint8)
    if gap_close_px and gap_close_px > 1:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_close_px, gap_close_px)))
    if thicken_px and thicken_px > 0:
        k = 2 * int(thicken_px) + 1
        u8 = cv2.dilate(u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = np.asarray(base, bool).copy()
    if thin_grid_px:
        out[r0:r1, x0:x1] = thin_vertical_lines(out[r0:r1, x0:x1], thin_grid_px)
    out[r0:r1, x0:x1] |= u8.astype(bool)
    return out
