# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scale_calibration: pixel column <-> curve value for linear and logarithmic tracks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ScaleMap:
    """Map x pixel -> value. ``kind`` 'linear' or 'log'. v_left may exceed v_right (reversed
    scales, e.g. DT 140 -> 40, NPHI 0.45 -> -0.15)."""
    x_left: float
    x_right: float
    v_left: float
    v_right: float
    kind: str = "linear"

    def __post_init__(self):
        if self.kind == "log" and (self.v_left <= 0 or self.v_right <= 0):
            raise ValueError("log scale needs positive end values")

    def _f(self, x):
        return (np.asarray(x, float) - self.x_left) / (self.x_right - self.x_left)

    def __call__(self, x):
        f = self._f(x)
        if self.kind == "linear":
            return self.v_left + f * (self.v_right - self.v_left)
        return 10 ** (np.log10(self.v_left) + f * (np.log10(self.v_right) - np.log10(self.v_left)))

    def inverse(self, v):
        v = np.asarray(v, float)
        if self.kind == "linear":
            f = (v - self.v_left) / (self.v_right - self.v_left)
        else:
            f = (np.log10(v) - np.log10(self.v_left)) / (np.log10(self.v_right) - np.log10(self.v_left))
        return self.x_left + f * (self.x_right - self.x_left)


def linear_scale(x_left: float, x_right: float, v_left: float, v_right: float) -> ScaleMap:
    return ScaleMap(x_left, x_right, v_left, v_right, "linear")


def log_scale(x_left: float, x_right: float, v_left: float, v_right: float) -> ScaleMap:
    return ScaleMap(x_left, x_right, v_left, v_right, "log")


def from_track(track: dict, v_left: float, v_right: float, kind: str | None = None) -> ScaleMap:
    """Scale from a detected track (uses the outer grid lines as the scale ends)."""
    gx = track["grid_x"]
    if kind is None:
        from pylogdigitizer.track_grid_detection import classify_grid
        kind = classify_grid(track)
        kind = "linear" if kind == "unknown" else kind
    return ScaleMap(gx[0], gx[-1], v_left, v_right, kind)
