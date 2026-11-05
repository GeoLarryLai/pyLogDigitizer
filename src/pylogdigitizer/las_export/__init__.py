# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""las_export: Write digitized curves to LAS 2.0 (lasio) and read reference LAS for QC."""

from __future__ import annotations
from pathlib import Path
import numpy as np

def read_las(path: str | Path):
    """Read a LAS file with lasio; returns lasio.LASFile."""
    import lasio
    return lasio.read(str(path))

def curves_summary(path: str | Path) -> dict[str, dict]:
    """Per-curve unit and first/last valid depth."""
    las = read_las(path)
    df = las.df()
    out = {}
    for c in las.curves[1:]:
        s = df[c.mnemonic].dropna()
        out[c.mnemonic] = {"unit": c.unit, "top": float(s.index.min()) if len(s) else None,
                           "base": float(s.index.max()) if len(s) else None}
    return out

LAS_STEPS = (0.5, 0.25)


def resample(depth: np.ndarray, values: np.ndarray, step: float = 0.5, max_gap: float | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """Resample a traced line onto a regular depth grid (default 0.5 ft, option 0.25 ft) by linear
    interpolation along the line only: no smoothing or filtering. A grid depth gets NaN (written as
    the LAS null) when either neighbouring trace sample is NaN (gap / clipped stretch) or the two
    neighbours are more than ``max_gap`` apart (default 4 steps)."""
    if not any(abs(step - s_) < 1e-9 for s_ in LAS_STEPS):
        raise ValueError(f"LAS step must be one of {LAS_STEPS} ft, got {step}")
    order = np.argsort(depth)
    d, v = np.asarray(depth, float)[order], np.asarray(values, float)[order]
    grid = np.arange(np.ceil(d[0] / step) * step, d[-1] + step / 2, step)
    j = np.clip(np.searchsorted(d, grid), 1, len(d) - 1)
    d0, d1, v0, v1 = d[j - 1], d[j], v[j - 1], v[j]
    t = np.where(d1 > d0, (grid - d0) / np.where(d1 > d0, d1 - d0, 1), 0.0)
    out = v0 + t * (v1 - v0)
    exact = np.isclose(grid, d1)
    out = np.where(exact, v1, out)
    mg = 4 * step if max_gap is None else max_gap
    bad = (~np.isfinite(v0) | ~np.isfinite(v1) | (d1 - d0 > mg)) & ~(exact & np.isfinite(v1))
    out[bad] = np.nan
    return grid, out

class LASExportError(ValueError):
    """The data cannot be written as a valid LAS file (message says why, in plain words)."""


def check_las_data(depth, curves: dict, units: dict | None = None, depth_unit: str = "FT",
                   null: float = -999.25, max_depth_ft: float = 40000.0) -> list[str]:
    """Checks before writing. Raises LASExportError for data that would make a broken LAS file
    (empty, depth not increasing, missing depths, length mismatch); returns a list of plain-language
    warnings for things worth a look (uneven depth step, missing units, odd depth range, the null
    value appearing as real data)."""
    d = np.asarray(depth, float)
    warn: list[str] = []
    if d.size < 2:
        raise LASExportError("Nothing to save: the curve has fewer than 2 depth points.")
    if not np.all(np.isfinite(d)):
        raise LASExportError("Some depths are missing (NaN); the LAS depth column must be complete.")
    st = np.diff(d)
    if np.any(st <= 0):
        raise LASExportError("Depths must increase from top to bottom; found depths going back up or repeated.")
    for name, a in curves.items():
        if len(a) != d.size:
            raise LASExportError(f"Curve {name} has {len(a)} values but there are {d.size} depths.")
        a = np.asarray(a, float)
        if np.any(np.isclose(a[np.isfinite(a)], null)):
            warn.append(f"Curve {name} contains the null value {null} as real data; those points will read as missing.")
        if not np.isfinite(a).any():
            warn.append(f"Curve {name} has no values at all.")
    if np.ptp(st) > 1e-6 * max(1.0, abs(st[0])):
        warn.append(f"The depth step is not constant ({st.min():g} to {st.max():g}); many LAS readers expect a fixed step.")
    if not depth_unit:
        warn.append("The depth unit is missing (use FT or M).")
    miss = [n for n in curves if not (units or {}).get(n)]
    if miss:
        warn.append("No unit given for: " + ", ".join(miss) + ".")
    scale = 1.0 if str(depth_unit).upper().startswith("F") else 3.28084
    lo, hi = d[0] * scale, d[-1] * scale
    if lo < -1000 or hi > max_depth_ft:
        warn.append(f"The depth range {d[0]:g}-{d[-1]:g} {depth_unit} looks odd for a well log; check the depth numbers.")
    return warn


def verify_las(path: str | Path, depth, curves: dict, null: float = -999.25, tol: float = 1e-3) -> list[str]:
    """Read a written LAS file back and compare it with what was meant to be written.
    Returns a list of problems (empty = the file reads back correctly)."""
    las = read_las(path)
    prob = []
    d = np.asarray(depth, float)
    dd = np.asarray(las.index, float)
    if dd.size != d.size or not np.allclose(dd, d, atol=tol):
        prob.append("depths read back differently")
    try:
        if abs(float(las.well["STRT"].value) - d[0]) > tol or abs(float(las.well["STOP"].value) - d[-1]) > tol:
            prob.append("STRT/STOP in the header do not match the data")
        if abs(float(las.well["NULL"].value) - null) > tol:
            prob.append("NULL value in the header is wrong")
    except (KeyError, TypeError, ValueError):
        prob.append("header lacks STRT/STOP/NULL")
    for name, a in curves.items():
        if name not in las.keys():
            prob.append(f"curve {name} missing from the file"); continue
        b, a = np.asarray(las[name], float), np.asarray(a, float)
        ok = np.isfinite(a)
        if b.size != a.size or not np.array_equal(ok, np.isfinite(b)) or not np.allclose(a[ok], b[ok], atol=tol, rtol=1e-6):
            prob.append(f"curve {name} values read back differently")
    return prob


def write_las(path: str | Path, depth: np.ndarray, curves: dict[str, np.ndarray],
              units: dict[str, str] | None = None, well: dict[str, str] | None = None,
              depth_unit: str = "FT", null: float = -999.25, check: bool = True) -> Path:
    """Write curves (same length as depth) to LAS 2.0. With ``check`` (default) the data are
    checked first (:func:`check_las_data`: hard problems raise LASExportError, the rest are
    issued as warnings) and the file is read back after writing (:func:`verify_las`)."""
    import lasio, warnings
    if check:
        for w in check_las_data(depth, curves, units, depth_unit, null):
            warnings.warn(w, UserWarning, stacklevel=2)
    units = units or {}
    las = lasio.LASFile()
    las.well["NULL"].value = null
    for k, v in (well or {}).items():
        las.well[k] = lasio.HeaderItem(k, value=v)
    las.append_curve("DEPT", np.asarray(depth, float), unit=depth_unit)
    for name, arr in curves.items():
        a = np.where(np.isnan(arr), null, arr)
        las.append_curve(name, a, unit=units.get(name, ""), descr="digitized by pyLogDigitizer")
    las.write(str(path), version=2.0)
    if check:
        prob = verify_las(path, depth, curves, null)
        if prob:
            raise LASExportError(f"{path}: written file did not read back correctly: " + "; ".join(prob))
    return Path(path)
