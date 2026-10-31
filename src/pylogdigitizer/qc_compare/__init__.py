# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""qc_compare: compare a digitized curve with a reference (e.g. paired KGS LAS)."""
from __future__ import annotations
import numpy as np


def to_grid(depth, values, grid):
    """Interpolate (depth, values) onto ``grid``; NaN outside the data range."""
    d = np.asarray(depth, float); v = np.asarray(values, float)
    ok = ~np.isnan(d) & ~np.isnan(v)
    d, v = d[ok], v[ok]
    o = np.argsort(d); d, v = d[o], v[o]
    return np.interp(grid, d, v, left=np.nan, right=np.nan)


def align_depth(ref: np.ndarray, test: np.ndarray, max_shift: int = 40) -> int:
    """Integer sample shift s maximizing corr(ref, shift(test, s)) on a common regular grid.
    Positive s: the digitized curve was s samples too deep (it is moved up)."""
    best, best_s = -2.0, 0
    for s in range(-max_shift, max_shift + 1):
        b = shift(test, s)
        ok = np.isfinite(ref) & np.isfinite(b)
        if ok.sum() < 20:
            continue
        r = np.corrcoef(ref[ok], b[ok])[0, 1]
        if r > best:
            best, best_s = r, s
    return best_s


def shift(values: np.ndarray, s: int) -> np.ndarray:
    """Shift a gridded series by s samples (positive -> moves values up), NaN padded."""
    out = np.full_like(values, np.nan, dtype=float)
    if s > 0:
        out[:-s] = values[s:]
    elif s < 0:
        out[-s:] = values[:s]
    else:
        out[:] = values
    return out


def metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    """r, RMSE, bias (test - ref), MAE, n over samples where both are finite."""
    ref = np.asarray(ref, float); test = np.asarray(test, float)
    ok = np.isfinite(ref) & np.isfinite(test)
    nan = dict(r=np.nan, rmse=np.nan, bias=np.nan, mae=np.nan, n=int(ok.sum()))
    if ok.sum() < 3:
        return dict(nan, reason=f"only {int(ok.sum())} overlapping samples (no depth overlap?)")
    a, b = ref[ok], test[ok]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(nan, n=int(ok.sum()), reason="constant reference or test curve (r undefined)")
    return dict(r=float(np.corrcoef(a, b)[0, 1]), rmse=float(np.sqrt(np.mean((b - a) ** 2))),
                bias=float(np.mean(b - a)), mae=float(np.mean(abs(b - a))), n=int(ok.sum()), reason="")


def compare(ref_depth, ref_values, test_depth, test_values, step: float = 0.5, max_shift_ft: float = 20.0,
            top: float | None = None, base: float | None = None) -> dict:
    """Full comparison: common grid, optional bulk depth shift by cross-correlation, metrics before
    and after alignment. Returns dict with grid, ref, test, test_aligned, shift_ft, raw, aligned."""
    t0 = max(np.nanmin(ref_depth), np.nanmin(test_depth)) if top is None else top
    t1 = min(np.nanmax(ref_depth), np.nanmax(test_depth)) if base is None else base
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < 10 * step:
        m = dict(r=np.nan, rmse=np.nan, bias=np.nan, mae=np.nan, n=0,
                 reason=f"no depth overlap: reference {np.nanmin(ref_depth):.0f}-{np.nanmax(ref_depth):.0f}, "
                        f"digitized {np.nanmin(test_depth):.0f}-{np.nanmax(test_depth):.0f}")
        e = np.array([])
        return dict(grid=e, ref=e, test=e, test_aligned=e, shift_ft=np.nan, raw=m, aligned=dict(m))
    grid = np.arange(np.ceil(t0 / step) * step, t1 + step / 2, step)
    r = to_grid(ref_depth, ref_values, grid); t = to_grid(test_depth, test_values, grid)
    s = align_depth(r, t, int(max_shift_ft / step)) if max_shift_ft else 0
    ta = shift(t, s)
    return dict(grid=grid, ref=r, test=t, test_aligned=ta, shift_ft=s * step,
                raw=metrics(r, t), aligned=metrics(r, ta))


def qc_plot(depth: np.ndarray, ref: np.ndarray, test: np.ndarray, out: str, title: str = "",
            labels: tuple[str, str] = ("reference LAS", "digitized"), xlim=None, extra: dict | None = None) -> str:
    """Overlay log plot (depth down) + crossplot. Saves PNG to ``out``."""
    from matplotlib.figure import Figure  # no pyplot: works headless, leaves the notebook backend alone
    fig = Figure(figsize=(8, 11)); ax, ax2 = fig.subplots(1, 2)
    ax.plot(ref, depth, "k", lw=0.6, label=labels[0])
    ax.plot(test, depth, "r", lw=0.6, alpha=0.8, label=labels[1])
    for k, v in (extra or {}).items():
        ax.plot(v, depth, lw=0.6, label=k)
    ax.invert_yaxis(); ax.set_ylabel("Depth (ft)"); ax.legend(loc="upper right", fontsize=8)
    if xlim:
        ax.set_xlim(*xlim)
    ok = np.isfinite(ref) & np.isfinite(test)
    ax2.plot(ref[ok], test[ok], ".", ms=1, alpha=0.3)
    lo = np.nanmin([ref[ok].min(), test[ok].min()]); hi = np.nanmax([ref[ok].max(), test[ok].max()])
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax2.set_xlabel(labels[0]); ax2.set_ylabel(labels[1])
    m = metrics(ref, test)
    ax2.set_title(f"r={m['r']:.3f} RMSE={m['rmse']:.2f} n={m['n']}")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110)
    return out
