# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""qc_compare: compare a digitized curve with a reference (e.g. paired KGS LAS)."""
from __future__ import annotations
import numpy as np


def to_grid(depth, values, grid):
    """Interpolate (depth, values) onto ``grid``; NaN outside the data range and inside null stretches
    (a grid depth whose neighbouring samples include a NaN stays NaN, so nulls are never scored as
    values interpolated across them)."""
    d = np.asarray(depth, float); v = np.asarray(values, float)
    keep = ~np.isnan(d)
    d, v = d[keep], v[keep]
    o = np.argsort(d, kind="stable"); d, v = d[o], v[o]
    grid = np.asarray(grid, float)
    ok = ~np.isnan(v)
    if not ok.any():
        return np.full(grid.shape, np.nan)
    out = np.interp(grid, d[ok], v[ok], left=np.nan, right=np.nan)
    if not ok.all():
        j = np.clip(np.searchsorted(d, grid), 1, len(d) - 1)
        exact = np.isclose(grid, d[j]) & ok[j] | np.isclose(grid, d[j - 1]) & ok[j - 1]
        out[~(ok[j - 1] & ok[j]) & ~exact] = np.nan
    return out


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
    tr, sh = register(r, ta, step)
    reg = metrics(r, tr); reg.update(value_fit(r, tr))
    reg["max_local_shift_ft"] = float(np.max(np.abs(sh))) if len(sh) else 0.0
    return dict(grid=grid, ref=r, test=t, test_aligned=ta, shift_ft=s * step, test_registered=tr,
                local_shift_ft=sh, raw=metrics(r, t), aligned=metrics(r, ta), registered=reg)


def register(ref: np.ndarray, test: np.ndarray, step: float = 0.5, window_ft: float = 60.0,
             max_local_ft: float = 5.0, min_r: float = 0.3, max_slope: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Depth registration on a common grid: local shifts (|s| <= ``max_local_ft``) from cross-
    correlation in half-overlapping windows, kept only where the window correlates (``min_r``),
    median-smoothed and limited to a stretch of ``max_slope`` ft/ft, so the warp stays monotone and
    cannot chase noise. Returns (test_registered, shift_ft per sample; positive = moved up)."""
    n = len(ref); w = max(20, int(window_ft / step)); ms = int(max_local_ft / step)
    cs, ss = [], []
    for a in range(0, max(1, n - w + 1), w // 2):
        r_, t_ = ref[a:a + w], test[a:a + w]
        best, bs = -2.0, None
        for k in range(-ms, ms + 1):
            b = shift(t_ if k == 0 else test, k)
            b = b if k == 0 else b[a:a + w]
            ok = np.isfinite(r_) & np.isfinite(b)
            if ok.sum() < w // 2 or np.ptp(r_[ok]) == 0 or np.ptp(b[ok]) == 0:
                continue
            c = np.corrcoef(r_[ok], b[ok])[0, 1]
            if c > best:
                best, bs = c, k
        if bs is not None and best >= min_r:
            cs.append(a + w / 2); ss.append(bs * step)
    if not cs:
        return test.copy(), np.zeros(n)
    ss = np.asarray(ss, float)
    if len(ss) >= 3:     # median of 3: one bad window cannot pull the warp
        ss = np.r_[ss[0], [np.median(ss[i - 1:i + 2]) for i in range(1, len(ss) - 1)], ss[-1]]
    sh = np.interp(np.arange(n), cs, ss)
    lim = max_slope * step     # limit the stretch rate (forward then backward pass)
    for i in range(1, n):
        sh[i] = np.clip(sh[i], sh[i - 1] - lim, sh[i - 1] + lim)
    for i in range(n - 2, -1, -1):
        sh[i] = np.clip(sh[i], sh[i + 1] - lim, sh[i + 1] + lim)
    idx = np.arange(n) + sh / step
    ok = np.isfinite(test)
    out = np.interp(idx, np.flatnonzero(ok), test[ok], left=np.nan, right=np.nan) if ok.sum() >= 2 else test.copy()
    out[~np.isfinite(np.interp(idx, np.arange(n), np.where(ok, 0.0, np.nan)))] = np.nan
    return out, sh


def value_fit(ref: np.ndarray, test: np.ndarray) -> dict:
    """Least-squares ref ~ gain * test + offset: a gain far from 1 or a large offset points at a
    wrong scale (ends, backup or log/linear) rather than a tracing error."""
    ok = np.isfinite(ref) & np.isfinite(test)
    if ok.sum() < 3 or np.ptp(test[ok]) == 0:
        return dict(gain=np.nan, offset=np.nan, rmse_fit=np.nan)
    g, o = np.polyfit(test[ok], ref[ok], 1)
    return dict(gain=float(g), offset=float(o), rmse_fit=float(np.sqrt(np.mean((g * test[ok] + o - ref[ok]) ** 2))))


def qc_plot(depth: np.ndarray, ref: np.ndarray, test: np.ndarray, out: str, title: str = "",
            labels: tuple[str, str] = ("reference LAS", "digitized"), xlim=None, extra: dict | None = None, dpi: int = 110,
            figsize: tuple[float, float] = (8, 11)) -> str:
    """Overlay log plot (depth down) + crossplot. Saves PNG to ``out`` at ``dpi`` (``figsize`` in inches)."""
    from matplotlib.figure import Figure  # no pyplot: works headless, leaves the notebook backend alone
    fig = Figure(figsize=figsize); ax, ax2 = fig.subplots(1, 2)
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
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=dpi)
    return out


def logs_figure(out: str, logs: dict, image: np.ndarray | None = None, title: str = "", height_in: float = 14.0,
                dpi: int = 110) -> str:
    """The scan (left, as printed) next to the digitized logs (right), height-aligned row for row.

    ``logs``: ``{tracks: [{name, left, right}], curves: [{name, color, track, row, frac}], ticks:
    [[depth, row, major]], width}`` in sheet pixels, as made by the app (``frac`` = position across the
    curve's track on its printed scale, null where off scale). Each curve is drawn in its own colour in
    its own track, with the 10 ft depth grid of the scan. Saves PNG to ``out``."""
    from matplotlib.figure import Figure
    cur = [c for c in logs["curves"] if len(c["row"])]
    if not cur:
        raise ValueError("no traced curves to plot")
    r0 = min(float(np.nanmin(c["row"])) for c in cur); r1 = max(float(np.nanmax(c["row"])) for c in cur)
    tr = {t["name"]: t for t in logs.get("tracks") or []}
    x0 = min([t["left"] for t in tr.values()] or [0.0]); x1 = max([t["right"] for t in tr.values()] or [float(logs.get("width", 1))])
    pw = max(2.5, height_in * (x1 - x0) / max(1.0, r1 - r0))
    n = 2 if image is not None else 1
    fig = Figure(figsize=(pw * n + 0.9, height_in + 0.9))
    axs = fig.subplots(1, n, sharey=True, squeeze=False)[0]
    if image is not None:
        a = axs[0]; im = image[int(r0):int(np.ceil(r1)) + 1, int(x0):int(np.ceil(x1)) + 1]
        a.imshow(im, cmap="gray" if im.ndim == 2 else None, extent=(x0, x1, r1, r0), aspect="auto", interpolation="lanczos")
        a.set_title("scan", fontsize=10); a.set_xticks([])
    ax = axs[-1]
    for d, r, major in logs.get("ticks") or []:
        ax.axhline(r, color="#888" if major else "#ddd", lw=0.8 if major else 0.5, zorder=0)
    for t in tr.values():
        ax.axvline(t["left"], color="#555", lw=0.8); ax.axvline(t["right"], color="#555", lw=0.8)
    for c in cur:
        t = tr.get(c.get("track")) or dict(left=x0, right=x1)
        f = np.asarray(c["frac"], float); f = np.where((f >= -0.01) & (f <= 1.01), f, np.nan)
        ax.plot(t["left"] + f * (t["right"] - t["left"]), c["row"], color=c["color"], lw=0.9, label=c["name"])
    ax.set_xlim(x0, x1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_title("digitized logs", fontsize=10)
    tk = [q for q in logs.get("ticks") or [] if q[2]]
    axs[0].set_yticks([q[1] for q in tk]); axs[0].set_yticklabels([f"{q[0]:g}" for q in tk]); axs[0].set_ylabel("Depth (ft)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=min(5, len(cur)), fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=dpi)
    return out
