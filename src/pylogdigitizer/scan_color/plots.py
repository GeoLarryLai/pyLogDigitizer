# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Compact colour-scan figures: auto-detection overlay, and isolated ink vs trace."""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_color_autodetect", "plot_color_trace_pairs"]


def _rgb01(palette: dict, name: str) -> tuple[float, float, float]:
    c = palette[name]
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def _as_image(im) -> np.ndarray:
    if isinstance(im, (str, Path)):
        from PIL import Image
        return np.asarray(Image.open(im))
    return np.asarray(im)


def _window_rows(depth_map, depth_window, n_rows: int) -> tuple[int, int, float, float]:
    top, base = float(min(depth_window)), float(max(depth_window))
    r0 = int(round(depth_map.inverse(top)))
    r1 = int(round(depth_map.inverse(base)))
    if r1 < r0:
        r0, r1 = r1, r0
    r0 = max(0, r0)
    r1 = min(n_rows, max(r0 + 1, r1))
    d0, d1 = float(depth_map(r0)), float(depth_map(r1 - 1 if r1 > r0 else r0))
    if d1 < d0:
        d0, d1 = d1, d0
    return r0, r1, d0, d1


def _edge_x(layout, name: str, rows):
    k = layout.edge_names.index(name)
    return np.interp(np.asarray(rows, float), layout.centers, layout.edges[:, k])


def _sheet_xy(layout, xs, ys):
    from pylogdigitizer.qc_overlay import strip_to_sheet
    return strip_to_sheet(layout, np.asarray(xs, float), np.asarray(ys, float))


def plot_color_autodetect(rgb, depth_map, depth_window: tuple[float, float],
                          palette: dict, curves: list[dict], traces: dict,
                          track_xlim: dict, layout=None, ocr_notes: list[dict] | None = None,
                          title: str = "", figsize: tuple[float, float] | None = None):
    """Wide overlay of what the colour path auto-identified, plus the raw crop.

    Left: the colour raster over ``depth_window`` with detected track borders,
    snapped grid lines (cyan), and traced curves in their print colours.
    OCR'd curve names and scales are annotated on each track (e.g.
    ``GR 0–150 (OCR)``). Right: the same crop with no overlay. A compact
    legend sits under the pair. No reference LAS and no separate value track.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rgb = np.asarray(rgb)
    r0, r1, d0, d1 = _window_rows(depth_map, depth_window, rgb.shape[0])
    crop = rgb[r0:r1]
    extent = (0, rgb.shape[1], d1, d0)
    if figsize is None:
        figsize = (10.4, 4.45)
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw={"width_ratios": [1.42, 1.0], "wspace": 0.04},
        layout="constrained",
    )
    for a in (ax, axr):
        a.imshow(crop, origin="upper", aspect="auto", interpolation="nearest", extent=extent)
        a.set_xlim(0, rgb.shape[1])
        a.set_ylim(d1, d0)
        a.tick_params(labelsize=7)
        a.set_xticks([])
    ax.set_ylabel("depth (ft)", fontsize=8)
    axr.tick_params(labelleft=False)
    ax.set_title("auto-detected", fontsize=9, pad=4)
    axr.set_title("raw crop", fontsize=9, pad=4)

    yy = np.linspace(r0, r1, max(40, (r1 - r0) // 8))
    if layout is not None:
        try:
            from pylogdigitizer.qc_overlay import refine_grid
            for nm in getattr(layout, "track_names", []) or list(track_xlim):
                if (r1 - r0) < 80:
                    break
                G = refine_grid(layout, nm, (int(r0), int(r1)))
                for ys, xs in G.get("v") or []:
                    X, Y = _sheet_xy(layout, xs, ys)
                    ax.plot(X, [depth_map(y) for y in Y], color="#00c2e0", lw=0.55,
                            alpha=0.85, solid_capstyle="butt", zorder=2)
                for xs, ys in G.get("h") or []:
                    X, Y = _sheet_xy(layout, xs, ys)
                    ax.plot(X, [depth_map(y) for y in Y], color="#00c2e0", lw=0.45,
                            alpha=0.7, solid_capstyle="butt", zorder=2)
        except Exception:
            pass
        for nm in getattr(layout, "track_names", []) or []:
            for side in ("left", "right"):
                key = f"{nm}.{side}"
                if key not in layout.edge_names:
                    continue
                xs = _edge_x(layout, key, yy)
                X, Y = _sheet_xy(layout, xs, yy)
                ax.plot(X, [depth_map(y) for y in Y], color="#1a1a1a", lw=0.9,
                        ls=(0, (3, 2)), alpha=0.9, zorder=3)
        labels = getattr(layout, "labels", None) or []
        for lab in labels:
            row = float(lab["row"] if isinstance(lab, dict) else lab[0])
            if not (r0 <= row < r1):
                continue
            d = float(depth_map(row))
            ax.axhline(d, color="0.35", lw=0.4, alpha=0.55, zorder=2)
    else:
        for x0, x1 in track_xlim.values():
            ax.axvline(float(x0), color="#1a1a1a", lw=0.9, ls=(0, (3, 2)), zorder=3)
            ax.axvline(float(x1), color="#1a1a1a", lw=0.9, ls=(0, (3, 2)), zorder=3)

    styles = ["-", "--", ":", "-."]
    by_track: dict[str, int] = {}
    for cv in curves:
        d, x = traces[cv["name"]]
        d, x = np.asarray(d, float), np.asarray(x, float)
        m = np.isfinite(d) & np.isfinite(x) & (d >= d0) & (d <= d1)
        if not m.any():
            continue
        k = by_track.get(cv["track"], 0)
        by_track[cv["track"]] = k + 1
        ax.plot(x[m], d[m], color=_rgb01(palette, cv["color"]),
                lw=1.15 if k == 0 else 0.9, ls=styles[k % len(styles)],
                solid_capstyle="butt", zorder=4, label=cv["name"])

    notes = list(ocr_notes or [])
    if not notes:
        for cv in curves:
            notes.append(dict(
                track=cv["track"],
                label=f"{cv['name']} {_scale_text(cv['left'], cv['right'])}",
            ))
    grouped: dict[str, list[str]] = {}
    for note in notes:
        grouped.setdefault(note["track"], []).append(note["label"])
    for track, texts in grouped.items():
        if track in track_xlim:
            x0, x1 = (float(v) for v in track_xlim[track])
        elif layout is not None and f"{track}.left" in getattr(layout, "edge_names", []):
            mid = 0.5 * (r0 + r1)
            x0 = float(_edge_x(layout, f"{track}.left", [mid])[0])
            x1 = float(_edge_x(layout, f"{track}.right", [mid])[0])
        else:
            continue
        text = "\n".join(texts)            # one OCR'd curve per line, so the box stays inside its track
        xc = 0.5 * (x0 + x1)
        ypos = d0 + 0.02 * (d1 - d0)
        ax.text(xc, ypos, text, ha="center", va="top", fontsize=6.5, color="#111", linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", alpha=0.88),
                zorder=5)

    handles = [
        Line2D([0], [0], color="#00c2e0", lw=1.2, label="grid (detected)"),
        Line2D([0], [0], color="#1a1a1a", lw=1.0, ls=(0, (3, 2)), label="track border"),
    ]
    for cv in curves:
        handles.append(Line2D([0], [0], color=_rgb01(palette, cv["color"]),
                              lw=1.4, label=cv["name"]))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(10, len(handles)),
               fontsize=6.5, framealpha=0.9, borderpad=0.35, handlelength=1.6,
               labelspacing=0.25, columnspacing=1.15)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _scale_text(left, right) -> str:
    """'0–150', or '30 to -10' when a bound is negative (an en dash next to a minus reads badly)."""
    left, right = float(left), float(right)
    return f"{left:g} to {right:g}" if min(left, right) < 0 else f"{left:g}–{right:g}"


def _value_ticks(left: float, right: float, kind: str) -> list[float]:
    lo, hi = sorted((float(left), float(right)))
    if kind == "log" and lo > 0:
        k0, k1 = int(np.floor(np.log10(lo) + 1e-9)), int(np.ceil(np.log10(hi) - 1e-9))
        mant = lo / 10 ** np.floor(np.log10(lo) + 1e-9)
        return [v for v in (mant * 10.0 ** k for k in range(k0, k1 + 1)) if lo * 0.999 <= v <= hi * 1.001]
    from matplotlib.ticker import MaxNLocator
    t = [float(v) for v in MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)]
    return [v for v in t if lo - 1e-9 <= v <= hi + 1e-9]


def _value_to_x(v, left: float, right: float, kind: str, x0: float, x1: float):
    v = np.asarray(v, float)
    if kind == "log":
        f = (np.log10(v) - np.log10(left)) / (np.log10(right) - np.log10(left))
    else:
        f = (v - left) / (right - left)
    return x0 + f * (x1 - x0)


def plot_color_trace_pairs(ink: dict, traces: dict, curves: list[dict],
                           depth_map, depth_window: tuple[float, float],
                           track_xlim: dict, palette: dict | None = None,
                           title: str = "", figsize: tuple[float, float] | None = None,
                           base_ink=None):
    """Pre-traced ink vs traced polyline, the pairs side by side.

    For each curve the left panel is its isolated working image cropped to exactly
    that curve's track, the right panel the trace; both share the same x extent
    (the track, ticks labelled in the curve's printed scale and unit, log ticks for
    log scales) and the same depth range, with a faint grid. ``traces[name]`` is
    ``(depth, x_sheet)``; ``ink[name]`` the working image (array or path). With
    ``base_ink`` (True = shared black ink: grid, text, black curves) only the curve's
    own stroke is shown in the pre-traced panel. No reference LAS is drawn.
    """
    import matplotlib.pyplot as plt

    n = len(curves)
    if n < 1:
        raise ValueError("curves must be non-empty")
    sample = _as_image(next(iter(ink.values())))
    r0, r1, d0, d1 = _window_rows(depth_map, depth_window, sample.shape[0])
    if figsize is None:
        figsize = (1.05 * 2 * n + 0.6, 3.5)
    fig, axs = plt.subplots(1, 2 * n, figsize=figsize, sharey=True,
                            gridspec_kw={"wspace": 0.08}, layout="constrained")
    axs = np.atleast_1d(axs)
    step = 10.0 if (d1 - d0) <= 60 else 20.0
    hlines = np.arange(np.ceil(d0 / step) * step, d1 + 1e-6, step)
    for i, cv in enumerate(curves):
        x0, x1 = (float(v) for v in track_xlim[cv["track"]])
        c0, c1 = int(np.ceil(x0)), int(np.floor(x1)) + 1
        img = _as_image(ink[cv["name"]])
        if img.ndim == 3:
            img = img[..., 0]
        sl = img[r0:r1, c0:c1]
        if sl.dtype == bool or (sl.size and sl.max() <= 1):
            bits = sl.astype(bool)
            mask = bits if float(bits.mean()) < 0.5 else ~bits          # True = ink
        else:
            mask = sl < 128
        if base_ink is not None:
            mask = mask & ~np.asarray(base_ink, bool)[r0:r1, c0:c1]
        show = np.where(mask, 0, 255).astype(np.uint8)
        ax_l, ax_r = axs[2 * i], axs[2 * i + 1]
        ax_l.imshow(show, cmap="gray", origin="upper", aspect="auto", interpolation="antialiased",
                    vmin=0, vmax=255, extent=(c0 - 0.5, c1 - 0.5, d1, d0))
        d, x = traces[cv["name"]]
        d, x = np.asarray(d, float), np.asarray(x, float)
        m = np.isfinite(d) & np.isfinite(x) & (d >= d0 - 1) & (d <= d1 + 1)
        col = _rgb01(palette, cv["color"]) if palette and cv.get("color") in palette else "0.1"
        ax_r.plot(x[m], d[m], color=col, lw=0.9, solid_capstyle="butt")
        kind = cv.get("scale_type", "linear")
        left, right = cv.get("left"), cv.get("right")
        ticks = _value_ticks(left, right, kind) if left is not None and right is not None else []
        tx = [float(_value_to_x(v, left, right, kind, x0, x1)) for v in ticks]
        unit = cv.get("unit", "")
        for k, a in enumerate((ax_l, ax_r)):
            a.set_xlim(x0, x1)
            a.set_ylim(d1, d0)
            a.set_xticks(tx)
            a.set_xticklabels([f"{v:g}" for v in ticks], fontsize=5.5)
            for xv in tx:
                a.axvline(xv, color="0.6", lw=0.35, alpha=0.5, zorder=0)
            for yv in hlines:
                a.axhline(yv, color="0.6", lw=0.35, alpha=0.5, zorder=0)
            a.tick_params(labelsize=6, length=2, pad=1)
        ax_l.set_title(f"{cv['name']}\npre-traced", fontsize=7, pad=2)
        ax_r.set_title(f"{cv['name']}\ntraced", fontsize=7, pad=2)
        lab = f"{cv['name']} ({unit})" if unit else cv["name"]
        ax_l.set_xlabel(lab + (", log" if kind == "log" else ""), fontsize=6, labelpad=1)
        ax_r.set_xlabel(lab + (", log" if kind == "log" else ""), fontsize=6, labelpad=1)
    axs[0].set_ylabel("depth (ft)", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig
