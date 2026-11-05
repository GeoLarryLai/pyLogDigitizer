# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Auto-detect structural regions on a digital-print log page.

Regions are geometry-first (text clusters + grid spans), not hand-drawn templates:

1. ``header`` — title / well-info block (page 0 forms, or left strip on USGS)
2. ``scale_legend`` — curve title + scale label clusters (upper / lower per track)
3. ``track`` — grid-bounded columns (vertical) or bands (horizontal)
4. ``depth_axis`` — depth tick column/row
5. ``curve`` — one region per digitized curve (bbox of its stroke)

Shared DepthMap / ScaleMap / LAS writing stay elsewhere; this module only
describes *where* things are so callers can annotate and QC.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from pylogdigitizer.pdf_vector.extract import PageVectors, TextItem, Stroke
from pylogdigitizer.pdf_vector.calibrate import (
    detect_depth_axis,
    depth_map_from_labels,
    parse_curve_headers,
    parse_scale_label,
    CurveHeader,
)
from pylogdigitizer.pdf_vector.assemble import (
    CurveTrace,
    _hgrid_ys,
    classify_strokes,
)


@dataclass
class Region:
    """Axis-aligned page region in PDF points (origin bottom-left)."""
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class PageLayout:
    """Structural parse of one digital-print page."""
    page_index: int
    axis: str  # "x" horizontal depth | "y" vertical depth
    width: float
    height: float
    regions: list[Region] = field(default_factory=list)
    headers: list[CurveHeader] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def by_kind(self, kind: str) -> list[Region]:
        return [r for r in self.regions if r.kind == kind]

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "axis": self.axis,
            "width": self.width,
            "height": self.height,
            "regions": [r.as_dict() for r in self.regions],
            "headers": [
                {"mnemonic": h.mnemonic, "v_left": h.v_left, "v_right": h.v_right,
                 "unit": h.unit, "kind": h.kind, "x": h.x, "y": h.y}
                for h in self.headers
            ],
            "info": self.info,
        }


def _bbox_texts(items: list[TextItem], pad: float = 4.0) -> tuple[float, float, float, float] | None:
    if not items:
        return None
    xs = [t.x for t in items]
    ys = [t.y for t in items]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad + 40, max(ys) + pad + 12)


def _bbox_stroke(s: Stroke, pad: float = 2.0) -> tuple[float, float, float, float]:
    return (
        float(s.pts[:, 0].min()) - pad,
        float(s.pts[:, 1].min()) - pad,
        float(s.pts[:, 0].max()) + pad,
        float(s.pts[:, 1].max()) + pad,
    )


def detect_page_layout(
    page: PageVectors,
    traces: list[CurveTrace] | None = None,
) -> PageLayout:
    """Detect header / scale-legend / track-grid / depth-axis / curve regions."""
    axis = detect_depth_axis(page.texts)
    headers = parse_curve_headers(page.texts)
    regions: list[Region] = []
    info: dict[str, Any] = {"n_strokes": len(page.strokes), "n_texts": len(page.texts)}

    # --- depth axis from numeric ticks ---
    dm = depth_map_from_labels(page.texts, axis=axis)
    depth_ticks = []
    for t in page.texts:
        s = t.text.strip()
        if s.isdigit() and 0 <= int(s) <= 50000:
            depth_ticks.append(t)
    if depth_ticks:
        bb = _bbox_texts(depth_ticks, pad=6)
        if bb:
            regions.append(Region(
                kind="depth_axis", x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3],
                label="depth", info={"n_ticks": len(depth_ticks), "axis": axis,
                                     "depths": list(dm.depths) if dm is not None else []},
            ))

    if axis == "x":
        bounds = _hgrid_ys(page.strokes)
        # Left strip: titles + scales (everything left of first depth tick).
        x_plot = float(dm.rows[0]) if dm is not None else page.width * 0.15
        if depth_ticks:
            x_plot = min(t.x for t in depth_ticks) - 5
        left_texts = [t for t in page.texts if t.x < x_plot]
        if left_texts:
            bb = _bbox_texts(left_texts, pad=8)
            if bb:
                regions.append(Region(
                    kind="header", x0=0.0, y0=bb[1], x1=max(bb[2], x_plot),
                    y1=bb[3], label="curve_header_strip",
                    info={"role": "left_titles_scales"},
                ))
        # Per-track bands from H-grid + scale legends inside each band.
        for i in range(len(bounds) - 1):
            y0, y1 = bounds[i], bounds[i + 1]
            if y1 - y0 < 20:
                continue
            # Scales sit near the lower edge of USGS tracks; require that.
            track_headers = [h for h in headers if y0 - 8 <= h.y <= y0 + 35 and h.y < y1]
            regions.append(Region(
                kind="track", x0=x_plot, y0=y0, x1=page.width, y1=y1,
                label=",".join(h.mnemonic for h in track_headers) or f"track_{i}",
                info={"y0": y0, "y1": y1, "curves": [h.mnemonic for h in track_headers],
                      "v_scales": [(h.mnemonic, h.v_left, h.v_right, h.unit) for h in track_headers]},
            ))
            # Scale/legend cluster: title+scale texts for those headers near band bottom.
            leg_texts = []
            for h in track_headers:
                for t in page.texts:
                    if abs(t.y - h.y) < 30 and t.x < x_plot + 5:
                        leg_texts.append(t)
            if leg_texts:
                bb = _bbox_texts(leg_texts, pad=6)
                if bb:
                    regions.append(Region(
                        kind="scale_legend", x0=bb[0], y0=max(bb[1], y0 - 5),
                        x1=min(bb[2], x_plot + 10), y1=min(bb[3], y1 + 5),
                        label=",".join(h.mnemonic for h in track_headers),
                        info={"placement": "lower" if (bb[1] + bb[3]) / 2 < (y0 + y1) / 2 else "upper",
                              "mnemonics": [h.mnemonic for h in track_headers]},
                    ))
        info["n_tracks"] = sum(1 for r in regions if r.kind == "track")
    else:
        # Vertical oilfield (FORGE / Schlumberger digital prints).
        grid_h, grid_v, body = classify_strokes(page.strokes)
        depth_x = float(dm.info["x_col"]) if dm is not None and "x_col" in dm.info else None

        # Header-only page: form / TOC with no depth ticks and almost no body strokes.
        if dm is None and len(body) < 2:
            bb = _bbox_texts(page.texts, pad=10)
            if bb:
                regions.append(Region(
                    kind="header", x0=0.0, y0=0.0, x1=page.width, y1=page.height,
                    label="well_header", info={"role": "form_or_toc", "n_texts": len(page.texts)},
                ))
            else:
                regions.append(Region(
                    kind="header", x0=0.0, y0=0.0, x1=page.width, y1=page.height,
                    label="well_header", info={"role": "empty_or_cover"},
                ))
        else:
            # Scale numbers ("0 150 gAPI") and non-depth name texts.
            scale_texts = [t for t in page.texts if parse_scale_label(t.text)]
            name_texts = []
            for t in page.texts:
                s = t.text.strip()
                if not s or (s.isdigit() and 0 <= int(s) <= 50000):
                    continue
                if parse_scale_label(s):
                    continue
                if len(s) < 2:
                    continue
                name_texts.append(t)

            # Explicit curve headers (mnemonic + scale) from calibrate.
            for h in headers:
                name_texts.append(TextItem(x=h.x, y=h.y, text=h.mnemonic, size=10.0))

            # Upper / lower scale sections = clusters of scale labels (preferred)
            # or mnemonic footers when body pages only reprint names.
            def _emit_scale(cluster, placement: str, label: str) -> None:
                if not cluster:
                    return
                bb = _bbox_texts(cluster, pad=12)
                if not bb:
                    return
                # Pull nearby titles into the same band (±36 pt).
                y_mid = 0.5 * (bb[1] + bb[3])
                extras = [t for t in name_texts if abs(t.y - y_mid) < 40]
                if extras:
                    bb2 = _bbox_texts(list(cluster) + extras, pad=12)
                    if bb2:
                        bb = bb2
                regions.append(Region(
                    kind="scale_legend",
                    x0=max(0.0, bb[0]), y0=max(0.0, bb[1]),
                    x1=min(page.width, bb[2]), y1=min(page.height, bb[3]),
                    label=label,
                    info={"placement": placement,
                          "labels": sorted({t.text.strip() for t in cluster})[:24],
                          "n_scale_labels": sum(1 for t in cluster if parse_scale_label(t.text))},
                ))

            if scale_texts:
                ys = [t.y for t in scale_texts]
                y_med = float(sorted(ys)[len(ys) // 2])
                upper_sc = [t for t in scale_texts if t.y >= y_med - 5]
                lower_sc = [t for t in scale_texts if t.y < y_med - 5]
                # Single cluster → classify by page position.
                if not lower_sc or (max(ys) - min(ys) < 80):
                    placement = "upper" if y_med >= 0.45 * page.height else "lower"
                    _emit_scale(scale_texts, placement, f"{placement}_scale")
                else:
                    _emit_scale(upper_sc, "upper", "upper_scale")
                    _emit_scale(lower_sc, "lower", "lower_scale")
            else:
                # Body pages: mnemonic reprint near top and/or bottom.
                upper = [t for t in name_texts if t.y >= 0.70 * page.height]
                lower = [t for t in name_texts if t.y <= 0.40 * page.height]
                _emit_scale(upper, "upper", "upper_scale")
                _emit_scale(lower, "lower", "lower_scale")

            # Remaining high text without scales → thin header strip (toolstring).
            if scale_texts or headers:
                top_band = [t for t in page.texts if t.y >= 0.85 * page.height]
                if top_band and not any(r.kind == "header" for r in regions):
                    bb = _bbox_texts(top_band, pad=8)
                    if bb and bb[3] - bb[1] < 0.25 * page.height:
                        regions.append(Region(
                            kind="header", x0=0.0, y0=bb[1], x1=page.width, y1=page.height,
                            label="page_header_strip",
                            info={"role": "toolstring_or_title"},
                        ))

            # Track columns: prefer spans from digitized traces; else v-grid + header x.
            spans: list[tuple[float, float, str]] = []
            if traces:
                seen_sp = set()
                for t in traces:
                    tr = t.info.get("track")
                    if not tr:
                        continue
                    key = (round(tr[0], 1), round(tr[1], 1))
                    if key in seen_sp:
                        # append mnemonic to existing
                        for i, (a, b, lab) in enumerate(spans):
                            if (round(a, 1), round(b, 1)) == key:
                                spans[i] = (a, b, lab + ("," + t.mnemonic if t.mnemonic not in lab else ""))
                                break
                        continue
                    seen_sp.add(key)
                    spans.append((float(tr[0]), float(tr[1]), t.mnemonic))
            if not spans:
                xs = sorted({float(np.mean(s.pts[:, 0])) for s in grid_v if float(np.ptp(s.pts[:, 1])) > 40})
                for h in headers:
                    xs.append(h.x)
                if depth_x is not None:
                    xs.append(depth_x)
                xs = sorted(set(round(x, 0) for x in xs))
                edges = [0.0] + [float(x) for x in xs] + [page.width]
                edges = sorted(set(edges))
                for i in range(len(edges) - 1):
                    x0, x1 = edges[i], edges[i + 1]
                    if x1 - x0 < 40:
                        continue
                    th = [h for h in headers if x0 - 20 <= h.x <= x1 + 10]
                    spans.append((x0, x1, ",".join(h.mnemonic for h in th) or f"col_{i}"))

            # Body vertical extent: between upper and lower scale bands when present.
            y_lo = 0.0
            y_hi = page.height
            for r in regions:
                if r.kind == "scale_legend" and r.info.get("placement") == "lower":
                    y_lo = max(y_lo, r.y1)
                if r.kind == "scale_legend" and r.info.get("placement") == "upper":
                    y_hi = min(y_hi, r.y0)
            if y_hi - y_lo < 100:
                y_lo, y_hi = 0.0, page.height

            for x0, x1, lab in spans:
                # Skip pure depth-column sliver.
                if depth_x is not None and abs(0.5 * (x0 + x1) - depth_x) < 15 and (x1 - x0) < 50:
                    continue
                # Grid line xs inside this track (for QC).
                grid_xs = sorted({
                    round(float(np.mean(s.pts[:, 0])), 1)
                    for s in grid_v
                    if x0 - 2 <= float(np.mean(s.pts[:, 0])) <= x1 + 2
                })
                grid_ys = sorted({
                    round(float(np.mean(s.pts[:, 1])), 1)
                    for s in grid_h
                    if float(np.ptp(s.pts[:, 0])) > 0.5 * (x1 - x0)
                    and x0 - 20 <= float(np.mean(s.pts[:, 0])) <= x1 + 20
                })
                regions.append(Region(
                    kind="track", x0=x0, y0=y_lo, x1=x1, y1=y_hi,
                    label=lab or "track",
                    info={"curves": [p for p in lab.split(",") if p],
                          "grid_v_xs": grid_xs[:30],
                          "grid_h_ys": grid_ys[:40],
                          "n_h_grid": len(grid_ys), "n_v_grid": len(grid_xs)},
                ))
            info["n_tracks"] = sum(1 for r in regions if r.kind == "track")

    # Per-curve regions from digitized traces (stroke bbox if available via info.track).
    if traces:
        for t in traces:
            if len(t.depth) == 0:
                continue
            track = t.info.get("track")
            if axis == "x" and track:
                y0, y1 = track
                if dm is not None:
                    x0 = float(dm.inverse(float(np.nanmin(t.depth))))
                    x1 = float(dm.inverse(float(np.nanmax(t.depth))))
                else:
                    x0, x1 = 0.0, page.width
                regions.append(Region(
                    kind="curve", x0=min(x0, x1), y0=y0, x1=max(x0, x1), y1=y1,
                    label=t.mnemonic,
                    info={"n_pts": t.info.get("n_pts", len(t.depth)),
                          "unit": t.unit, "v_left": t.info.get("v_left"),
                          "v_right": t.info.get("v_right"),
                          "values_depth_range": [float(np.nanmin(t.depth)), float(np.nanmax(t.depth))],
                          "values_range": [float(np.nanmin(t.value)), float(np.nanmax(t.value))]
                          if np.isfinite(t.value).any() else None},
                ))
            elif axis == "y" and track:
                x0, x1 = track
                # Vertical body band for the curve column.
                y0, y1 = 0.0, page.height
                for r in regions:
                    if r.kind == "track" and abs(r.x0 - x0) < 2 and abs(r.x1 - x1) < 2:
                        y0, y1 = r.y0, r.y1
                        break
                regions.append(Region(
                    kind="curve", x0=x0, y0=y0, x1=x1, y1=y1,
                    label=t.mnemonic,
                    info={"n_pts": t.info.get("n_pts", len(t.depth)),
                          "unit": t.unit,
                          "v_left": t.info.get("v_left"),
                          "v_right": t.info.get("v_right"),
                          "values_depth_range": [float(np.nanmin(t.depth)), float(np.nanmax(t.depth))],
                          "values_range": [float(np.nanmin(t.value)), float(np.nanmax(t.value))]
                          if np.isfinite(t.value).any() else None},
                ))

    return PageLayout(
        page_index=page.page_index, axis=axis,
        width=page.width, height=page.height,
        regions=regions, headers=headers, info=info,
    )


def annotate_layout_figure(
    page: PageVectors,
    layout: PageLayout,
    traces: list[CurveTrace] | None = None,
    out_path: str | None = None,
    dpi: int = 120,
    kinds: set[str] | None = None,
    title: str | None = None,
):
    """Draw region boxes (+ optional curve strokes); save PNG if ``out_path`` given.

    ``kinds`` limits which region kinds are outlined (default: all).
    Returns the matplotlib Figure (caller may close). Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(max(4.0, page.width / 90), max(5.0, page.height / 90)), dpi=dpi)
    ax.set_xlim(0, page.width)
    ax.set_ylim(0, page.height)
    ax.set_aspect("equal")
    ax.set_title(title or f"page {page.page_index} axis={layout.axis}")

    colors = {
        "header": "#1f77b4",
        "scale_legend": "#ff7f0e",
        "track": "#2ca02c",
        "depth_axis": "#9467bd",
        "curve": "#d62728",
    }
    show = kinds or set(colors)

    for r in layout.regions:
        if r.kind == "track" and "track" in show:
            ax.add_patch(Rectangle(
                (r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
                fill=True, facecolor=colors["track"], alpha=0.07,
                edgecolor=colors["track"], linewidth=1.0,
            ))
            # Draw a few H-grid guides if recorded.
            for y in (r.info.get("grid_h_ys") or [])[::2][:12]:
                ax.plot([r.x0, r.x1], [y, y], color=colors["track"], lw=0.3, alpha=0.5)

    for r in layout.regions:
        if r.kind == "track" or r.kind not in show:
            continue
        c = colors.get(r.kind, "#333333")
        ax.add_patch(Rectangle(
            (r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
            fill=False, edgecolor=c, linewidth=1.4 if r.kind != "curve" else 0.9,
            linestyle="--" if r.kind == "curve" else "-",
        ))
        ax.text(r.x0 + 3, min(r.y1 - 6, page.height - 6), f"{r.kind}:{r.label}"[:48],
                fontsize=6, color=c, va="top")

    # Curve strokes from page (style match) — light context.
    for s in page.strokes:
        if len(s.pts) < 40:
            continue
        ax.plot(s.pts[:, 0], s.pts[:, 1], color="0.55", lw=0.35, alpha=0.35)

    if traces:
        cmap = plt.cm.tab10
        for i, t in enumerate(traces):
            track = t.info.get("track")
            if not track:
                continue
            ax.text(2, page.height - 12 * (i + 1),
                    f"{t.mnemonic} n={t.info.get('n_pts', len(t.depth))}",
                    fontsize=7, color=cmap(i % 10))

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, label=k)
        for k, c in colors.items() if k in show
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.9)
    ax.set_xlabel("x (pt)")
    ax.set_ylabel("y (pt)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi)
    return fig
