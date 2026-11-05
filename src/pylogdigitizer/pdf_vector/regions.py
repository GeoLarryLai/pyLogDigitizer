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
        # Vertical oilfield: header/legend near top (above first deep body), tracks as x columns.
        grid_h, grid_v, _ = classify_strokes(page.strokes)
        if headers:
            hy = max(h.y for h in headers)
            top_texts = [t for t in page.texts if t.y >= hy - 30]
            bb = _bbox_texts(top_texts, pad=8) if top_texts else None
            if bb:
                regions.append(Region(
                    kind="header", x0=0.0, y0=bb[1], x1=page.width, y1=page.height,
                    label="title_header", info={"role": "top_block"},
                ))
                regions.append(Region(
                    kind="scale_legend", x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3],
                    label="curve_scales",
                    info={"placement": "upper",
                          "mnemonics": [h.mnemonic for h in headers]},
                ))
        # Track columns from v-grid xs + header x positions.
        xs = sorted({float(np.mean(s.pts[:, 0])) for s in grid_v if float(np.ptp(s.pts[:, 1])) > 40})
        for h in headers:
            xs.append(h.x)
        xs = sorted(set(round(x, 0) for x in xs))
        # Merge into spans ~180 pt for linear, wider for log — use consecutive edges.
        edges = [0.0] + [float(x) for x in xs] + [page.width]
        edges = sorted(set(edges))
        body_top = max((h.y for h in headers), default=page.height) - 40
        for i in range(len(edges) - 1):
            x0, x1 = edges[i], edges[i + 1]
            if x1 - x0 < 40:
                continue
            track_headers = [h for h in headers if x0 - 20 <= h.x <= x1 + 10]
            if not track_headers and x1 - x0 > 400:
                continue
            regions.append(Region(
                kind="track", x0=x0, y0=0.0, x1=x1, y1=body_top,
                label=",".join(h.mnemonic for h in track_headers) or f"col_{i}",
                info={"curves": [h.mnemonic for h in track_headers]},
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
                # Depth extent → x via inverse depth map when possible.
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
                          "v_right": t.info.get("v_right")},
                ))
            elif axis == "y" and track:
                x0, x1 = track
                regions.append(Region(
                    kind="curve", x0=x0, y0=0.0, x1=x1, y1=page.height,
                    label=t.mnemonic,
                    info={"n_pts": t.info.get("n_pts", len(t.depth)),
                          "unit": t.unit},
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
):
    """Draw region boxes + curve polylines; save PNG if ``out_path`` given.

    Returns the matplotlib Figure (caller may close). Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(page.width / 90, page.height / 90), dpi=dpi)
    ax.set_xlim(0, page.width)
    ax.set_ylim(0, page.height)
    ax.set_aspect("equal")
    ax.set_title(f"page {page.page_index} axis={layout.axis}")

    colors = {
        "header": "#1f77b4",
        "scale_legend": "#ff7f0e",
        "track": "#2ca02c",
        "depth_axis": "#9467bd",
        "curve": "#d62728",
    }
    # Light track fills first.
    for r in layout.regions:
        if r.kind != "track":
            continue
        ax.add_patch(Rectangle(
            (r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
            fill=True, facecolor=colors["track"], alpha=0.08,
            edgecolor=colors["track"], linewidth=0.8,
        ))
    for r in layout.regions:
        if r.kind == "track":
            continue
        c = colors.get(r.kind, "#333333")
        ax.add_patch(Rectangle(
            (r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
            fill=False, edgecolor=c, linewidth=1.2 if r.kind != "curve" else 0.6,
            linestyle="--" if r.kind == "curve" else "-",
        ))
        if r.label and r.kind in {"scale_legend", "header", "depth_axis"}:
            ax.text(r.x0 + 2, r.y1 - 8, f"{r.kind}:{r.label}"[:40],
                    fontsize=6, color=c, va="top")

    # Raw long strokes for context.
    for s in page.strokes:
        if len(s.pts) < 40:
            continue
        ax.plot(s.pts[:, 0], s.pts[:, 1], color="0.5", lw=0.4, alpha=0.5)

    if traces:
        cmap = plt.cm.tab10
        for i, t in enumerate(traces):
            # Re-project depth→x or depth→y for overlay is layout-specific; use track strip label only.
            ax.text(2, page.height - 12 * (i + 1), t.mnemonic, fontsize=7, color=cmap(i % 10))

    # Legend
    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, label=k)
        for k, c in colors.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.9)
    ax.set_xlabel("x (pt)")
    ax.set_ylabel("y (pt)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi)
    return fig
