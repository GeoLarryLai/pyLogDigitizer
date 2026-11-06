#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Annotate FORGE induction off-chart wraps → digital_print_wrap/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

from pylogdigitizer.pdf_vector import (
    digitize_pdf, detect_page_layout, annotate_layout_figure, OFF_CHART_WRAP_NOTE,
)
from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.assemble import traces_from_page, is_near_white, classify_strokes
from pylogdigitizer.pdf_vector.digitize import _collect_headers

PDF = Path("/workspace/pld_vector_pdf/forge_78B_graphics/UOU_FORGE-78B-32_R1B_TD_SECTION_Induction_30Jul21_070636.Pdf")
VIZ = Path("/workspace/pld_viz_out/digital_print_wrap")
OUT = Path("artifacts/pdf_vector")
VIZ.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PAGE = 10


def main() -> None:
    reader = PdfReader(str(PDF))
    headers = _collect_headers(reader, [PAGE])
    pv = extract_page(reader, PAGE)
    traces = traces_from_page(pv, headers=headers)
    layout = detect_page_layout(pv, traces=traces)

    wrap_curves = [t for t in traces if t.info.get("off_chart_wrap")]
    print("OFF_CHART_WRAP_NOTE:", OFF_CHART_WRAP_NOTE[:80], "...")
    print("traces", [(t.mnemonic, t.info.get("off_chart_wrap"), t.info.get("n_wraps"),
                      t.info.get("n_wrap_pts"), round(float(np.nanmax(t.value)), 1)) for t in traces])
    print("scale regions wrap policy:",
          [(r.label, r.info.get("off_chart_wrap_policy"), bool(r.info.get("off_chart_wrap_note")))
           for r in layout.regions if r.kind == "scale_legend"])

    # Overview with wrap callout
    fig = annotate_layout_figure(
        pv, layout, traces=traces,
        out_path=str(VIZ / "induction_p10_wrap_overview.png"),
        title="FORGE induction p10 — off-chart wraps",
    )
    plt.close(fig)

    # Dedicated wrap figure: track + primary/wrap strokes highlighted
    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
    ax.set_xlim(0, pv.width); ax.set_ylim(0, pv.height); ax.set_aspect("equal")
    ax.set_title("AT* log track: primary + left-edge off-chart wrap")
    # shade log track from first wrap/log curve
    log_t = next((t for t in traces if t.kind == "log" and t.info.get("track")), None)
    if log_t:
        x0, x1 = log_t.info["track"]
        ax.axvspan(x0, x1, color="#2ca02c", alpha=0.08, label="log track")
        ax.axvline(x0 + 0.35 * (x1 - x0), color="#d62728", ls="--", lw=0.8, label="left wrap zone")
    gh, gv, curves = classify_strokes(pv.strokes)
    for s in curves:
        if float(np.median(s.pts[:, 0])) < 200:
            continue
        if is_near_white(s.color):
            ax.plot(s.pts[:, 0], s.pts[:, 1], color="0.75", lw=1.2, alpha=0.9)
        else:
            ax.plot(s.pts[:, 0], s.pts[:, 1], color=s.color, lw=0.7, alpha=0.7)
    # mark wrap segments
    for t in wrap_curves:
        for seg in t.info.get("wrap_segments") or []:
            xr, yr = seg["x_range"], seg["y_range"]
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((xr[0], yr[0]), xr[1]-xr[0], yr[1]-yr[0],
                                   fill=False, edgecolor="#d62728", lw=1.5, ls="--"))
            ax.text(xr[0], yr[1], f"{t.mnemonic} WRAP +1 decade", color="#d62728", fontsize=8)
    ax.text(0.02, 0.02,
            "OFF-CHART WRAP: overflow reappears on opposite (left) edge → value on next decade/backup",
            transform=ax.transAxes, fontsize=8, color="#d62728",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d62728", alpha=0.9))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(VIZ / "induction_p10_wrap_detail.png", dpi=120)
    plt.close(fig)

    # Per wrap-curve value plot
    for t in wrap_curves or [t for t in traces if t.mnemonic == "AT20"][:1]:
        fig, ax = plt.subplots(figsize=(4, 8))
        ok = np.isfinite(t.depth) & np.isfinite(t.value)
        ax.plot(t.value[ok], t.depth[ok], lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel(t.unit or "value"); ax.set_ylabel("depth (ft)")
        ax.set_title(f"{t.mnemonic} wrap={t.info.get('off_chart_wrap')} n_wraps={t.info.get('n_wraps')}")
        ax.axvline(t.info.get("v_right", np.nan), color="0.5", ls=":", label="scale right")
        fig.tight_layout()
        fig.savefig(VIZ / f"induction_p10_{t.mnemonic}_values.png", dpi=120)
        plt.close(fig)

    payload = {
        "source_pdf": str(PDF), "page": PAGE,
        "off_chart_wrap_note": OFF_CHART_WRAP_NOTE,
        "layout": layout.as_dict(),
        "curves": {
            t.mnemonic: {
                "off_chart_wrap": t.info.get("off_chart_wrap"),
                "n_wraps": t.info.get("n_wraps"),
                "n_wrap_pts": t.info.get("n_wrap_pts"),
                "wrap_segments": t.info.get("wrap_segments"),
                "v_range": [float(np.nanmin(t.value)), float(np.nanmax(t.value))],
                "n_pts": t.info.get("n_pts"),
            } for t in traces
        },
    }
    (VIZ / "induction_p10_wrap.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "forge78B_induction_wrap.regions.json").write_text(json.dumps(payload, indent=2, default=str))
    print("figs →", VIZ)


if __name__ == "__main__":
    main()
