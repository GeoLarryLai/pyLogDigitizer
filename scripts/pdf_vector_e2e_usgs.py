#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""E2E: USGS GeoLog Locator digital-print PDF → regions + LAS + scores.

Matterhorn Ro-699 (usgs_21.pdf + Matterhorn_7-10.las). Figures under
/workspace/pld_viz_out/digital_print_usgs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pylogdigitizer.pdf_vector import (
    digitize_pdf,
    annotate_layout_figure,
    extract_page,
)
from pylogdigitizer.las_export import verify_las, read_las
from pylogdigitizer.qc_compare import metrics, to_grid

USGS = Path("/workspace/pld_vector_pdf/usgs")
PDF = USGS / "usgs_21.pdf"
LAS = USGS / "Matterhorn_7-10.las"
VIZ = Path("/workspace/pld_viz_out/digital_print_usgs")
OUT = Path("artifacts/pdf_vector")
VIZ.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ALIAS = {
    "GAMMA": "GAMMA",
    "CALIPER": "CALIPER",
    "FL_COND_AMB": "FL COND AMB",
    "FL_COND_PMP": "FL COND PMP",
    "FL_TEMP_AMB": "FL  TEMP AMB",
    "FL_TEMP_PMP": "FL TEMP PMP",
}


def main() -> None:
    log = digitize_pdf(PDF, pages=[1], step=0.5)
    # Enrich well from header page.
    from pypdf import PdfReader
    from pylogdigitizer.pdf_vector.calibrate import parse_well_header
    from pylogdigitizer.pdf_vector.extract import extract_page as ep

    reader = PdfReader(str(PDF))
    well0 = parse_well_header(ep(reader, 0).texts)
    well1 = parse_well_header(ep(reader, 1).texts)
    log.well = {**well0, **well1, **log.well}

    las_path = OUT / "matterhorn_ro699_digitized.las"
    log.to_las(las_path)
    print("wrote", las_path, "verify", verify_las(las_path, log.depth, log.curves) or "ok")
    try:
        print("lasio curves", [c.mnemonic for c in read_las(las_path).curves])
    except Exception as e:
        print("lasio reload skipped", e)

    # Layout annotation
    pv = ep(reader, 1)
    layout = log.layouts[0] if log.layouts else None
    if layout is not None:
        fig_path = VIZ / "matterhorn_regions.png"
        fig = annotate_layout_figure(pv, layout, traces=log.traces, out_path=str(fig_path))
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("regions", [(r.kind, r.label) for r in layout.regions])
        (VIZ / "matterhorn_layout.json").write_text(json.dumps(layout.as_dict(), indent=2, default=str))

    # Per-curve overlay figures
    import matplotlib.pyplot as plt
    import lasio
    df = lasio.read(str(LAS)).df()
    scores = {}
    for t in log.traces:
        m = ALIAS.get(t.mnemonic)
        if not m or m not in df.columns:
            continue
        ref = df[m].replace(-99999, np.nan).dropna()
        ok = np.isfinite(log.depth) & np.isfinite(log.curves.get(t.mnemonic, np.array([])))
        if t.mnemonic not in log.curves:
            continue
        td = log.depth[ok]
        tv = log.curves[t.mnemonic][ok]
        if len(td) < 20:
            continue
        # Align on digitized depths
        ok2 = np.isfinite(t.depth) & np.isfinite(t.value)
        td2, tv2 = t.depth[ok2], t.value[ok2]
        rv = to_grid(ref.index.values, ref.values, td2)
        sc = metrics(rv, tv2)
        scores[t.mnemonic] = sc
        print(f"  {t.mnemonic}->{m}: r={sc['r']:.4f} rmse={sc['rmse']:.3g} n={sc['n']}")
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(ref.index, ref.values, lw=0.8, color="0.6", label="LAS")
        ax.plot(td2, tv2, lw=0.9, color="C0", label="digitized")
        ax.set_title(f"{t.mnemonic} r={sc['r']:.3f}")
        ax.set_xlabel("depth (ft)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(VIZ / f"matterhorn_{t.mnemonic}.png", dpi=120)
        plt.close(fig)

    (VIZ / "e2e_scores.json").write_text(json.dumps({"well": log.well, "scores": scores}, indent=2, default=str))
    print("scores →", VIZ / "e2e_scores.json")
    print("well", log.well)


if __name__ == "__main__":
    main()
