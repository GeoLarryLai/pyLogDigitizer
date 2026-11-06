#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FORGE CBL digital-print layout demo: regions + annotated figures + LAS JSON.

Figures → /workspace/pld_viz_out/digital_print_forge_layout/
Uses existing CBL digitize path; confirms DigitizedLog.layouts; writes regions
JSON next to the LAS under artifacts/pdf_vector/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from pypdf import PdfReader

from pylogdigitizer.pdf_vector import (
    digitize_pdf,
    detect_page_layout,
    annotate_layout_figure,
)
from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.assemble import traces_from_page
from pylogdigitizer.las_export import verify_las

PDF = Path("/workspace/pld_vector_pdf/UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf")
VIZ = Path("/workspace/pld_viz_out/digital_print_forge_layout")
OUT = Path("artifacts/pdf_vector")
VIZ.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Header form + mid curve page (p20) + deep legend-ish page for upper/lower scales.
HEADER_PAGE = 0
SCALE_PAGE = 4   # first log page with numeric scale labels
MID_PAGE = 20
# Digitize a short span that includes mid page for real curves.
DIG_PAGES = [5, 10, 19, 20, 29]


def _save_kinds(pv, layout, traces, stem: str, kinds_list: list[tuple[str, set[str] | None]]) -> None:
    for tag, kinds in kinds_list:
        fig = annotate_layout_figure(
            pv, layout, traces=traces,
            out_path=str(VIZ / f"{stem}_{tag}.png"),
            kinds=kinds,
            title=f"{stem} — {tag}",
        )
        plt.close(fig)


def main() -> None:
    log = digitize_pdf(PDF, pages=DIG_PAGES, step=0.5)
    assert log.layouts, "DigitizedLog.layouts must be filled"
    las_path = OUT / "forge78B_cbl_digitized.las"
    log.to_las(las_path)
    print("LAS", las_path, "verify", verify_las(las_path, log.depth, log.curves) or "ok")
    print("curves", sorted(log.curves))
    print("layouts pages", [L.page_index for L in log.layouts], "axes", [L.axis for L in log.layouts])

    # Regions JSON next to LAS
    regions_path = OUT / "forge78B_cbl_digitized.regions.json"
    payload = {
        "source_pdf": str(PDF),
        "las": str(las_path),
        "well": log.well,
        "layouts": [L.as_dict() for L in log.layouts],
    }
    regions_path.write_text(json.dumps(payload, indent=2, default=str))
    print("regions JSON", regions_path)

    reader = PdfReader(str(PDF))

    # --- Header page (form) ---
    pv0 = extract_page(reader, HEADER_PAGE)
    lay0 = detect_page_layout(pv0, traces=None)
    (VIZ / "p00_header_layout.json").write_text(json.dumps(lay0.as_dict(), indent=2, default=str))
    _save_kinds(pv0, lay0, None, "p00", [
        ("overview", None),
        ("header_only", {"header"}),
    ])
    print("p0 regions", [(r.kind, r.label) for r in lay0.regions])

    # --- First scale page (numeric upper/lower scale sections) ---
    pv4 = extract_page(reader, SCALE_PAGE)
    page_traces4 = traces_from_page(pv4, headers=log.headers)
    lay4 = detect_page_layout(pv4, traces=page_traces4)
    (VIZ / "p04_layout.json").write_text(json.dumps(lay4.as_dict(), indent=2, default=str))
    _save_kinds(pv4, lay4, page_traces4, "p04", [
        ("overview", None),
        ("header_strip", {"header"}),
        ("upper_scale", {"scale_legend"}),
        ("tracks_grids", {"track", "depth_axis", "scale_legend"}),
        ("curves", {"curve", "track"}),
    ])
    print("p4 regions", [(r.kind, r.label, r.info.get("placement")) for r in lay4.regions])

    # --- Mid curve page with doc headers for style match ---
    pv20 = extract_page(reader, MID_PAGE)
    page_traces = traces_from_page(pv20, headers=log.headers)
    lay20 = detect_page_layout(pv20, traces=page_traces)
    (VIZ / "p20_layout.json").write_text(json.dumps(lay20.as_dict(), indent=2, default=str))
    _save_kinds(pv20, lay20, page_traces, "p20", [
        ("overview", None),
        ("upper_scale", {"scale_legend"}),
        ("lower_scale", {"scale_legend"}),
        ("tracks_grids", {"track", "depth_axis"}),
        ("curves", {"curve", "track"}),
    ])
    # One figure per curve
    for t in page_traces:
        fig = annotate_layout_figure(
            pv20, lay20, traces=[t],
            out_path=str(VIZ / f"p20_curve_{t.mnemonic}.png"),
            kinds={"curve", "track", "depth_axis"},
            title=f"p20 curve {t.mnemonic}",
        )
        plt.close(fig)
    print("p20 regions", [(r.kind, r.label, r.info.get("placement")) for r in lay20.regions])
    print("p20 traces", [(t.mnemonic, t.info.get("n_pts"), t.info.get("track")) for t in page_traces])
    print("figs →", VIZ)


if __name__ == "__main__":
    main()
