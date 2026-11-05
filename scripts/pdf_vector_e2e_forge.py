#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""E2E: digitize FORGE 78B-32 CBL + TD induction vector PDFs; QC vs companion LAS.

Legend styles (colour+width+dash) are auto-read from the PDF header. Explicit
``mnemonic_colors`` / ``mnemonic_styles`` remain as optional overrides.
Figures → /workspace/pld_viz_out/pdf_vector2/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lasio

from pylogdigitizer.pdf_vector import digitize_pdf
from pylogdigitizer.qc_compare import qc_plot

VIZ = Path("/workspace/pld_viz_out/pdf_vector2")
VIZ.mkdir(parents=True, exist_ok=True)
DATA = Path("/workspace/pld_vector_pdf")
OUT = Path("/workspace/pld_pdf/artifacts/pdf_vector")
OUT.mkdir(parents=True, exist_ok=True)


def qc_curves(log, las_path: Path, alias: dict[str, str], tag: str) -> dict:
    from pylogdigitizer.qc_compare import metrics, to_grid, align_depth, shift as shift_arr
    las = lasio.read(str(las_path))
    df = las.df()
    scores = {}
    for t in log.traces:
        m = alias.get(t.mnemonic, t.mnemonic)
        if m not in df.columns:
            scores[t.mnemonic] = {"status": "no_las_curve", "las": m}
            continue
        ref = df[m].replace(-999.25, np.nan).dropna()
        if log.depth is not None and t.mnemonic in log.curves:
            td, tv = log.depth, log.curves[t.mnemonic]
            ok = np.isfinite(td) & np.isfinite(tv)
            td, tv = td[ok], tv[ok]
        else:
            td, tv = np.asarray(t.depth, float), np.asarray(t.value, float)
            ok = np.isfinite(td) & np.isfinite(tv)
            td, tv = td[ok], tv[ok]
        if len(td) < 20:
            scores[t.mnemonic] = {"status": "too_few_samples", "n": int(len(td))}
            continue
        # Interpolate reference onto digitized depths only (never bridge PDF page gaps).
        rv = to_grid(ref.index.values, ref.values, td)
        # For log curves, also score in log10 space (wrap-friendly).
        raw = metrics(rv, tv)
        s = align_depth(rv, tv, max_shift=10)
        aligned = metrics(rv, shift_arr(tv, s))
        entry = {
            "las": m,
            "raw_r": raw["r"], "raw_rmse": raw["rmse"], "raw_n": raw["n"],
            "aligned_r": aligned["r"], "aligned_rmse": aligned["rmse"],
            "shift_ft": s * 0.5, "reason": raw.get("reason", ""),
            "kind": t.kind,
        }
        if t.kind == "log":
            mask = np.isfinite(rv) & np.isfinite(tv) & (rv > 0) & (tv > 0)
            if mask.sum() >= 20:
                lr = metrics(np.log10(rv[mask]), np.log10(tv[mask]))
                entry["log10_r"] = lr["r"]
                entry["log10_rmse"] = lr["rmse"]
                entry["log10_n"] = lr["n"]
        scores[t.mnemonic] = entry
        if raw["n"] >= 20 and np.isfinite(raw.get("r", np.nan)):
            qc_plot(
                td, rv, tv,
                str(VIZ / f"{tag}_{t.mnemonic}.png"),
                title=f"{tag} {t.mnemonic}→{m} r={raw['r']:.3f}",
                labels=("reference LAS", "digitized PDF"),
            )
        print(
            f"  {t.mnemonic}->{m}: r={raw['r']} rmse={raw['rmse']} "
            f"n={raw['n']} aligned_r={aligned['r']} shift={s * 0.5}"
            + (f" log10_r={entry.get('log10_r')}" if "log10_r" in entry else "")
        )
    return scores


def run_cbl():
    pdf = DATA / "UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf"
    las = DATA / "other" / "UniversityofUtah_Forge78B-32_6Oct21_CBL_Main_REV1.las"
    pages = [5, 19, 29, 39]  # representative body pages
    print("CBL pages", [p + 1 for p in pages], "(auto legend styles — no mnemonic_colors)")
    log = digitize_pdf(pdf, pages=pages, step=0.5)  # NO hand colours
    styles = {
        h.mnemonic: {"color": h.color, "width": h.width, "dash": h.dash}
        for h in log.headers if h.color is not None
    }
    print("legend styles", styles)
    print("traces", [(t.mnemonic, int(t.depth.min()), int(t.depth.max()),
                      t.info.get("width"), t.info.get("dash")) for t in log.traces])
    las_out = OUT / "forge78B_cbl_digitized.las"
    log.to_las(las_out)
    print("wrote", las_out)
    scores = qc_curves(
        log, las,
        {"ECGR_STGC": "GR_STGC", "TT": "TT", "CBL": "CBL", "MTEM": "MTEM", "TENS": "TENS"},
        "cbl",
    )
    return {
        "pages": [p + 1 for p in pages], "well": log.well, "scores": scores,
        "curves": list(log.curves), "las": str(las_out), "legend_styles": {
            k: {"color": list(v["color"]) if v["color"] else None,
                "width": v["width"], "dash": list(v["dash"]) if v["dash"] else []}
            for k, v in styles.items()
        },
    }


def run_induction():
    pdf = DATA / "forge_78B_graphics" / "UOU_FORGE-78B-32_R1B_TD_SECTION_Induction_30Jul21_070636.Pdf"
    # TD-section MAIN LAS overlaps the induction PDF depths (≈7447–9553).
    las = DATA / "other" / (
        "UOU_FORGE-78B-32_R1B_TD_SECTION_ConPr_R06_L004Up___QAIT_QSLT_QCNT_QTGC_MAIN-GenericV12.las"
    )
    if not las.is_file():
        las = DATA / "other" / "UOU_FORGE-78B-32_R2A_HLDS-QCNT-QAIT_Main Pass-GenericV12.las"
    # Body pages overlapping TD LAS.
    pages = [8, 10, 12, 14]
    print("IND pages", [p + 1 for p in pages], "LAS", las.name)
    log = digitize_pdf(pdf, pages=pages, step=0.5)  # auto legend + log wrap
    styles = {
        h.mnemonic: {"color": h.color, "width": h.width, "dash": h.dash}
        for h in log.headers if h.color is not None
    }
    print("legend styles", styles)
    print("traces", [(t.mnemonic, float(t.depth.min()), float(t.depth.max()),
                      t.kind, t.color) for t in log.traces])
    las_out = OUT / "forge78B_induction_digitized.las"
    if log.traces:
        log.to_las(las_out)
        print("wrote", las_out)
    scores = qc_curves(
        log, las,
        {"ECGR_STGC": "GR_STGC", "AT10": "AT10", "AT20": "AT20", "AT30": "AT30",
         "AT60": "AT60", "AT90": "AT90", "SP": "SP"},
        "ind",
    )
    return {
        "pages": [p + 1 for p in pages], "well": log.well, "scores": scores,
        "curves": list(log.curves), "las": str(las_out) if log.traces else None,
        "legend_styles": {
            k: {"color": list(v["color"]) if v["color"] else None,
                "width": v["width"], "dash": list(v["dash"]) if v["dash"] else []}
            for k, v in styles.items()
        },
        "ref_las": str(las),
    }


def main():
    report = {"cbl": run_cbl(), "induction": run_induction()}

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if o is None or isinstance(o, (str, int, float, bool)):
            return o
        return str(o)

    path = VIZ / "e2e_scores.json"
    path.write_text(json.dumps(conv(report), indent=2))
    print("scores →", path)


if __name__ == "__main__":
    main()
