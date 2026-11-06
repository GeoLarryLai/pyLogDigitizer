#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""E2E: CBL + TD induction → LAS; figures → $PLD_OUT/pdf_vector3/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lasio

from pylogdigitizer.pdf_vector import digitize_pdf
from pylogdigitizer.las_export import read_las, verify_las
from pylogdigitizer.qc_compare import qc_plot
import os
DATA_ROOT = Path(os.environ.get("PLD_DATA", "data"))   # example PDFs / LAS
OUT_ROOT = Path(os.environ.get("PLD_OUT", "pld_out"))   # figures / outputs

VIZ = OUT_ROOT / "pdf_vector3"
VIZ.mkdir(parents=True, exist_ok=True)
DATA = DATA_ROOT
OUT = OUT_ROOT / "artifacts/pdf_vector"
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
        rv = to_grid(ref.index.values, ref.values, td)
        raw = metrics(rv, tv)
        s = align_depth(rv, tv, max_shift=10)
        aligned = metrics(rv, shift_arr(tv, s))
        entry = {
            "las": m,
            "raw_r": raw["r"], "raw_rmse": raw["rmse"], "raw_n": raw["n"],
            "aligned_r": aligned["r"], "aligned_rmse": aligned["rmse"],
            "shift_ft": s * 0.5, "kind": t.kind,
        }
        if t.kind == "log":
            mask = np.isfinite(rv) & np.isfinite(tv) & (rv > 0) & (tv > 0)
            if mask.sum() >= 20:
                lr = metrics(np.log10(rv[mask]), np.log10(tv[mask]))
                entry["log10_r"] = lr["r"]
                entry["log10_rmse"] = lr["rmse"]
        scores[t.mnemonic] = entry
        if raw["n"] >= 20 and np.isfinite(raw.get("r", np.nan)):
            qc_plot(
                td, rv, tv,
                str(VIZ / f"{tag}_{t.mnemonic}.png"),
                title=f"{tag} {t.mnemonic}→{m} r={raw['r']:.3f}",
                labels=("reference LAS", "digitized PDF"),
            )
        print(
            f"  {t.mnemonic}->{m}: r={raw['r']:.4f} rmse={raw['rmse']:.3g} n={raw['n']} "
            f"aligned_r={aligned['r']:.4f}"
            + (f" log10_r={entry['log10_r']:.4f}" if "log10_r" in entry else "")
        )
    return scores


def run_cbl():
    pdf = DATA / "UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf"
    las = DATA / "other" / "UniversityofUtah_Forge78B-32_6Oct21_CBL_Main_REV1.las"
    # Include shallow + mid + deep so MTEM continuity unwrap can anchor on-scale.
    pages = [5, 10, 19, 29, 39]
    print("CBL pages", [p + 1 for p in pages])
    log = digitize_pdf(pdf, pages=pages, step=0.5)
    print("well", log.well)
    print("styles", {h.mnemonic: (h.color, h.width, h.dash) for h in log.headers if h.color})
    las_out = OUT / "forge78B_cbl_digitized.las"
    log.to_las(las_out)
    # Verify round-trip with lasio
    probs = verify_las(las_out, log.depth, log.curves)
    print("wrote", las_out, "verify", probs or "ok")
    back = read_las(las_out)
    print("lasio curves", [c.mnemonic for c in back.curves], "well DATE", getattr(back.well, "DATE", None))
    scores = qc_curves(
        log, las,
        {"ECGR_STGC": "GR_STGC", "TT": "TT", "CBL": "CBL", "MTEM": "MTEM", "TENS": "TENS"},
        "cbl",
    )
    return {
        "pages": [p + 1 for p in pages], "well": log.well, "scores": scores,
        "curves": list(log.curves), "las": str(las_out), "verify": probs,
    }


def run_induction():
    pdf = DATA / "forge_78B_graphics" / "UOU_FORGE-78B-32_R1B_TD_SECTION_Induction_30Jul21_070636.Pdf"
    las = DATA / "other" / (
        "UOU_FORGE-78B-32_R1B_TD_SECTION_ConPr_R06_L004Up___QAIT_QSLT_QCNT_QTGC_MAIN-GenericV12.las"
    )
    pages = [8, 10, 12, 14]
    print("IND pages", [p + 1 for p in pages], "LAS", las.name)
    log = digitize_pdf(pdf, pages=pages, step=0.5)
    print("well", log.well)
    print("styles", {h.mnemonic: (h.color, h.width, h.dash, h.kind) for h in log.headers if h.color})
    print("traces", [(t.mnemonic, float(t.depth.min()), float(t.depth.max()), t.kind) for t in log.traces])
    las_out = OUT / "forge78B_induction_digitized.las"
    log.to_las(las_out)
    probs = verify_las(las_out, log.depth, log.curves)
    print("wrote", las_out, "verify", probs or "ok")
    back = read_las(las_out)
    print("lasio curves", [c.mnemonic for c in back.curves])
    scores = qc_curves(
        log, las,
        {"ECGR_STGC": "GR_STGC", "AT10": "AT10", "AT20": "AT20", "AT30": "AT30",
         "AT60": "AT60", "AT90": "AT90", "SP": "SP", "TENS": "TENS"},
        "ind",
    )
    return {
        "pages": [p + 1 for p in pages], "well": log.well, "scores": scores,
        "curves": list(log.curves), "las": str(las_out), "verify": probs,
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
