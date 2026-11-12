#!/usr/bin/env python3
"""Digital-print E2E on more USGS PDF+LAS pairs → digital_print_more/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import lasio
from pypdf import PdfReader

from pylogdigitizer.pdf_vector import digitize_pdf, detect_page_layout, annotate_layout_figure
from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.digitize import _log_pages
from pylogdigitizer.las_export import verify_las
from pylogdigitizer.qc_compare import metrics, to_grid
import os
DATA_ROOT = Path(os.environ.get("PLD_DATA", "data"))   # example PDFs / LAS
OUT_ROOT = Path(os.environ.get("PLD_OUT", "pld_out"))   # figures / outputs

VIZ = OUT_ROOT / "digital_print_more"
OUT = Path("artifacts/pdf_vector")
VIZ.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

WELLS = [
    {
        "id": "usgs_18",
        "pdf": DATA_ROOT / "usgs/usgs_18.pdf",
        "las": DATA_ROOT / "usgs/usgs_18.las",
        "alias": {"GAM": "GAM(", "GAMMA": "GAM(", "CALIPER": "CALI", "TEMP": "TEMP:1",
                  "FL": "SP_C:1"},
    },
    {
        "id": "usgs_caville",
        "pdf": DATA_ROOT / "usgs/usgs_100.pdf",
        "las": None,  # LAS 99 not downloadable (403)
        "alias": {},
    },
]


def pick_pages(pdf: Path, max_pages: int = 6) -> list[int]:
    r = PdfReader(str(pdf))
    lp = _log_pages(r)
    if not lp:
        return list(range(min(max_pages, len(r.pages))))
    if len(lp) <= max_pages:
        return lp
    # Prefer early curve pages (legend/styles) + mid + late — avoid only-tail samples.
    picks = [lp[0], lp[1], lp[len(lp)//3], lp[len(lp)//2], lp[(2*len(lp))//3], lp[-1]]
    return list(dict.fromkeys(picks))[:max_pages]


def score_well(log, las_path: Path | None, alias: dict) -> dict:
    if las_path is None or not las_path.exists():
        return {}
    df = lasio.read(str(las_path)).df()
    # Normalize weird usgs columns
    colmap = {c: c for c in df.columns}
    scores = {}
    for t in log.traces:
        m = alias.get(t.mnemonic, t.mnemonic)
        if m not in df.columns:
            # fuzzy
            hits = [c for c in df.columns if m.upper().replace("(", "") in c.upper()]
            if not hits:
                continue
            m = hits[0]
        ref = df[m].replace([-999.25, -99999, -9999], np.nan).dropna()
        if t.mnemonic not in log.curves:
            continue
        ok = np.isfinite(log.depth) & np.isfinite(log.curves[t.mnemonic])
        td, tv = log.depth[ok], log.curves[t.mnemonic][ok]
        if len(td) < 20 or len(ref) < 20:
            continue
        rv = to_grid(ref.index.values, ref.values, td)
        sc = metrics(rv, tv)
        if sc.get("n", 0) >= 20:
            scores[t.mnemonic] = {"ref": m, **{k: sc[k] for k in ("r", "rmse", "n", "bias")}}
    return scores


def overlay_fig(log, las_path, alias, out_png, title):
    if not las_path or not las_path.exists():
        return
    df = lasio.read(str(las_path)).df()
    keys = [t for t in log.traces if alias.get(t.mnemonic, t.mnemonic) in df.columns
            or any(alias.get(t.mnemonic, t.mnemonic).upper().replace("(", "") in c.upper() for c in df.columns)]
    keys = keys[:6] or log.traces[:4]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 7), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, t in zip(axes, keys):
        m = alias.get(t.mnemonic, t.mnemonic)
        if m not in df.columns:
            hits = [c for c in df.columns if m.upper().replace("(", "") in c.upper()]
            m = hits[0] if hits else None
        if m is None or t.mnemonic not in log.curves:
            ax.set_visible(False)
            continue
        ref = df[m].replace([-999.25, -99999], np.nan).dropna()
        ok = np.isfinite(log.depth) & np.isfinite(log.curves[t.mnemonic])
        td, tv = log.depth[ok], log.curves[t.mnemonic][ok]
        ax.plot(ref.values, ref.index, color="0.65", lw=0.5, label="LAS")
        ax.plot(tv, td, color="C0", lw=0.7, label="dig")
        ax.invert_yaxis()
        ax.set_title(t.mnemonic, fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("depth (ft)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    table = {}
    for w in WELLS:
        wid, pdf = w["id"], w["pdf"]
        print(f"\n=== {wid} ===")
        pages = pick_pages(pdf)
        print("pages", pages)
        log = digitize_pdf(pdf, pages=pages, step=0.5)
        print("curves", list(log.curves))
        las_out = OUT / f"{wid}_digitized.las"
        if log.curves:
            log.to_las(las_out)
            print("LAS", las_out, verify_las(las_out, log.depth, log.curves) or "ok")
        scores = score_well(log, w["las"], w["alias"])
        table[wid] = {
            "pages": pages,
            "curves": list(log.curves),
            "headers": [h.mnemonic for h in log.headers],
            "scores": scores,
            "well": log.well,
            "las_ref": str(w["las"]) if w["las"] else None,
        }
        print("scores", {k: round(v["r"], 4) for k, v in scores.items()})

        # Regions figure from a mid page
        reader = PdfReader(str(pdf))
        mid = pages[len(pages) // 2]
        pv = extract_page(reader, mid)
        # use traces from that page via layouts if present
        layout = next((L for L in log.layouts if L.page_index == mid), None)
        if layout is None:
            layout = detect_page_layout(pv, traces=log.traces)
        fig = annotate_layout_figure(
            pv, layout, traces=log.traces,
            out_path=str(VIZ / f"{wid}_p{mid}_regions.png"),
            title=f"{wid} p{mid} regions",
        )
        plt.close(fig)
        (VIZ / f"{wid}_regions.json").write_text(json.dumps(layout.as_dict(), indent=2, default=str))
        overlay_fig(log, w["las"], w["alias"], VIZ / f"{wid}_overlay.png", f"{wid} vs LAS")
        (OUT / f"{wid}_digitized.regions.json").write_text(json.dumps({
            "layouts": [L.as_dict() for L in log.layouts],
            "scores": scores,
        }, indent=2, default=str))

    (VIZ / "scores_table.json").write_text(json.dumps(table, indent=2, default=str))
    # markdown-ish plaintext table
    lines = ["well,curve,r,rmse,n,ref"]
    for wid, info in table.items():
        for c, sc in info["scores"].items():
            lines.append(f"{wid},{c},{sc['r']:.4f},{sc['rmse']:.4g},{sc['n']},{sc['ref']}")
        if not info["scores"]:
            lines.append(f"{wid},(none),,,,")
    (VIZ / "scores_table.csv").write_text("\n".join(lines) + "\n")
    print("\nTABLE\n" + "\n".join(lines))
    print("figs →", VIZ)


if __name__ == "__main__":
    main()
