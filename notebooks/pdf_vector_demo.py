# %% digital_print demo — colour PDF → regions → LAS (pypdf; not ink scans)
# Figs: /workspace/pld_viz_out/digital_print_demo/
# Run: PYTHONPATH=src python notebooks/pdf_vector_demo.py
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pylogdigitizer.pdf_vector import (
    digitize_pdf,
    detect_page_layout,
    annotate_layout_figure,
    OFF_CHART_WRAP_NOTE,
)
from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.assemble import traces_from_page
from pylogdigitizer.las_export import read_las, verify_las
from pylogdigitizer.qc_compare import metrics, to_grid
from pypdf import PdfReader

DATA = Path("example_data/pdf_vector")
DATA.mkdir(parents=True, exist_ok=True)
VIZ = Path("/workspace/pld_viz_out/digital_print_demo")
VIZ.mkdir(parents=True, exist_ok=True)

CBL_URL = "https://gdr.openei.org/files/1330/78B-32_CBL_LOG.zip"
ZIP = DATA / "78B-32_CBL_LOG.zip"
PDF_NAME = "UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf"

# %% 1) Download FORGE CBL PDF + reference LAS
if not (DATA / PDF_NAME).exists():
    if not ZIP.exists():
        print("downloading", CBL_URL)
        urllib.request.urlretrieve(CBL_URL, ZIP)
    with zipfile.ZipFile(ZIP) as zf:
        for n in zf.namelist():
            if n.lower().endswith((".pdf", ".las")):
                zf.extract(n, DATA)
                (DATA / n).rename(DATA / Path(n).name)
    print("extracted →", DATA)
else:
    print("using cached", DATA / PDF_NAME)

pdf = DATA / PDF_NAME
reader = PdfReader(str(pdf))
print("pages", len(reader.pages))

# %% 2) Digitize (auto legend + wrap continuity)
PAGES = [5, 10, 19, 20, 29]
log = digitize_pdf(pdf, pages=PAGES, step=0.5)
print("well", log.well)
print("layouts", [(L.page_index, L.axis, len(L.regions)) for L in log.layouts])
print("curves", {
    t.mnemonic: (round(float(t.depth.min()), 1), round(float(t.depth.max()), 1),
                 int(t.info.get("n_pts", 0)))
    for t in log.traces
})
assert log.layouts, "DigitizedLog.layouts must be filled"

# %% 3) Auto regions: header, upper/lower scale + wrap note, grids, curves
HEADER_PAGE, SCALE_PAGE, MID_PAGE = 0, 4, 20

pv0 = extract_page(reader, HEADER_PAGE)
lay0 = detect_page_layout(pv0)
fig = annotate_layout_figure(
    pv0, lay0, out_path=str(VIZ / "01_header.png"),
    kinds={"header"}, title="FORGE CBL p0 — well header",
)
plt.close(fig)
print("p0", [(r.kind, r.label) for r in lay0.regions])

pv4 = extract_page(reader, SCALE_PAGE)
tr4 = traces_from_page(pv4, headers=log.headers)
lay4 = detect_page_layout(pv4, traces=tr4)
# Confirm wrap note sits on scale_legend
for r in lay4.regions:
    if r.kind == "scale_legend":
        print("scale_legend", r.label, "wrap_note=", bool(r.info.get("off_chart_wrap_note")))
        print(" ", (r.info.get("off_chart_wrap_note") or "")[:120], "...")
fig = annotate_layout_figure(
    pv4, lay4, traces=tr4, out_path=str(VIZ / "02_scale_upper_lower_wrap.png"),
    kinds={"scale_legend", "header", "depth_axis"},
    title="FORGE CBL p4 — upper/lower scale (wrap note on figure)",
)
plt.close(fig)
fig = annotate_layout_figure(
    pv4, lay4, traces=tr4, out_path=str(VIZ / "03_tracks_grids.png"),
    kinds={"track", "depth_axis", "scale_legend"},
    title="FORGE CBL p4 — track grids",
)
plt.close(fig)

pv20 = extract_page(reader, MID_PAGE)
tr20 = traces_from_page(pv20, headers=log.headers)
lay20 = detect_page_layout(pv20, traces=tr20)
fig = annotate_layout_figure(
    pv20, lay20, traces=tr20, out_path=str(VIZ / "04_curves_overview.png"),
    kinds={"curve", "track", "depth_axis", "scale_legend"},
    title="FORGE CBL p20 — curves digitized separately",
)
plt.close(fig)
for t in tr20:
    fig = annotate_layout_figure(
        pv20, lay20, traces=[t],
        out_path=str(VIZ / f"05_curve_{t.mnemonic}.png"),
        kinds={"curve", "track", "depth_axis"},
        title=f"p20 {t.mnemonic}",
    )
    plt.close(fig)
print("p20 curves", [t.mnemonic for t in tr20])

(VIZ / "regions_p0_p4_p20.json").write_text(json.dumps({
    "wrap_note_canonical": OFF_CHART_WRAP_NOTE,
    "p0": lay0.as_dict(), "p4": lay4.as_dict(), "p20": lay20.as_dict(),
}, indent=2, default=str))

# %% 4) Write LAS
out = DATA / "forge78B_cbl_digitized.las"
log.to_las(out)
print("wrote", out, "verify", verify_las(out, log.depth, log.curves) or "ok")
print("lasio", [c.mnemonic for c in read_las(out).curves])
(DATA / "forge78B_cbl_digitized.regions.json").write_text(json.dumps({
    "layouts": [L.as_dict() for L in log.layouts],
    "well": log.well,
}, indent=2, default=str))

# %% 5) Overlay vs reference LAS
ref_path = next(DATA.glob("*CBL*Main*.las"), None) or next(
    (p for p in DATA.glob("*.las") if p.resolve() != out.resolve()), None
)
alias = {"ECGR_STGC": "GR_STGC", "TT": "TT", "CBL": "CBL", "MTEM": "MTEM", "TENS": "TENS"}
scores = {}
if ref_path is not None:
    import lasio
    df = lasio.read(str(ref_path)).df()
    n = max(1, len(log.traces))
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 7), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, log.traces):
        m = alias.get(t.mnemonic)
        if not m or m not in df.columns or t.mnemonic not in log.curves:
            ax.set_visible(False)
            continue
        ref = df[m].replace(-999.25, np.nan).dropna()
        ok = np.isfinite(log.depth) & np.isfinite(log.curves[t.mnemonic])
        td, tv = log.depth[ok], log.curves[t.mnemonic][ok]
        if len(td) < 20:
            continue
        rv = to_grid(ref.index.values, ref.values, td)
        sc = metrics(rv, tv)
        scores[t.mnemonic] = sc
        ax.plot(ref.values, ref.index.values, color="0.65", lw=0.6, label="LAS")
        ax.plot(tv, td, color="C0", lw=0.8, label="digitized")
        ax.invert_yaxis()
        ax.set_title(f"{t.mnemonic}\nr={sc['r']:.3f}", fontsize=9)
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel("depth (ft)")
    fig.suptitle("FORGE CBL digital-print vs reference LAS", fontsize=11)
    fig.tight_layout()
    fig.savefig(VIZ / "06_overlay_vs_las.png", dpi=130)
    plt.close(fig)
    print("scores", {k: round(v["r"], 4) for k, v in scores.items()})
else:
    print("no reference LAS found next to PDF")

(VIZ / "demo_scores.json").write_text(json.dumps({
    "well": log.well, "pages": PAGES, "scores": scores, "las": str(out),
}, indent=2, default=str))
print("figs →", VIZ)
