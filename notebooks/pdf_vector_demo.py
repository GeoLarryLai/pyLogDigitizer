# %%
# pdf_vector brief flow: legend styles → digitize → continuity unwrap → LAS.
# Run: PYTHONPATH=src python notebooks/pdf_vector_demo.py
# Or open as a notebook (jupytext / VS Code cells).
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from pylogdigitizer.pdf_vector import digitize_pdf
from pylogdigitizer.las_export import read_las, verify_las
from pylogdigitizer.qc_compare import metrics, to_grid

DATA = Path("example_data/pdf_vector")
DATA.mkdir(parents=True, exist_ok=True)
CBL_URL = "https://gdr.openei.org/files/1330/78B-32_CBL_LOG.zip"
ZIP = DATA / "78B-32_CBL_LOG.zip"
PDF_NAME = "UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf"

if not (DATA / PDF_NAME).exists():
    if not ZIP.exists():
        print("downloading", CBL_URL)
        urllib.request.urlretrieve(CBL_URL, ZIP)
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        for n in zf.namelist():
            if n.lower().endswith((".pdf", ".las")):
                zf.extract(n, DATA)
                (DATA / n).rename(DATA / Path(n).name)
    print("extracted to", DATA)

pdf = DATA / PDF_NAME
# Auto legend styles (colour+width+dash) — no hand mnemonic_colors required.
# Pages span shallow→deep so MTEM linear wrap unwraps by continuity.
log = digitize_pdf(pdf, pages=[5, 10, 19, 29], step=0.5)
print("well", log.well)
print("legend", {
    h.mnemonic: {"color": h.color, "width": h.width, "dash": h.dash,
                 "scale": f"{h.v_left}-{h.v_right} {h.unit}"}
    for h in log.headers if h.color is not None
})
print("curves", {
    t.mnemonic: (round(float(t.depth.min()), 1), round(float(t.depth.max()), 1), int(t.info["n_pts"]))
    for t in log.traces
})
out = DATA / "forge78B_cbl_pages_digitized.las"
log.to_las(out)
print("wrote", out, "verify", verify_las(out, log.depth, log.curves) or "ok")
print("lasio reload", [c.mnemonic for c in read_las(out).curves])

las_path = next(DATA.glob("*CBL*Main*.las"), None) or next(DATA.glob("*.las"), None)
if las_path and las_path.resolve() != out.resolve():
    import lasio
    df = lasio.read(str(las_path)).df()
    alias = {"ECGR_STGC": "GR_STGC", "TT": "TT", "CBL": "CBL", "MTEM": "MTEM"}
    for t in log.traces:
        m = alias.get(t.mnemonic)
        if not m or m not in df.columns or t.mnemonic not in log.curves:
            continue
        ref = df[m].replace(-999.25, np.nan).dropna()
        ok = np.isfinite(log.depth) & np.isfinite(log.curves[t.mnemonic])
        td, tv = log.depth[ok], log.curves[t.mnemonic][ok]
        if len(td) < 20:
            continue
        rv = to_grid(ref.index.values, ref.values, td)
        print(t.mnemonic, metrics(rv, tv))
