# %%
# Short demo: digitize a public FORGE vector PDF page → LAS (shared scale/depth/LAS helpers).
# Run from repo root with: PYTHONPATH=src python notebooks/pdf_vector_demo.py
# Or open as a notebook (jupytext / VS Code cells).
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from pylogdigitizer.pdf_vector import digitize_pdf
from pylogdigitizer.qc_compare import metrics, to_grid, qc_plot

DATA = Path("example_data/pdf_vector")
DATA.mkdir(parents=True, exist_ok=True)
# Public Utah FORGE GDR — CBL vector PDF (do not commit the download).
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
            if n.lower().endswith(".pdf") or n.lower().endswith(".las"):
                zf.extract(n, DATA)
                # flatten
                src = DATA / n
                src.rename(DATA / Path(n).name)
    print("extracted to", DATA)

pdf = DATA / PDF_NAME
# Two mid-well pages; green=GR, blue=TT, black=CBL (Schlumberger colours on this plot).
COLORS = {"ECGR_STGC": (0.0, 0.75, 0.0), "TT": (0.0, 0.0, 1.0), "CBL": (0.0, 0.0, 0.0)}
log = digitize_pdf(pdf, pages=[19, 29], step=0.5, mnemonic_colors=COLORS)
print("well", log.well)
print("curves", {t.mnemonic: (float(t.depth.min()), float(t.depth.max()), int(t.info["n_pts"])) for t in log.traces})
out = DATA / "forge78B_cbl_pages_digitized.las"
log.to_las(out)
print("wrote", out)

# Optional QC if the companion LAS was in the zip
las_path = next(DATA.glob("*CBL*Main*.las"), None) or next(DATA.glob("*.las"), None)
if las_path:
    import lasio
    df = lasio.read(str(las_path)).df()
    alias = {"ECGR_STGC": "GR_STGC", "TT": "TT", "CBL": "CBL"}
    for t in log.traces:
        m = alias.get(t.mnemonic)
        if not m or m not in df.columns:
            continue
        ref = df[m].replace(-999.25, np.nan).dropna()
        ok = np.isfinite(log.depth) & np.isfinite(log.curves[t.mnemonic])
        td, tv = log.depth[ok], log.curves[t.mnemonic][ok]
        rv = to_grid(ref.index.values, ref.values, td)
        print(t.mnemonic, metrics(rv, tv))
