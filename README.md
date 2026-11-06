<p align="center"><img src="image/logo.png" alt="pyLogDigitizer" width="220"></p>
<p align="center">
<a href="https://doi.org/10.5281/zenodo.23832013"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23832013-blue" alt="DOI"></a> <a href="https://github.com/GeoLarryLai/pyLogDigitizer/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-lightgrey" alt="License"></a>
<br>
<a href="https://pypi.org/project/pylogdigitizer/"><img src="https://img.shields.io/badge/python-3.10%2B-lightgrey" alt="Python versions"></a> <a href="https://pypi.org/project/pylogdigitizer/"><img src="https://img.shields.io/pypi/v/pylogdigitizer?color=orange" alt="PyPI version"></a> <a href="https://pepy.tech/projects/pylogdigitizer"><img src="https://img.shields.io/pepy/dt/pylogdigitizer?label=pypi%20downloads&color=orange" alt="PyPI downloads"></a> <a href="https://anaconda.org/conda-forge/pylogdigitizer"><img src="https://img.shields.io/conda/vn/conda-forge/pylogdigitizer?label=conda-forge&color=green" alt="conda-forge version"></a> <a href="https://anaconda.org/conda-forge/pylogdigitizer"><img src="https://img.shields.io/conda/dn/conda-forge/pylogdigitizer?label=conda%20downloads&color=green" alt="conda downloads"></a>
</p>

# pyLogDigitizer

> [!WARNING]
> **pyLogDigitizer** is under active development, and many functions are still experimental and not fully validated. Please use with caution.

**pyLogDigitizer** turns scanned paper well logs (TIFF) and clean colour digital-print PDF logs into LAS files. Lots of older wells only exist as paper scans, and digitizing them by hand is slow and hard to check. So I built a tool that does the tedious parts — finding tracks and grids, reading depth labels, tracing curves — and gives you a local browser app to check and fix every step. The same scale, depth, null and LAS code is shared by both input paths. Curves that run off the track and come back on a backup scale get unwrapped automatically. Overlay QA draws the grids and traces on the original scan or the real rendered PDF page so you can check alignment by eye. Everything runs on your own machine and nothing is uploaded.

## Installation

```bash
pip install pylogdigitizer
```

From source (builds the Rust tracer, needs a Rust toolchain):

```bash
git clone https://github.com/GeoLarryLai/pyLogDigitizer && cd pyLogDigitizer
pip install .
```

Or `conda env create -f environment.yml && conda activate pylogdigitizer`, then `maturin develop --release`.
Without the Rust part everything still works, only slower.

## Citation

Lai, L.S.-H. (2026) pyLogDigitizer. *Zenodo*, [https://doi.org/10.5281/zenodo.23832013](https://doi.org/10.5281/zenodo.23832013)

![From a scanned paper log to a LAS curve](image/hero_scan_to_las.png)
*The whole idea in one picture: a scanned paper log going in and a LAS curve coming out.*

## What it can do

- Two input paths that share the same scale / depth / null / LAS code:
  - **Inked paper scans (TIFF)** — find tracks and grids, deskew, OCR depth labels, calibrate scales (including backup-scale wraparound), and trace curves with a fast dynamic-programming tracer (Rust, with a Python fallback)
  - **Clean colour digital-print PDFs** — read vector paths straight from the PDF with `pypdf` (`pylogdigitizer.pdf_vector.digitize_pdf`); no rasterizing. Auto-identifies the header, upper/lower scale sections and grids per track/column, and digitizes each curve separately
- Overlay QA: identified grid lines and traced curves drawn on the original scan or the real rendered PDF page, so you can check alignment by eye (not only LAS scores)
- Backup-scale wrap / off-chart overflow: when a curve runs off a track and reappears on the opposite edge it is unwrapped onto the backup scale; overflow pieces are valued correctly and drawn where they sit on the print. Nulls where a curve is clipped by the track’s fixed scale. Log scales supported. LAS step 0.5 ft by default, 0.25 ft optional
- Bundled ONNX scan segmenter (~a few MB): default `scan_seg_v2`; `scan_seg_overflow` is opt-in for inked scans with overflow spikes. Meant as a general scan model — you shouldn’t need to retrain per print. Clean colour digital-print PDFs don’t use this model. Optional retraining: `notebooks/03_train_scan_segmenter.ipynb` writes to your own folder and doesn’t overwrite the bundled default
- Manual fix in the local browser app: click the real curve to pin it, drag a pin, re-trace just the part inside a cursor circle, or drop a node (**D**) that snaps to potential curve ink inside an adjustable-diameter circle (or force the clicked position) and reconnects the curve above and below instantly with no model rerun
- Nudge track edges, set a scale from two grid lines, and click any OCR word in the header to correct it (the scale is re-fitted)
- Optionally train a U-Net cost map on your own corrected traces; batch runs send low-confidence curves to review
- Compare results against a reference LAS and write LAS 2.0
- Work through one local browser app, the command line or Python

## Examples

![Chitwood 12 scan overlay](image/overlay_scan_chitwood.png)
*Chitwood 12 (KGS): traced GR and snapped grid lines drawn on the original scan, so you can check that the grid sits on the printed lines in every track.*

![Minzer 1 wrap overlay](image/overlay_scan_minzer_wrap.png)
*Minzer 1 (KGS): high-value overflow is read on the backup scale and drawn where the backup stroke actually sits on the print.*

![Utah FORGE CBL digital-print overlay](image/overlay_pdf_forge_cbl.jpg)
*Utah FORGE 78B-32 CBL, page 7: digitized curves and the PDF’s own grid lines drawn on the rendered colour page.*

![Node editor](image/node_editor.png)
*Node editor: drop a node (D) that snaps to nearby ink inside an adjustable circle and reconnects the curve instantly, without rerunning the model.*

Scan examples are from the [Kansas Geological Survey](https://www.kgs.ku.edu/) (`example_data/SOURCES.txt`). The digital-print example is the public Utah FORGE 78B-32 CBL ([GDR / OpenEI](https://gdr.openei.org/)).

Notebooks: `notebooks/pdf_vector_demo.ipynb` (digital-print PDF → LAS), `notebooks/03_train_scan_segmenter.ipynb` (optional ONNX retrain into your own folder), plus `notebooks/01_conventional_vs_assisted.ipynb` and `notebooks/02_kansas_ml_train_batch.ipynb` for the scan / cost-map workflow.

On some multi-curve digital prints a GR piece may be only partly captured (FORGE porosity is the example); messy multi-curve inked scans are meant to be fixed in the editor rather than fully automatically.

## Quick start

```bash
pylogdigitizer info example_data/USA_A_16_15-055-20536/4342.tif   # scan size and resolution
pylogdigitizer-app .                        # the app at http://127.0.0.1 (local only); `pylogdigitizer editor .` is an alias
```

Scan path from Python:

```python
from pylogdigitizer import workflows as wf
R = wf.digitize_curve("example_data/USA_A_16_15-055-20536/4342.tif", track="track1", scale=(0, 150),
                      first_depth=1500, depth_range=(1500, 7030), backup=(150, 300))
R["depth"], R["value"], R["conf"]           # then las_export.resample + write_las
```

`backup=(150, 300)` is the **backup scale**: when a curve runs off the edge of its track it is redrawn from the other edge on a second scale, and the tracer unwraps it.

Digital-print PDF path (pages are 0-based; step 0.5 ft is the default):

```python
from pylogdigitizer.pdf_vector import digitize_pdf

log = digitize_pdf("UniversityofUtah_Forge78B-32_6Oct21_CBL.Pdf", step=0.5)
log.to_las("forge_cbl.las")
```

The optional ML cost map (`cost_mode="ml"` or `"blend"`) needs PyTorch (`pip install torch`) and a trained checkpoint. Without them the tracer uses the conventional cost map.

Third-party notices and data credits: [`NOTICE`](NOTICE).
