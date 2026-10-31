<p align="center"><img src="image/logo.png" alt="pyLogDigitizer" width="200"></p>

# pyLogDigitizer

[![PyPI version](https://img.shields.io/pypi/v/pylogdigitizer.svg)](https://pypi.org/project/pylogdigitizer/) [![pepy downloads](https://static.pepy.tech/badge/pylogdigitizer)](https://pepy.tech/project/pylogdigitizer) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23832013.svg)](https://doi.org/10.5281/zenodo.23832013) [![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](./LICENSE)

v0.1.0

Turns scanned paper well logs (TIFF) into LAS files, with a local browser app to check and fix every step.

Status: early-stage research software. APIs may change; check every digitized curve before use.

## Installation

```bash
pip install pylogdigitizer
pip install "pylogdigitizer[app,fast]"      # optional: browser app and faster tracing
pip install "pylogdigitizer[ml]"            # optional: PyTorch U-Net cost map
```

From source (builds the Rust tracer, needs a Rust toolchain):

```bash
git clone https://github.com/GeoLarryLai/pyLogDigitizer && cd pyLogDigitizer
pip install ".[app,fast]"
```

Or `conda env create -f environment.yml && conda activate pylogdigitizer`, then `maturin develop --release --extras app`.
Without the Rust part everything still works, only slower.

## Citation

Lai, L.S.-H. (2026) pyLogDigitizer. *Zenodo*, [https://doi.org/10.5281/zenodo.23832013](https://doi.org/10.5281/zenodo.23832013)

![From a scanned paper log to a LAS curve](image/hero_scan_to_las.png)

## Key Features

- **Layout and calibration:** finds tracks and grids, deskews, reads depth labels (OCR) and calibrates scales,
  including backup-scale wraparound.
- **Curve tracing:** a fast dynamic-programming tracer (Rust, with a Python fallback); click anchors to re-trace locally.
- **Optional ML cost map:** a U-Net learns from your corrected traces; batch runs send low-confidence curves to review.
- **QC and export:** compares with a reference LAS and writes LAS 2.0. Everything runs locally; nothing is uploaded.

## Examples

![Tracing in the app](image/app_07_trace.png)
*The guided app: a traced curve on the scan, ready to review and fix.*

![Digitized vs reference LAS](image/ex_qc_vs_reference_las.png)
*A digitized curve (assisted workflow) checked against the paired KGS LAS.*

![ML vs conventional cost map](image/ex_ml_vs_conventional.png)
*Held-out Kansas wells: correlation and error against the reference LAS, conventional vs ML cost map.*

Notebooks: `notebooks/01_conventional_vs_assisted.ipynb` (one scan, step by step) and
`notebooks/02_kansas_ml_train_batch.ipynb` (train the cost map and batch-digitize). Example scans and LAS files are
from the [Kansas Geological Survey](https://www.kgs.ku.edu/) (`example_data/SOURCES.txt`).

## Quick start

```bash
pylogdigitizer info example_data/USA_A_16_15-055-20536/4342.tif   # scan size and resolution
pylogdigitizer-app .                        # guided app at http://127.0.0.1 (local only)
pylogdigitizer editor . --open              # web editor (".[web]" extra)
```

The same from Python:

```python
from pylogdigitizer import workflows as wf
R = wf.digitize_curve("example_data/USA_A_16_15-055-20536/4342.tif", track="track1", scale=(0, 150),
                      first_depth=1500, depth_range=(1500, 7030), backup=(150, 300))
R["depth"], R["value"], R["conf"]           # then las_export.resample + write_las
```

`backup=(150, 300)` is the **backup scale**: when a curve runs off the edge of its track it is redrawn from the other
edge on a second scale, and the tracer unwraps it.

Third-party notices and data credits: [`NOTICE`](NOTICE).
