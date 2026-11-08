# pyLogDigitizer

pyLogDigitizer turns well-log prints into LAS files. It supports inked paper scans (TIFF) and clean colour digital-print PDFs through shared scale, depth, null-value, and LAS export code.

The scan workflow finds tracks and grids, reads depth labels, and traces curves. The digital-print workflow reads coloured vector paths directly. Overlay quality checks and a local browser editor support review and correction, including backup-scale unwrapping, before LAS 2.0 export. All processing remains on the user's own computer.

**Caution:** The package is under active development. Many functions are experimental and not fully validated; review overlays and exported curves before using the results.

Installation details, examples, and notebooks are in the [GitHub repository](https://github.com/GeoLarryLai/pyLogDigitizer).

Citation: Lai, L. S.-H. (2026). pyLogDigitizer. [Zenodo](https://doi.org/10.5281/zenodo.23832013).

License: AGPL-3.0-or-later. Third-party credits are recorded in NOTICE.
