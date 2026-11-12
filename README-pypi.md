# pyLogDigitizer

pyLogDigitizer turns well-log image scans (TIFF, PNG, JPEG or a scanned PDF) into LAS files. It finds the tracks, grid and depth numbers, reads the curve header (names and scales) with OCR, tells the curves apart (by ink colour on colour raster scans, by line style on black-and-white scans) and traces each one. Off-scale values are read on the backup scale where the overflow wraps back in from the opposite edge, and written as null otherwise. It first tells whether a file is an image scan, a vector PDF or a mixed file; digital-print PDFs that still carry vector curves are read exactly, without tracing (experimental).

A local browser app in five tabs (open scan; tracks and depth; curves; trace and fix; check and save) supports review and correction before LAS 2.0 export at 0.5 ft by default, or any depth step you type. It works best on colour scans; black-ink scans carry higher uncertainty. All processing remains on the user's own computer.

**Caution:** The package is under active development. Many functions are experimental and not fully validated; review overlays and exported curves before using the results.

Installation details, examples, and notebooks are in the [GitHub repository](https://github.com/GeoLarryLai/pyLogDigitizer).

Citation: Lai, L. S.-H. (2026). pyLogDigitizer. [Zenodo](https://doi.org/10.5281/zenodo.23832013).

License: AGPL-3.0-or-later. Third-party credits are recorded in NOTICE.
