# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyLogDigitizer: semi-automatic digitization of raster well-log scans."""
__version__ = "0.1.1"


def _openmp_guard() -> None:
    """macOS + conda: a pip-installed torch wheel bundles its own libomp.dylib while conda's OpenBLAS
    (openmp build) / numba load conda's llvm-openmp libomp. Two OpenMP runtimes in one process abort
    with "OMP: Error #15". Allow the duplicate only in exactly that situation, before torch loads.
    Not needed for pure-pip installs (numpy/scipy wheels use pthreads OpenBLAS) or conda-forge torch."""
    import os, sys
    if sys.platform != "darwin" or "KMP_DUPLICATE_LIB_OK" in os.environ or "torch" in sys.modules:
        return
    from pathlib import Path
    import importlib.util
    env_omp = [Path(sys.prefix, "lib", n) for n in ("libomp.dylib", "libiomp5.dylib")]
    if not any(p.exists() for p in env_omp):
        return
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin and Path(spec.origin).parent.joinpath("lib", "libomp.dylib").exists():
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


_openmp_guard()
MODULES = ['scan_io', 'las_export', 'scan_cleanup', 'track_grid_detection', 'depth_calibration', 'scale_calibration', 'header_ocr', 'curve_tracing', 'scale_wraparound', 'ml_training_data', 'ml_models', 'ml_training', 'ml_batch_digitize', 'qc_compare', 'dataset_profiles', 'web_editor', 'cli']
