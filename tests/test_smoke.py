# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import importlib
import numpy as np
import pytest

MODULES = ['scan_io', 'las_export', 'scan_cleanup', 'track_grid_detection', 'depth_calibration', 'scale_calibration', 'header_ocr', 'curve_tracing', 'scale_wraparound', 'ml_training_data', 'ml_models', 'ml_training', 'ml_batch_digitize', 'qc_compare', 'dataset_profiles', 'web_editor', 'assisted_tracing', 'cli']

@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    if name.startswith("ml_") and name != "ml_training_data":
        pytest.importorskip("torch")
    importlib.import_module(f"pylogdigitizer.{name}")

def test_python_tracer_follows_line():
    from pylogdigitizer.curve_tracing import cost_from_ink, trace_dp_python
    ink = np.zeros((50, 30), bool)
    xs = (10 + 5 * np.sin(np.arange(50) / 8)).astype(int)
    ink[np.arange(50), xs] = True
    assert np.array_equal(trace_dp_python(cost_from_ink(ink)), xs)

def test_las_roundtrip(tmp_path):
    from pylogdigitizer.las_export import write_las, read_las
    d = np.arange(1000, 1010, 0.5); gr = np.linspace(20, 120, d.size)
    p = write_las(tmp_path / "t.las", d, {"GR": gr}, units={"GR": "API"})
    las = read_las(p)
    assert np.allclose(las["GR"], gr)

def test_profile_loads():
    from pylogdigitizer.dataset_profiles import load_profile
    assert load_profile("kansas_kgs")["name"] == "kansas_kgs"
