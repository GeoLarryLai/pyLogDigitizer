# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""End-to-end check on the paired KGS example (skipped when example_data is absent)."""
import pathlib
import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "example_data" / "MOBIL_12-1_15-189-20087"


def _tess():
    from pylogdigitizer.depth_calibration import ocr_available
    return ocr_available()


@pytest.mark.skipif(not (DATA / "227233.tif").exists(), reason="example_data not present")
@pytest.mark.skipif(not _tess(), reason="needs RapidOCR (OCR reads the 4100 ft first label; config 4000 is the fallback)")
def test_mobil_dt_matches_las():
    from pylogdigitizer import workflows as wf, qc_compare as qc
    from pylogdigitizer.las_export import read_las
    R = wf.digitize_curve(DATA / "227233.tif", "track2", (100, 40), 4000, (3997, 5707), width=740,
                          backup=(160, 100), wrap_side="left", use_hatch=False)
    ref = read_las(DATA / "1044084189.las").df()["DT"].dropna()
    c = qc.compare(ref.index.values, ref.values, R["depth"], R["value"])
    assert c["aligned"]["r"] > 0.85 and abs(c["shift_ft"]) <= 2
