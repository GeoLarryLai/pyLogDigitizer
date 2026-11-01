# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np, pytest
from pylogdigitizer import depth_calibration as dc, qc_compare as qc, workflows as wf
from pylogdigitizer.curve_tracing import bridge_rows


def _lattice(n=8, sp=600.0, r0=1000.0):
    return [r0 + i * sp for i in range(n)]


def test_infer_label_values_outvotes_misreads():
    rows = _lattice()
    vals = [1500, 1600, None, 1800, 8900, 2000, 2100, 2200]
    d0, iv, n = dc.infer_label_values(rows, vals)
    assert (d0, iv) == (1500, 100) and n >= 5


def test_infer_label_values_none_when_unreadable():
    assert dc.infer_label_values(_lattice(), [None] * 8) is None


def test_calibrate_drops_leading_junk_labels():
    # two junk blobs above the first real label (1500): OCR unread -> dropped, depths stay right
    rows = [400.0, 1000.0] + [1600.0 + 600 * i for i in range(8)]
    vals = [None, None] + [1500 + 100 * i for i in range(8)]
    heavy = np.arange(0, 7000, 300.0)
    dm, anc, res, info = dc.calibrate_labels(rows, heavy, vals)
    assert info["source"] == "ocr"
    assert abs(float(dm(1600.0)) - 1500) < 1 and abs(float(dm(4600.0)) - 2000) < 1


def test_calibration_error_without_values_or_first_depth():
    with pytest.raises(dc.CalibrationError):
        dc.calibrate_labels(_lattice(), np.arange(0, 6000, 300.0), None, None)


def test_calibration_error_too_few_labels():
    with pytest.raises(dc.CalibrationError):
        dc.calibrate_labels([100.0, 700.0], [], None, 1000)


def test_validate_rejects_implausible_px_per_ft():
    rows = _lattice(sp=3800)            # 38 px/ft for 100 ft labels
    dm, anc, res, _ = dc.calibrate_labels(rows, [], None, 1000, interval=100, line_ft=0)
    with pytest.raises(dc.CalibrationError, match="implausible"):
        dc.validate_depth_map(dm, anc, res, 40000, n_labels=len(rows))


def test_regular_rows_rejects_junk():
    rows = _lattice() + [1234.0, 3333.0]
    idx = dc.regular_rows(sorted(rows))
    kept = sorted(np.array(sorted(rows))[idx])
    assert kept == _lattice()


def test_qc_compare_nan_reason_no_overlap():
    d = np.arange(1000, 1100, 0.5)
    c = qc.compare(d, np.sin(d), d + 5000, np.sin(d))
    assert np.isnan(c["aligned"]["r"]) and "no depth overlap" in c["aligned"]["reason"]


def test_qc_metrics_constant_reason():
    m = qc.metrics(np.ones(50), np.arange(50.0))
    assert np.isnan(m["r"]) and m["reason"]


def test_bridge_rows_reconnects_vertical_stroke():
    ink = np.zeros((40, 20), bool); ink[:, 10] = True; ink[18:21, :] = True
    rows = np.zeros(40, bool); rows[18:21] = True
    out = bridge_rows(ink, rows)
    assert out[19, 10] and not out[19, 2]


def test_layout_error_on_blank_sheet(tmp_path):
    import cv2
    p = tmp_path / "blank.tif"; cv2.imwrite(str(p), np.full((4000, 1500), 255, np.uint8))
    with pytest.raises((wf.LayoutError, wf.CalibrationError)):
        wf.digitize_curve(str(p), "track1", (0, 150))


def test_ml_cost_falls_back_without_torch(monkeypatch):
    import numpy as np, warnings
    from pylogdigitizer import curve_tracing as ct
    monkeypatch.setattr(ct, "ml_available", lambda: False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ct.ml_fallback("blend") == "conventional"
    assert any("torch" in str(x.message) for x in w)
    panel = np.full((600, 200), 255, np.uint8); panel[:, 95:100] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = ct.trace_track(panel, cost_mode="ml")
    assert t["cost_mode"] == "conventional" and t["ml_prob"] is None
