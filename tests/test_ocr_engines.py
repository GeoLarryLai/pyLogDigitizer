# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""OCR engine: RapidOCR or off (manual / semi-automatic fallback), install hints, real reads."""
import numpy as np
import pytest
from pylogdigitizer import ocr_engines as oe


@pytest.fixture(autouse=True)
def _reset():
    oe.set_default(None); yield; oe.set_default(None)


def test_auto_and_off(monkeypatch):
    monkeypatch.setattr(oe, "available", lambda e="rapidocr": True)
    assert oe.resolve("auto") == "rapidocr" and oe.status()["default"] == "rapidocr"
    oe.set_default("none"); assert oe.resolve() is None
    oe.set_default(False); assert oe.resolve() is None
    oe.set_default("auto"); assert oe.resolve() == "rapidocr"
    monkeypatch.setenv("PYLOGDIGITIZER_OCR", "none"); assert oe.resolve() is None
    monkeypatch.setattr(oe, "available", lambda e="rapidocr": False)
    monkeypatch.delenv("PYLOGDIGITIZER_OCR")
    assert oe.resolve("auto") is None and oe.status()["rapidocr"]["hint"]
    with pytest.raises(RuntimeError, match="rapidocr_onnxruntime"):
        oe.resolve("rapidocr")
    for bad in ("tesseract", "easyocr9"):
        with pytest.raises(ValueError):
            oe.set_default(bad)
    assert oe.ENGINES == ("rapidocr",)


def test_no_engine_depth_values_none(monkeypatch):
    from pylogdigitizer import depth_calibration as dc
    monkeypatch.setattr(oe, "available", lambda e="rapidocr": False)
    labels = [dict(row=10, x0=0, x1=5, y0=5, y1=15)]
    assert dc.ocr_depth_labels(np.full((30, 30), 255, np.uint8), labels) == [(10, None)]
    assert not dc.ocr_available()


def _label(text, h=40, w=140):
    import cv2
    img = np.full((h, w), 255, np.uint8)
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return img


def test_real_engine_reads_depth_labels(engine="rapidocr"):
    if not oe.available(engine):
        pytest.skip(f"{engine} not installed")
    from pylogdigitizer import depth_calibration as dc
    col = np.vstack([_label("1500"), np.full((60, 140), 255, np.uint8), _label("1600")])
    labels = [dict(row=20, x0=0, x1=140, y0=0, y1=40), dict(row=120, x0=0, x1=140, y0=100, y1=140)]
    vals = [v for _, v in dc.ocr_depth_labels(col, labels, rotations=(0,), engine=engine)]
    assert vals == [1500.0, 1600.0]


def test_rapidocr_header_words():
    if not oe.available("rapidocr"):
        pytest.skip("rapidocr not installed")
    from pylogdigitizer import header_ocr as ho
    img = np.full((60, 300), 255, np.uint8)
    import cv2
    cv2.putText(img, "GAMMA RAY 0 150", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    words = ho.ocr_region(img, backend="rapidocr", remove_lines=False, offset=(100, 10))
    txt = " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"])).upper()
    assert "GAMMA" in txt and "150" in txt and min(w["x0"] for w in words) >= 100
