# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ocr_engines: the OCR engine that reads header text and depth labels.

One engine: ``rapidocr`` - RapidOCR 3.x (``rapidocr`` + ``onnxruntime``) running the PP-OCRv4 mobile
det/rec and v2.0 cls ONNX models bundled in ``ocr_models/`` (no model download, no system program). Tesseract was compared on the Kansas
scans (same accuracy, ~9x slower, needs a system install) and dropped.

``engine='auto'`` = RapidOCR if importable, else None (no OCR: depth falls back to first label + interval,
header rows are typed by hand). ``set_default(False/'none')`` or env ``PYLOGDIGITIZER_OCR=none`` turns OCR off.
"""
from __future__ import annotations
import importlib.util, os, threading
import numpy as np

ENGINES = ("rapidocr",)
HINTS = {"rapidocr": "pip install rapidocr onnxruntime   (included in: pip install pylogdigitizer)"}
_MODELS = os.path.join(os.path.dirname(__file__), "ocr_models")
_DEFAULT: str | None = None
_LOCK = threading.Lock()
_RAPID = None


def available(engine: str = "rapidocr") -> bool:
    return engine == "rapidocr" and (importlib.util.find_spec("rapidocr") is not None
                                   and importlib.util.find_spec("onnxruntime") is not None)


def status() -> dict:
    """{engine: {ok, hint}} plus 'default' = what 'auto' resolves to (None = no OCR)."""
    d = {e: dict(ok=available(e), hint=None if available(e) else HINTS[e]) for e in ENGINES}
    d["default"] = resolve("auto")
    return d


def set_default(engine: str | None) -> None:
    """Process-wide choice for engine='auto' ('auto'/None resets to auto-detect)."""
    global _DEFAULT
    if engine is False:
        engine = "none"
    if engine not in (None, "auto", "none", *ENGINES):
        raise ValueError(f"unknown OCR engine {engine!r} (auto, none, {', '.join(ENGINES)})")
    _DEFAULT = None if engine in (None, "auto") else engine


def resolve(engine: str | None = "auto") -> str | None:
    """Engine name to use, or None when nothing usable is installed. An explicit engine that is
    not installed raises RuntimeError with the install hint (never silently swapped)."""
    if engine in (None, "auto"):
        engine = _DEFAULT or os.environ.get("PYLOGDIGITIZER_OCR") or "auto"
    if engine == "none":
        return None
    if engine == "auto":
        return next((e for e in ENGINES if available(e)), None)
    if engine not in ENGINES:
        raise ValueError(f"unknown OCR engine {engine!r} (auto, none, {', '.join(ENGINES)})")
    if not available(engine):
        raise RuntimeError(f"OCR engine {engine!r} is not installed: {HINTS[engine]}")
    return engine


def any_available() -> bool:
    return resolve("auto") is not None


# ------------------------------------------------------------------ RapidOCR
def _rapid():
    global _RAPID
    with _LOCK:
        if _RAPID is None:
            from rapidocr import RapidOCR
            m = lambda f: os.path.join(_MODELS, f)
            _RAPID = RapidOCR(params={"Global.log_level": "warning",
                                      "Det.model_path": m("ch_PP-OCRv4_det_infer.onnx"),
                                      "Rec.model_path": m("ch_PP-OCRv4_rec_infer.onnx"),
                                      "Cls.model_path": m("ch_ppocr_mobile_v2.0_cls_infer.onnx")})
    return _RAPID


def _rgb(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img)
    return np.dstack([img] * 3) if img.ndim == 2 else img


def rapid_words(img: np.ndarray, min_conf: float = 30.0) -> list[dict]:
    """Detect + recognise text lines; split each line into words (x shared by character count).
    Returns ``{text, conf (0-100), x0, y0, x1, y1, pass_}`` in ``img`` pixels."""
    # rapidocr 3.x keeps use_det/use_cls/use_rec from the previous call: always pass all three
    r = _rapid()(_rgb(img), use_det=True, use_cls=False, use_rec=True)
    out = []
    for box, text, conf in zip(r.boxes if r.boxes is not None else [], r.txts or (), r.scores or ()):
        conf = 100 * float(conf)
        if conf < min_conf or not text.strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        n = max(1, len(text)); pos = 0
        for w in text.split(" "):
            if w:
                out.append(dict(text=w, conf=conf, x0=x0 + (x1 - x0) * pos / n, x1=x0 + (x1 - x0) * (pos + len(w)) / n,
                                y0=y0, y1=y1, pass_="rapidocr"))
            pos += len(w) + 1
    return out


def rapid_line(img: np.ndarray) -> tuple[str, float]:
    """Recognise one already-cropped text line (no detection). Returns (text, conf 0-1)."""
    r = _rapid()(_rgb(img), use_det=False, use_cls=False, use_rec=True)
    if not r.txts:
        return "", 0.0
    return str(r.txts[0]), float(r.scores[0])
