# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ocr_engines: the OCR engine that reads header text and depth labels.

One engine: ``rapidocr`` - RapidOCR (``rapidocr_onnxruntime``: PaddleOCR models on ONNX Runtime), a base
dependency (pip wheels only, models bundled, no system program). Tesseract was compared on the Kansas
scans (same accuracy, ~9x slower, needs a system install) and dropped.

``engine='auto'`` = RapidOCR if importable, else None (no OCR: depth falls back to first label + interval,
header rows are typed by hand). ``set_default(False/'none')`` or env ``PYLOGDIGITIZER_OCR=none`` turns OCR off.
"""
from __future__ import annotations
import importlib.util, os, threading
import numpy as np

ENGINES = ("rapidocr",)
HINTS = {"rapidocr": "pip install rapidocr_onnxruntime   (included in: pip install pylogdigitizer)"}
_DEFAULT: str | None = None
_LOCK = threading.Lock()
_RAPID = None


def available(engine: str = "rapidocr") -> bool:
    return engine == "rapidocr" and importlib.util.find_spec("rapidocr_onnxruntime") is not None


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
            from rapidocr_onnxruntime import RapidOCR
            _RAPID = RapidOCR()
    return _RAPID


def _rgb(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img)
    return np.dstack([img] * 3) if img.ndim == 2 else img


def rapid_words(img: np.ndarray, min_conf: float = 30.0) -> list[dict]:
    """Detect + recognise text lines; split each line into words (x shared by character count).
    Returns ``{text, conf (0-100), x0, y0, x1, y1, pass_}`` in ``img`` pixels."""
    res, _ = _rapid()(_rgb(img), use_cls=False)
    out = []
    for box, text, conf in res or []:
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
    eng = _rapid()
    rec = getattr(eng, "text_rec", None) or getattr(eng, "text_recognizer", None)
    if rec is None:
        res, _ = eng(_rgb(img), use_det=False, use_cls=False, use_rec=True)
    else:   # call the recogniser directly: rapidocr_onnxruntime < 1.3 (the only one on Python >= 3.13) has no use_det
        res, _ = rec([_rgb(img)])
    if not res:
        return "", 0.0
    t, c = res[0][0], res[0][1]
    return str(t), float(c)
