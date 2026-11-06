"""ONNX scan segmenter (runtime: numpy + onnxruntime only). Labels each pixel of a 1-bit log scan
as paper / grid / curve / text. A small model (~1.5 MB) trained on synthetic tiles from clean KGS
curves is bundled as the default (:func:`default_model`); retrain with notebooks/03_train_scan_segmenter
(writes to your own folder and never replaces the bundled file) and pass ``seg_model=<path>``."""
from __future__ import annotations

import numpy as np

import pathlib

CLASSES4 = ("paper", "grid", "curve", "text")
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"
MODELS = {"default": "scan_seg_v2.onnx",          # synthetic KGS-style tiles (pilot 2)
          "overflow": "scan_seg_overflow.onnx"}   # + overflow spikes / backup strokes (opt-in)


def default_model(name: str = "default") -> str:
    """Path of a bundled segmenter model ("default" or "overflow")."""
    p = MODEL_DIR / MODELS[name]
    if not p.exists():
        raise FileNotFoundError(f"bundled segmenter model missing: {p}")
    return str(p)


def classes_for(n: int) -> tuple:
    return CLASSES4 if n == 4 else tuple(f"c{i}" for i in range(n))
_SESS: dict = {}


def _session(model_path: str):
    if model_path not in _SESS:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.log_severity_level = 3
        _SESS[model_path] = ort.InferenceSession(model_path, o, providers=["CPUExecutionProvider"])
    return _SESS[model_path]


def _softmax(z, axis=0):
    z = z - z.max(axis, keepdims=True); e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def segment_scan(ink: np.ndarray, model_path: str, tile: int = 256, overlap: int = 32, mult: int = 8) -> np.ndarray:
    """``ink``: 2-D bool (True = ink). Returns float32 probabilities (n_classes, H, W), computed
    on overlapping tiles blended with a tapered weight so tile seams do not show."""
    ink = np.asarray(ink, bool); H, W = ink.shape
    sess = _session(model_path); name = sess.get_inputs()[0].name
    t = max(mult, (min(tile, max(H, W)) + mult - 1) // mult * mult)
    Hp, Wp = max(H, t), max(W, t)
    x = np.zeros((Hp, Wp), np.float32); x[:H, :W] = ink
    step = max(t - overlap, 1)
    ys = sorted({*range(0, Hp - t + 1, step), Hp - t}); xs = sorted({*range(0, Wp - t + 1, step), Wp - t})
    r = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
    w2 = np.minimum.outer(np.minimum(r, overlap / 2 + 1), np.minimum(r, overlap / 2 + 1))
    acc = None; wsum = np.zeros((Hp, Wp), np.float32)
    for y0 in ys:
        for x0 in xs:
            z = sess.run(None, {name: x[None, None, y0:y0 + t, x0:x0 + t]})[0][0]
            if acc is None:
                acc = np.zeros((z.shape[0], Hp, Wp), np.float32)
            acc[:, y0:y0 + t, x0:x0 + t] += _softmax(z) * w2; wsum[y0:y0 + t, x0:x0 + t] += w2
    return (acc / wsum)[:, :H, :W]


def label_scan(ink: np.ndarray, model_path: str, **kw) -> np.ndarray:
    """Argmax class index (uint8, H x W); paper outside ink is forced to 0."""
    p = segment_scan(ink, model_path, **kw)
    lab = p.argmax(0).astype(np.uint8)
    lab[~np.asarray(ink, bool)] = 0
    return lab


def curve_probability(ink: np.ndarray, model_path: str, **kw) -> np.ndarray:
    """P(curve) per pixel (sum of curve classes), float32 H x W."""
    p = segment_scan(ink, model_path, **kw)
    names = classes_for(p.shape[0])
    return sum(p[i] for i, n in enumerate(names) if n.startswith("curve")).astype(np.float32)

