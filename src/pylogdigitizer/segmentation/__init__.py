# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""segmentation: pixel classes (paper / grid / curve / other) for binarized track panels.

* :func:`render_synthetic` draws a random track panel with its per-pixel truth: bent grid with heavy
  lines, one or two curves of varied width (solid or dashed), depth text, stamps, scribbles, hatch,
  speckle and broken strokes. It is the training data source (``scripts/train_segmenter.py``).
* :class:`Segmenter` runs the bundled ONNX model with plain ``onnxruntime`` (CPU provider), tiled.
* :func:`curve_probability` -> P(curve) per pixel, used by ``curve_tracing.trace_track(seg=...)``.

Inference needs only numpy + onnxruntime; torch is used only to train the model (dev tool)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

PAPER, GRID, CURVE, OTHER = 0, 1, 2, 3
CLASSES = ("paper", "grid", "curve", "other")
MODEL_PATH = Path(__file__).with_name("curve_seg.onnx")


# ----------------------------------------------------------------------------- synthetic data
def _smooth_walk(rng, n, scale, smooth):
    x = np.cumsum(rng.normal(0, 1, n))
    k = np.ones(max(1, smooth)) / max(1, smooth)
    x = np.convolve(x, k, "same")
    x = (x - x.min()) / max(1e-9, np.ptp(x))
    return x * scale


def render_synthetic(H: int = 384, W: int = 256, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ink bool HxW, label uint8 HxW). Labels follow draw order: later layers win."""
    import cv2
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W), np.uint8)

    def draw(mask, cls):
        lab[mask > 0] = cls

    # grid: vertical lines (slightly bent), horizontal lines with heavy every 5th
    g = np.zeros((H, W), np.uint8)
    nv = int(rng.integers(4, 12)); x0s = np.linspace(rng.uniform(-3, 3), W - 1 + rng.uniform(-3, 3), nv)
    yy = np.arange(H)
    bend_a, bend_p = rng.uniform(0, 4), rng.uniform(200, 2000)
    for x0 in x0s:
        xs = x0 + bend_a * np.sin(yy / bend_p * 2 * np.pi + rng.uniform(0, 6)) + rng.uniform(-0.01, 0.01) * yy
        t = int(rng.integers(1, 4))
        pts = np.c_[xs, yy].astype(np.int32)
        style = rng.choice(["solid", "dash", "dot"], p=[0.6, 0.2, 0.2])
        if style == "solid":
            cv2.polylines(g, [pts], False, 255, t)
        else:                                  # dashed / dotted grid lines (common on old prints)
            on = int(rng.integers(2, 4)) if style == "dot" else int(rng.integers(6, 16))
            per = on + int(rng.integers(2, 10))
            for a in range(int(rng.integers(0, per)), H, per):
                cv2.polylines(g, [pts[a:a + on]], False, 255, t)
    sp = rng.uniform(8, 40); y = rng.uniform(0, sp); k = int(rng.integers(0, 5))
    while y < H:
        t = int(rng.integers(2, 7)) if k % 5 == 0 else int(rng.integers(1, 3))
        cv2.line(g, (0, int(y)), (W - 1, int(y + rng.uniform(-2, 2))), 255, t)
        y += sp; k += 1
    if rng.random() < 0.35:    # thick straight reference lines (track centre, bit size, offset lines)
        for _ in range(int(rng.integers(1, 3))):
            xr = rng.uniform(0, W); a0 = int(rng.integers(0, H // 2)) if rng.random() < 0.4 else 0
            cv2.line(g, (int(xr), a0), (int(xr + rng.uniform(-2, 2)), H - 1), 255, int(rng.integers(3, 10)))
    if rng.random() < 0.7:     # broken / faded grid
        g[rng.random((H, W)) < rng.uniform(0, 0.25)] = 0
    draw(g, GRID)

    # other: hatch, text, stamps, scribbles (drawn under the curves)
    o = np.zeros((H, W), np.uint8)
    if rng.random() < 0.25:    # wrap hatch in a band at one edge
        a = int(rng.integers(0, H - 60)); b = a + int(rng.integers(40, 150)); wdt = int(rng.integers(15, W // 3))
        for yh in range(a - wdt, b, int(rng.integers(5, 10))):
            cv2.line(o, (0, yh + wdt), (wdt, yh), 255, 1)
    for _ in range(int(rng.poisson(1.2))):
        txt = "".join(rng.choice(list("0123456789 FTGRAPIMNSCL"), int(rng.integers(2, 8))))
        sc = rng.uniform(0.4, 1.3); th = int(rng.integers(1, 4))
        cv2.putText(o, txt, (int(rng.integers(0, W - 40)), int(rng.integers(20, H))), int(rng.choice([0, 1, 2, 3, 4])), sc, 255, th)
    if rng.random() < 0.2:
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H)); r = int(rng.integers(20, 70))
        if rng.random() < 0.5:
            cv2.rectangle(o, (cx - r, cy - r // 2), (cx + r, cy + r // 2), 255, -1 if rng.random() < 0.5 else 3)
        else:
            cv2.circle(o, (cx, cy), r, 255, 3)
    if rng.random() < 0.15:    # handwriting-like scribble
        n = 40; c0 = rng.uniform([0, 0], [W, H])
        pts = (c0 + np.cumsum(rng.normal(0, 4, (n, 2)), 0)).astype(np.int32)
        cv2.polylines(o, [pts], False, 255, int(rng.integers(1, 3)))
    draw(o, OTHER)

    # curves: main + optional second (caliper-like smooth, or dashed)
    for ci in range(1 + int(rng.random() < 0.5)):
        c = np.zeros((H, W), np.uint8)
        smooth = int(rng.integers(3, 40)) if ci == 0 else int(rng.integers(30, 200))
        x = rng.uniform(0, 0.3) * W + _smooth_walk(rng, H, rng.uniform(0.2, 1.0) * W, smooth)
        if ci == 0:                            # thin-bed wiggle
            x += np.convolve(rng.normal(0, rng.uniform(0, 0.03) * W, H), np.ones(3) / 3, "same")
        if ci == 0 and rng.random() < 0.3:   # spikes
            for _ in range(int(rng.integers(1, 4))):
                i = int(rng.integers(5, H - 5)); x[i - 2:i + 3] += rng.uniform(-0.5, 0.5) * W
        pts = np.c_[x, yy].astype(np.int32)
        t = int(rng.integers(1, 6))
        if ci == 1 and rng.random() < 0.5:   # dashed
            dl = int(rng.integers(6, 20))
            for a in range(0, H, 2 * dl):
                cv2.polylines(c, [pts[a:a + dl]], False, 255, t)
        else:
            cv2.polylines(c, [pts], False, 255, t)
        c[:, :1] = 0; c[:, -1:] = 0
        draw(c, CURVE)

    ink = lab > 0
    # scan noise: speckle and dropouts
    ink = ink | (rng.random((H, W)) < rng.uniform(0, 0.004))
    drop = rng.random((H, W)) < rng.uniform(0, 0.05)
    ink = ink & ~drop
    lab = np.where(ink, lab, PAPER).astype(np.uint8)
    lab[ink & (lab == PAPER)] = OTHER     # speckle
    return ink, lab


# ----------------------------------------------------------------------------- inference
class Segmenter:
    """ONNX segmentation model; input (1,1,h,w) float32 ink, output (1,4,h,w) logits."""

    def __init__(self, path: str | Path | None = None, threads: int | None = None):
        import onnxruntime as ort
        p = Path(path) if path else MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(f"segmentation model not found: {p} (train with scripts/train_segmenter.py)")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def predict(self, ink: np.ndarray, tile: int = 512, overlap: int = 32) -> np.ndarray:
        """Class probabilities (4, H, W) float32, tiled with overlap (tile edges discarded)."""
        a = (np.asarray(ink) > 0).astype(np.float32)
        H, W = a.shape
        m = 16   # network stride
        Wp = int(np.ceil(W / m) * m)
        out = np.zeros((len(CLASSES), H, W), np.float32)
        step = tile - 2 * overlap
        for y0 in range(0, H, step):
            a0 = max(0, y0 - overlap); a1 = min(H, y0 + step + overlap)
            hp = int(np.ceil((a1 - a0) / m) * m)
            x = np.zeros((1, 1, hp, Wp), np.float32); x[0, 0, :a1 - a0, :W] = a[a0:a1]
            lg = self.sess.run(None, {self.inp: x})[0][0, :, :a1 - a0, :W]
            e = np.exp(lg - lg.max(0, keepdims=True)); pr = e / e.sum(0, keepdims=True)
            b0, b1 = y0, min(H, y0 + step)
            out[:, b0:b1] = pr[:, b0 - a0:b1 - a0]
        return out


_CACHE: dict = {}


def get_segmenter(path=None) -> Segmenter:
    k = str(path or MODEL_PATH)
    if k not in _CACHE:
        _CACHE[k] = Segmenter(path)
    return _CACHE[k]


def available(path=None) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return Path(path or MODEL_PATH).exists()


def curve_probability(ink: np.ndarray, path=None) -> np.ndarray:
    """P(curve) per pixel (H, W) float32; 0 on paper."""
    p = get_segmenter(path).predict(ink)[CURVE]
    return np.where(np.asarray(ink) > 0, p, 0.0).astype(np.float32)
