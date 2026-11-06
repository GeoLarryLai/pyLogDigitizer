# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Train the 4-class scan segmenter (paper/grid/curve/text) on synthetic 1-bit tiles rendered from
clean LAS curves, and export ONNX. Dev tool: needs torch (+ onnx for export); the package runs the
result with onnxruntime only.

The output goes to ``out_dir`` (your folder). The bundled default model in the package is never
replaced unless ``overwrite_default=True`` is passed explicitly. Use the new model with
``digitize_curve(..., appearance_trace="model", seg_model="<out_dir>/scan_seg.onnx")``.

    python scripts/train_scan_segmenter.py --las-glob "data/*/*.las" --out-dir ~/pld_models --steps 3000
"""
from __future__ import annotations
import argparse, glob, json, pathlib, time
import numpy as np


def make_tiles(las_paths, n, seed0, overflow=0.5, exclude=()):
    from pylogdigitizer.ml_training_data import synth_scan as ss
    C = ss.load_curves([pathlib.Path(p) for p in las_paths if not any(e in str(p) for e in exclude)])
    X, Y = [], []
    for s in range(seed0, seed0 + n):
        t = ss.render_scan_tile(np.random.default_rng(s), C, 256, 256, overflow=overflow)
        y = t["label"].copy(); y[y == ss.CURVE_B] = ss.CURVE_A; y[y == ss.TEXT] = 3   # 4 classes
        X.append(t["image"] < 128); Y.append(y)
    return np.stack(X), np.stack(Y)


def _model(k=4):
    import torch.nn as nn, torch.nn.functional as F, torch

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class UNet(nn.Module):
        def __init__(s, c=(16, 32, 64, 96)):
            super().__init__()
            s.e = nn.ModuleList([blk(1, c[0])] + [blk(c[i], c[i + 1]) for i in range(len(c) - 1)])
            s.u = nn.ModuleList([blk(c[i + 1] + c[i], c[i]) for i in range(len(c) - 1)])
            s.h = nn.Conv2d(c[0], k, 1)

        def forward(s, x):
            f = []
            for i, b in enumerate(s.e):
                x = b(x if i == 0 else F.max_pool2d(x, 2)); f.append(x)
            for i in range(len(s.u) - 1, -1, -1):
                x = s.u[i](torch.cat([F.interpolate(x, scale_factor=2.0, mode="nearest"), f[i]], 1))
            return s.h(x)
    return UNet()


def train(las_glob: str, out_dir: str, steps: int = 3000, n_train: int = 3000, n_val: int = 200,
          overflow: float = 0.5, exclude=(), device: str | None = None, overwrite_default: bool = False) -> dict:
    import torch, torch.nn.functional as F
    from pylogdigitizer import scan_segmenter as sg
    out = pathlib.Path(out_dir).expanduser().resolve()
    if out == sg.MODEL_DIR.resolve() and not overwrite_default:
        raise ValueError("out_dir is the bundled model folder; pass overwrite_default=True to replace the default")
    out.mkdir(parents=True, exist_ok=True)
    las = sorted(glob.glob(str(pathlib.Path(las_glob).expanduser())))
    X, Y = make_tiles(las, n_train, 1000, overflow, exclude)
    Xv, Yv = make_tiles(las, n_val, 90000, overflow, exclude)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(0)
    m = _model().to(dev)
    X = torch.from_numpy(X).float()[:, None]; Y = torch.from_numpy(Y).long()
    w = torch.tensor([0.5, 1.0, 3.0, 1.5], device=dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    t0 = time.time()
    for it in range(steps):
        i = torch.randint(0, len(X), (12,)); x, y = X[i].to(dev), Y[i].to(dev)
        if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
        loss = F.cross_entropy(m(x), y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0:
            print(it, round(loss.item(), 3), round(time.time() - t0), "s", flush=True)
    m.eval().cpu()
    with torch.no_grad():
        P = torch.cat([m(torch.from_numpy(Xv[k:k + 20]).float()[:, None]).argmax(1) for k in range(0, len(Xv), 20)]).numpy()
    iou = {n: float(((P == c) & (Yv == c)).sum() / max(((P == c) | (Yv == c)).sum(), 1)) for c, n in enumerate(sg.CLASSES4)}
    path = out / "scan_seg.onnx"
    torch.onnx.export(m, torch.zeros(1, 1, 256, 256), str(path), input_names=["ink"], output_names=["logits"],
                      dynamic_axes={"ink": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}}, opset_version=17)
    res = dict(model=str(path), iou=iou, train_s=time.time() - t0, steps=steps, n_las=len(las))
    (out / "scan_seg.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--las-glob", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--overflow", type=float, default=0.5)
    ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--overwrite-default", action="store_true")
    a = ap.parse_args()
    print(train(a.las_glob, a.out_dir, a.steps, overflow=a.overflow, exclude=a.exclude, overwrite_default=a.overwrite_default))
