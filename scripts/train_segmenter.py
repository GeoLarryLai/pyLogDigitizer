# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Train the small panel segmentation model on synthetic panels and export it to ONNX.

Dev tool only: needs torch (and onnx for the exporter); the package runs the exported model with
plain onnxruntime. Output: src/pylogdigitizer/segmentation/curve_seg.onnx (default).

    python scripts/train_segmenter.py --steps 3000 --out src/pylogdigitizer/segmentation/curve_seg.onnx
"""
from __future__ import annotations
import argparse, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylogdigitizer import segmentation as sg

ROOT = pathlib.Path(__file__).resolve().parents[1]


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """4 levels (stride 16 bottleneck, wide receptive field for line-vs-curve context), ~120k params."""
    def __init__(self, c=(8, 16, 32, 48, 64), n_cls=4):
        super().__init__()
        self.enc = nn.ModuleList([block(1, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips[-1]
        for i in range(3, -1, -1):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


def batch(rng, n, H, W):
    xs, ys = [], []
    for _ in range(n):
        ink, lab = sg.render_synthetic(H, W, seed=int(rng.integers(1 << 31)))
        xs.append(ink.astype(np.float32)); ys.append(lab.astype(np.int64))
    return torch.from_numpy(np.stack(xs)[:, None]), torch.from_numpy(np.stack(ys))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--size", type=int, nargs=2, default=(256, 256)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "src" / "pylogdigitizer" / "segmentation" / "curve_seg.onnx"))
    ap.add_argument("--ckpt", default=None, help="checkpoint file (.pt): saved every 200 steps, resumed if present")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    net = TinyUNet(); print("params", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.steps)
    w = torch.tensor([0.05, 1.0, 3.0, 1.5])        # paper is trivial (no ink); curve matters most
    xv, yv = batch(np.random.default_rng(12345), 24, *a.size)
    t0 = time.time(); start = 0
    ck = pathlib.Path(a.ckpt) if a.ckpt else None
    if ck and ck.exists():
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng = st["rng"]; start = st["step"] + 1; print("resumed at step", start)
    for step in range(start, a.steps):
        net.train(); x, y = batch(rng, a.batch, *a.size)
        loss = F.cross_entropy(net(x), y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                pv = net(xv).argmax(1)
            ink = xv[:, 0] > 0
            rec = {c: float(((pv == k) & (yv == k)).sum() / max(1, (yv == k).sum())) for k, c in enumerate(sg.CLASSES) if k}
            pre = float(((pv == 2) & (yv == 2)).sum() / max(1, ((pv == 2) & ink).sum()))
            if ck:
                tmp = ck.with_suffix(".tmp")
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=rng, step=step), tmp)
                tmp.replace(ck)
            print(f"step {step} loss {loss.item():.3f} recall {rec} curve precision {pre:.3f} {time.time() - t0:.0f}s", flush=True)
    net.eval()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 1, 256, 256), str(out), input_names=["ink"], output_names=["logits"],
                      dynamic_axes={"ink": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
