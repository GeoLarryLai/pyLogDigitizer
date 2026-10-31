# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Header OCR figures: OCR overlay on one header + dev/held-out metric bars."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pylogdigitizer import header_ocr as ho

IMG = ROOT / "outputs" / "header_ocr"


def overlay(scan: Path, out: Path):
    ink = ho._load_top(str(scan))                       # same deskewed sheet the proposal coordinates use
    H = ho.propose_header(ink)
    r0, r1 = H["header_rows"]
    img = np.where(ink[: r1 + 60], 0, 255).astype(np.uint8)
    hh = 12 * (r1 + 60 - max(0, r0 - 60)) / img.shape[1]
    nt = len(H["curves"])
    fig, (ax, at) = plt.subplots(2, 1, figsize=(12, hh + 0.3 * nt + 0.8), height_ratios=[hh, 0.3 * nt + 0.2])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_ylim(r1 + 60, max(0, r0 - 60))
    for t in H["tracks"].values():
        x0, x1 = t["x"]
        ax.add_patch(plt.Rectangle((x0, r0), x1 - x0, r1 - r0, fill=False, ec="tab:blue", lw=1.5))
        for w in t["words"]:
            ax.add_patch(plt.Rectangle((w["x0"], w["y0"]), w["x1"] - w["x0"], w["y1"] - w["y0"],
                                       fill=False, ec="tab:orange", lw=0.6))
    for i, c in enumerate(H["curves"]):
        def v(k, c=c):
            return c[k]["value"]
        rv = [k for k in ("mnemonic", "unit", "scale_left", "scale_right", "backup") if c[k]["needs_review"]]
        txt = f"{c['track']}: {v('mnemonic')} [{v('unit')}] {v('scale_left')}→{v('scale_right')} {v('scale_type')}" + \
              (f" backup {v('backup')}" if v("backup") else "") + (f"  review: {','.join(rv)}" if rv else "")
        at.text(0.0, 1 - (i + 0.5) / max(nt, 1), txt, transform=at.transAxes, va="center", fontsize=10,
                family="monospace", color="darkred" if rv else "darkgreen")
    ax.set_title(f"header_ocr proposal - {scan.name} ({H['runtime_s']:.1f} s; orange = OCR words, blue = track header boxes)")
    ax.axis("off"); at.axis("off"); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def bars(evalj: Path, out: Path):
    m = json.loads(evalj.read_text())["summary"]
    keys = ["curve_recall", "curve_recall_family", "precision", "precision_family", "unit_acc",
            "scale_acc_header", "scale_acc_when_proposed", "backup_acc", "type_acc_all"]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(keys))
    for j, (split, col) in enumerate([("dev", "tab:blue"), ("held-out", "tab:orange")]):
        fr = [m[split][k][0] / max(m[split][k][1], 1) for k in keys]
        b = ax.bar(x + (j - 0.5) * 0.38, fr, 0.38, color=col, label=split)
        for bi, k in zip(b, keys):
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 0.01,
                    f"{m[split][k][0]}/{m[split][k][1]}", ha="center", fontsize=7)
    ax.set_xticks(x, keys, rotation=25, ha="right", fontsize=8); ax.set_ylim(0, 1.12); ax.legend(loc="lower right")
    ax.set_title("header_ocr on the Kansas example wells: dev (4 wells) vs held-out (8 wells, not strictly blind)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    overlay(ROOT / "example_data/WEATHERFORD_B_1_15-129-20464/194290.tif", IMG / "header_ocr_194290.png")
    bars(ROOT / "outputs/header_ocr/eval.json", IMG / "header_ocr_metrics.png")
    print("wrote", IMG / "header_ocr_194290.png", IMG / "header_ocr_metrics.png")
