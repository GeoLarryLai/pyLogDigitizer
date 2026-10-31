#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Replay the editor self-check sessions in-process and measure the anchors' *local* effect:
r vs paired LAS within +-50 ft of each anchor, before (conventional) and after (anchors), plus a
before/after editor-view PNG. Usage: python scripts/assisted_local_effect.py WELL [WELL ...] --tag _v4"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))


def local_r(c, centers, half=50.0):
    from pylogdigitizer import qc_compare as qc
    g = c["grid"]; m = np.zeros(len(g), bool)
    for d in centers:
        m |= np.abs(g - d) <= half
    return qc.metrics(c["ref"][m], c["test_aligned"][m])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("wells", nargs="+"); ap.add_argument("--curve", default="GR")
    ap.add_argument("--tag", default="_v4"); a = ap.parse_args()
    from pylogdigitizer.assisted_tracing import AssistedSession
    from render_editor_view import render
    from matplotlib.figure import Figure
    out = ROOT / "outputs" / "editor_selfcheck"; res = {}
    for w in a.wells:
        h = json.loads((out / f"{w}_{a.curve}{a.tag}_history.json").read_text())
        sess = ROOT / h["session"]["path"]
        s = AssistedSession.from_session_file(sess, root=ROOT)          # global replay with all anchors
        anchors = list(s.anchors); depths = [float(s.dm(r)) for r, _ in anchors]
        q_rep = s.qc(); c_after = q_rep["compare"]
        T_after, bi = s.T, list(s.backup_intervals)
        s.anchors = []; s.backup_intervals = []; s.auto_trace(); c_before = s.qc()["compare"]
        depths += [float(s.dm((a_ + b_) / 2)) for a_, b_ in bi]
        lb, la = local_r(c_before, depths), local_r(c_after, depths)
        res[w] = dict(global_before=s.qc()["metrics"]["r"], global_replay=q_rep["metrics"]["r"],
                      interactive_after=h["hist"][-1]["r"], local_before=lb["r"], local_after=la["r"],
                      n_anchors=len(anchors), n_backup_intervals=len(bi), anchor_depths=[round(d) for d in depths])
        # before/after view at the anchor with the biggest local gain
        gains = [local_r(c_after, [d], 25)["r"] - local_r(c_before, [d], 25)["r"] for d in depths]
        dbest = depths[int(np.nanargmax(gains))]
        fig = Figure(figsize=(10, 7)); ax1, ax2 = fig.subplots(1, 2)
        render(s, dbest, 25, None, f"{w} {a.curve} before (conventional)", ax=ax1)
        s.T = T_after; s.anchors = anchors; s.backup_intervals = bi; s._update_values()
        render(s, dbest, 25, None, f"after ({len(anchors)} anchors = green, {len(bi)} backup intervals = orange)", ax=ax2)
        fig.tight_layout(); fig.savefig(out / f"{w}_{a.curve}_before_after.png", dpi=90)
        # value tracks (what the QC panel shows): LAS vs conventional vs assisted, around the corrections
        fig = Figure(figsize=(11, 6)); axs = fig.subplots(1, 2, sharey=False)
        cen = sorted([float(s.dm((a_ + b_) / 2)) for a_, b_ in bi][:2]) or sorted(depths)
        for ax, (d0, d1) in zip(axs, ((cen[0] - 60, cen[0] + 60), (cen[-1] - 60, cen[-1] + 60)) if len(cen) > 1 else [(cen[0] - 100, cen[0] + 100)] * 2):
            for c_, col, lab in ((c_before, "r", f"conventional"), (c_after, "g", f"assisted")):
                m = (c_["grid"] >= d0) & (c_["grid"] <= d1)
                ax.plot(c_["test_aligned"][m], c_["grid"][m], col, lw=0.9, alpha=0.85, label=lab)
            m = (c_before["grid"] >= d0) & (c_before["grid"] <= d1)
            ax.plot(c_before["ref"][m], c_before["grid"][m], "k", lw=0.9, label="paired LAS")
            for a_, b_ in bi:
                da_, db_ = float(s.dm(a_)), float(s.dm(b_)); ax.axhspan(da_, db_, color="orange", alpha=0.15)
            for d_ in [float(s.dm(r)) for r, _ in anchors]:
                if d0 <= d_ <= d1:
                    ax.axhline(d_, color="lime", lw=1.5, alpha=0.8)
            ax.set_ylim(d1, d0); ax.set_xlabel(a.curve); ax.set_ylabel("depth (ft)"); ax.legend(fontsize=7, loc="lower right")
            ax.set_title(f"{w} {d0:.0f}-{d1:.0f} ft", fontsize=9)
        fig.suptitle(f"{w} {a.curve}: whole-section r {res[w]['global_before']:.3f} -> {res[w]['global_replay']:.3f} "
                     f"(orange = backup interval, green = anchor)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"{w}_{a.curve}_qc_before_after.png", dpi=90)
        print(w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[w].items()}, flush=True)
    (out / f"local_effect_{a.curve}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
