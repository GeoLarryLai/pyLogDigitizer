#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Headless end-to-end check of the editor API over HTTP (server must be running:
``pylogdigitizer-app . --port 8765``; the editor API is mounted at ``/core``).

For each well/curve preset: open the scan (auto layout + OCR + conventional trace), then a
*simulated user* adds anchors QC-guided: at the worst 10-ft window of |digitized - LAS| where the
LAS is on the main scale, it clicks the curve (click x = paired LAS value mapped to the track,
then snapped by the editor to the nearest ink within 8 px); where the LAS is beyond the main
scale it clicks just outside the track edge (= "off-scale here", unless --on-scale-only).
Before anchors, up to --max-wraps "backup-scale interval" marks are made where the LAS is beyond
the main scale for >= 5 ft but the digitized curve is not (missed wraps). r is recorded after each anchor.
This exercises the same endpoints the browser uses; the LAS-derived click x is a stand-in for a
human eye and is NOT an independent accuracy claim.

    python scripts/editor_selfcheck.py --wells USA_A_16 --curve GR --anchors 10
Writes outputs/editor_selfcheck/{results_<curve>.csv, <well>_<curve>_editor.png, <well>_<curve>_history.json}.
"""
from __future__ import annotations
import argparse, csv, io, json, math, pathlib, time
import numpy as np
import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "editor_selfcheck"


def render_view(c: httpx.Client, sid: str, st: dict, depth_c: float, half_ft: float, ax, title: str, ref=None):
    """Editor-like view from the server's own tiles + overlay (what the browser draws)."""
    from PIL import Image
    cal = st["calibration"]; da = np.array(cal["depth_anchors"], float)
    row_of = lambda d: float(np.interp(d, da[:, 1], da[:, 0]))          # clamped to the calibrated range
    depth_c = float(np.clip(depth_c, da[:, 1].min() + half_ft, da[:, 1].max() - half_ft))
    r0, r1 = int(row_of(depth_c - half_ft)), int(row_of(depth_c + half_ft))
    tr = [t for t in st["tracks"] if t["name"] == (st["section"].get("track") or "")]
    xl, xr = (tr[0]["left"], tr[0]["right"]) if tr else (st["section"]["x0"], st["section"]["x1"])
    x0, x1 = max(0, int(xl) - 60), min(st["width"], int(xr) + 60)
    lv = math.ceil(math.log2(max(st["width"], st["height"]))); T = 512
    canvas = np.full((r1 - r0, x1 - x0), 255, np.uint8)
    for ty in range(r0 // T, (r1 - 1) // T + 1):
        for tx in range(x0 // T, (x1 - 1) // T + 1):
            im = np.asarray(Image.open(io.BytesIO(c.get(f"/core/api/sessions/{sid}/image_files/{lv}/{tx}_{ty}.png").content)))
            ya, xa = ty * T, tx * T
            sy0, sy1 = max(r0, ya), min(r1, ya + im.shape[0]); sx0, sx1 = max(x0, xa), min(x1, xa + im.shape[1])
            canvas[sy0 - r0:sy1 - r0, sx0 - x0:sx1 - x0] = im[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
    ov = c.get(f"/core/api/sessions/{sid}/overlay", params=dict(r0=r0, r1=r1, max_points=20000)).json()
    ax.imshow(canvas, cmap="gray", extent=(x0, x1, r1, r0), aspect="auto")
    t = np.array(ov["trace"]) if ov["trace"] else np.zeros((0, 4))
    if len(t):
        ax.plot(t[:, 1], t[:, 0], "r-", lw=1.3, alpha=0.85, label="editor trace")
    if ref is not None:
        ax.plot(ref[1], ref[0], "c--", lw=1, alpha=0.9, label="paired LAS (main scale)")
    for a in ov["anchors"]:
        if r0 <= a[0] <= r1:
            ax.plot(a[1], a[0], "o", ms=10, mfc="none", mec="lime", mew=2.5)
    yt = np.linspace(r0, r1, 5); ax.set_yticks(yt)
    ax.set_yticklabels([f"{np.interp(y, da[:, 0], da[:, 1]):.0f}" for y in yt]); ax.set_ylabel("depth (ft)")
    ax.set_title(title, fontsize=9); ax.legend(loc="lower right", fontsize=7)


def ref_xy(st, las_path, curve, depth_c, half_ft, overlay_tracks):
    from pylogdigitizer.las_export import read_las
    df = read_las(las_path).df(); s = df[curve].dropna(); s = s[(s.index > depth_c - half_ft) & (s.index < depth_c + half_ft)]
    da = np.array(st["calibration"]["depth_anchors"], float); sc = st["calibration"]["scale"]
    rows = np.interp(s.index.values, da[:, 1], da[:, 0])
    f = (s.values - sc["left"]) / (sc["right"] - sc["left"])
    tr = [t for t in st["tracks"] if t["name"] == st["section"].get("track")][0]
    x = tr["left"] + np.clip(f, -0.05, 1.05) * (tr["right"] - tr["left"])
    return rows, x


def run_case(c: httpx.Client, entry: dict, n_anchors: int, window_ft: float, off_scale_clicks: bool = True,
             max_wraps: int = 3):
    t0 = time.time()
    st = c.post("/core/api/sessions", json=dict(scan=entry["scan"], curve=entry["curve"], well=entry["well"], las=entry["las"],
                                           settings=entry["settings"]), timeout=900).json()
    sid = st["sid"]; S = f"/core/api/sessions/{sid}"
    t_open = time.time() - t0
    hist = [dict(n=0, r=st["qc"]["r"] if st.get("qc") else None, rmse=st["qc"]["rmse"] if st.get("qc") else None)]
    used, t_clicks = [], []
    sc = st["calibration"]["scale"]; lo, hi = sorted((sc["left"], sc["right"]))
    da = np.array(st["calibration"]["depth_anchors"], float)
    row_of = lambda d: float(np.interp(d, da[:, 1], da[:, 0]))
    tr = [t for t in st["tracks"] if t["name"] == st["section"].get("track")][0]
    n_wrap = 0
    for k in range(n_anchors):
        q = c.get(S + "/qc").json(); sh = q["metrics"]["shift_ft"]
        g, R, D = (np.array(q[k_], float) for k_ in ("depth", "ref", "digitized"))
        step = float(np.median(np.diff(g)))
        t1 = time.time()
        # 1) missed wraps: the curve visibly continues on the backup scale (hatched / re-entered)
        #    -> marks that depth interval (2 clicks). Up to max_wraps intervals >= 5 ft, longest first.
        hot = np.isfinite(D) & (R > hi) & (D < hi)
        runs = []
        if sc["backup"] and n_wrap < max_wraps and hot.any():
            from scipy.ndimage import binary_closing, label
            lab, n = label(binary_closing(hot, np.ones(max(1, int(3 / step)) * 2 + 1)))
            runs = sorted(((np.sum(lab == i) * step, g[lab == i].min(), g[lab == i].max()) for i in range(1, n + 1)), reverse=True)
            runs = [r_ for r_ in runs if r_[0] >= 5 and not any(abs(r_[1] - u) < 1 for u in used)]
        if runs:
            _, d0, d1 = runs[0]; used.append(d0); n_wrap += 1
            st = c.post(S + "/backup_interval", json=dict(rows=[row_of(d0 + sh), row_of(d1 + sh)])).json()
            act = f"backup interval {d0:.0f}-{d1:.0f} ft"
        else:
            # 2) anchor: click the curve at the worst 10-ft window
            e = np.abs(D - R); e[~np.isfinite(e)] = 0
            if not off_scale_clicks:
                e[(R > hi * 0.97) | (R < lo)] = 0
            w = np.convolve(e, np.ones(max(1, int(10 / step))), "same")
            for u in used:
                w[np.abs(g - u) < 15] = 0
            j = int(np.argmax(w)); dep = g[j] + sh; used.append(g[j])   # aligned(g) = digitized(g + shift)
            f = (R[j] - sc["left"]) / (sc["right"] - sc["left"])
            if f > 1.0 or f < 0.0:      # off the main scale there: click just outside that track edge
                f = 1.04 if f > 1.0 else -0.04
            x = tr["left"] + f * (tr["right"] - tr["left"])
            st = c.post(S + "/anchors", json=dict(row=row_of(dep), x=x, window_ft=window_ft), timeout=600).json()
            act = f"anchor {dep:.0f} ft"
        t_clicks.append(time.time() - t1)
        hist.append(dict(n=k + 1, r=st["qc"]["r"], rmse=st["qc"]["rmse"], action=act))
    return sid, st, hist, dict(open_s=t_open, mean_anchor_s=float(np.mean(t_clicks)) if t_clicks else None), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--wells", nargs="+", default=["USA_A_16"])
    ap.add_argument("--curve", default="GR"); ap.add_argument("--anchors", type=int, default=10)
    ap.add_argument("--window-ft", type=float, default=150.0)
    ap.add_argument("--on-scale-only", action="store_true", help="never click outside the track (no off-scale anchors)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-wraps", type=int, default=3, help="max 'backup-scale interval' corrections (0 = anchors only)")
    a = ap.parse_args()
    from matplotlib.figure import Figure
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(base_url=a.url, timeout=900) as c:
        cat = c.get("/core/api/catalog").json()
        for w in a.wells:
            ent = [e for e in cat if e["well"] == w and e["curve"] == a.curve][0]
            sid, st, hist, tm, used = run_case(c, ent, a.anchors, a.window_ft, not a.on_scale_only, a.max_wraps)
            S = f"/core/api/sessions/{sid}"
            (OUT / f"{w}_{a.curve}{a.tag}_history.json").write_text(json.dumps(dict(hist=hist, timing=tm), indent=1))
            for h in hist:
                rows.append(dict(well=w, curve=a.curve, **{k: h.get(k) for k in ("n", "r", "rmse", "action")}, **tm))
            # figure: worst anchored window, before/after needs the history; show 2 windows of the final trace
            fig = Figure(figsize=(12, 7.5)); axs = fig.subplots(1, 3)
            las_path = ROOT / ent["las"]
            dcs = sorted(used[:2]) if used else [st["depth_top_base"][0] + 200]
            for ax, dc in zip(axs[:2], dcs):
                ref = ref_xy(st, las_path, a.curve, dc, 30, st["tracks"])
                render_view(c, sid, st, dc, 30, ax, f"{w} {a.curve} @ {dc:.0f} ft (final, {st['n_anchors']} anchors, {st['n_backup_intervals']} intervals)", ref)
            n = [h["n"] for h in hist]; r = [h["r"] for h in hist]
            axs[2].plot(n, r, "o-"); axs[2].set_xlabel("corrections (anchors / backup intervals)"); axs[2].set_ylabel("r vs paired LAS")
            axs[2].set_title(f"{w} {a.curve}: r {r[0]:.3f} -> {r[-1]:.3f}"); axs[2].grid(alpha=.3)
            fig.tight_layout(); fig.savefig(OUT / f"{w}_{a.curve}{a.tag}_editor.png", dpi=90)
            print(w, [(h["n"], round(h["r"], 3)) for h in hist], tm, flush=True)
            c.delete(S)
    with open(OUT / f"results_{a.curve}{a.tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
