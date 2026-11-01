#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Tracing benchmark against guided reference traces (training_data/corrections format).

For each reference folder the rectified panel (image.png) is re-traced with NO anchors (0 clicks) and
compared with the reference x_panel row by row:
  acc2   = % rows with |x - x_ref| <= 2 px  (the "99%" target)
  wrong  = % rows with |x - x_ref| > 10 px  (on another curve / lost)
The reference is itself a guided trace (not hand-digitised), so pair this with the LAS r check.

    python scripts/bench_trace.py                       # latest snapshot per well/curve/track
    python scripts/bench_trace.py --root training_data/corrections --all --csv out.csv
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def latest(folders):
    best = {}
    for d in sorted(folders):
        m = json.loads((d / "metadata.json").read_text())
        best[(m["well"], m["scan"], m["curve"], str(m.get("track")))] = d
    return list(best.values())


def score(d: pathlib.Path) -> dict:
    from PIL import Image
    from pylogdigitizer import curve_tracing as ct
    m = json.loads((d / "metadata.json").read_text()); cal = json.loads((d / "calibration.json").read_text())
    pan = np.asarray(Image.open(d / "image.png")) < 128
    ref = np.genfromtxt(d / "trace.csv", delimiter=",", names=True)
    t0 = time.time()
    T = ct.trace_track(pan, 0, px_per_ft=cal.get("px_per_ft"))
    xr = np.interp(ref["panel_row"], np.asarray(T["rows"], float), np.asarray(T["x"], float))
    err = np.abs(xr - ref["x_panel"]); err = err[np.isfinite(err)]
    return dict(ref=d.name, well=m["well"], curve=m["curve"], rows=len(err), n_ref_anchors=m.get("n_anchors"),
                acc1=round(100 * float((err <= 1).mean()), 2), acc2=round(100 * float((err <= 2).mean()), 2),
                wrong=round(100 * float((err > 10).mean()), 2), median_px=round(float(np.median(err)), 2),
                las_r_ref=(m.get("qc") or {}).get("r"), seconds=round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "training_data" / "corrections"))
    ap.add_argument("--all", action="store_true", help="every snapshot, not only the latest per well/curve")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    ds = [p for p in pathlib.Path(a.root).iterdir() if (p / "trace.csv").exists()]
    rows = [score(d) for d in (sorted(ds) if a.all else latest(ds))]
    for r in rows:
        print(f"{r['ref'][:44]:44s} rows={r['rows']:6d} acc2={r['acc2']:6.2f}% wrong={r['wrong']:5.2f}% med={r['median_px']}")
    if rows:
        w = np.array([r["rows"] for r in rows]); acc = np.array([r["acc2"] for r in rows])
        print(f"row-weighted acc2 = {float((w * acc).sum() / w.sum()):.2f}%  over {len(rows)} windows")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
