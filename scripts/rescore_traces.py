#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Re-score saved traces (run_regression --save-traces) on the depth interval both the print trace
and the LAS cover, and report coverage separately. ``r_onscale`` also drops rows where the trace is
parked at a scale end (curve off the track: no traced value there).

    python scripts/rescore_traces.py --traces DIR --out rescore.csv
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    from run_regression import jobs_from_config
    from pylogdigitizer.las_export import read_las
    from pylogdigitizer import qc_compare as qc
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    a = ap.parse_args(argv)
    rows = []
    for j in jobs_from_config(pathlib.Path(a.config)):
        f = pathlib.Path(a.traces) / f"{j['well']}_{pathlib.Path(j['scan']).stem}_{j['curve'].replace(':', '_')}.npz"
        if not f.exists():
            continue
        z = np.load(f); d, v = z["depth"].astype(float), z["value"].astype(float)
        ref = read_las(j["las_path"]).df()[j["curve"]].dropna(); ref = ref[ref > -999]
        log = j.get("scale_type") == "log"
        rv = np.log10(np.clip(ref.values.astype(float), 1e-3, None)) if log else ref.values.astype(float)
        tv = np.log10(np.clip(v, 1e-3, None)) if log else v
        t0, t1 = np.nanmin(d), np.nanmax(d); l0, l1 = ref.index.min(), ref.index.max()
        ov = max(0.0, min(t1, l1) - max(t0, l0))
        sc = sorted(j["scale"]); lo, hi = (np.log10(sc[0]), np.log10(sc[1])) if log else sc
        tol = 0.005 * (hi - lo)
        park = (np.abs(tv - lo) <= tol) | (np.abs(tv - hi) <= tol)
        if j.get("backup"):
            park[:] = False      # a wrapped curve legitimately sits at the scale ends between wraps
        c = qc.compare(ref.index.values, rv, d, tv)
        c2 = qc.compare(ref.index.values, rv, d[~park], tv[~park]) if (~park).sum() > 50 else None
        rows.append(dict(well=j["well"], scan=j["scan"], curve=j["curve"], trace_top=t0, trace_base=t1,
                         las_top=l0, las_base=l1, overlap_ft=ov, cov_las=ov / max(l1 - l0, 1e-9),
                         cov_print=ov / max(t1 - t0, 1e-9), parked_frac=float(park.mean()),
                         r=c["aligned"].get("r"), r_reg=c["registered"].get("r"),
                         r_onscale=None if c2 is None else c2["aligned"].get("r"),
                         r_reg_onscale=None if c2 is None else c2["registered"].get("r")))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
