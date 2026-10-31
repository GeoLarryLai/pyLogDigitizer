#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Validate header_ocr proposals on the Kansas KGS paired wells.

    python scripts/eval_header_ocr.py                  # all wells, writes outputs/header_ocr/
    python scripts/eval_header_ocr.py --wells MOBIL_12-1

Ground truth (no manual reading of headers is used):
  * curve identity / track / scale ends / scale type: dataset_profiles/kansas_kgs_wells.yaml runs.
    Scale ends are scored only for runs with ``scale_source: header`` (values read off the header
    when the profile was written); ``assumed`` scales are vendor conventions and are not scored.
    Scale type is scored for every run (header and assumed) and reported separately.
  * units: the paired LAS ~Curve section (mnemonic -> unit), where the LAS gives a unit.
  * precision: a proposed mnemonic counts as correct if it is in the YAML runs of that scan or in the
    paired LAS curve list of the well (canonicalised with lookup_mnemonic); "family" precision also
    accepts any resistivity mnemonic (RES/ILD/ILM/LLD/LLS/SFL/MSFL) when the well has a resistivity
    curve. Header curves that are printed but absent from both (e.g. CALI on a GR track, SP when
    the LAS omits it) count as false positives, so precision is a lower bound.
Dev wells (used while tuning heuristics): USA_A_16, WEATHERFORD_B_1, MOBIL_12-1, GOHEEN_UNIT_1.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEV = {"USA_A_16", "WEATHERFORD_B_1", "MOBIL_12-1", "GOHEEN_UNIT_1"}
RES = {"RES", "ILD", "ILM", "LLD", "LLS", "SFL", "MSFL", "COND"}


def canon(m: str) -> str | None:
    from pylogdigitizer.header_ocr import lookup_mnemonic
    m = m.split(":")[0]
    return lookup_mnemonic(m) or m.upper()


def close(a, b, span) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= 0.02 * abs(span) + 1e-9


def main(argv=None):
    import yaml
    from pylogdigitizer.header_ocr import propose_header, normalize_las_unit, units_match
    from pylogdigitizer.las_export import read_las
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "header_ocr"))
    ap.add_argument("--backend", default="auto", help="auto | rapidocr | easyocr | paddle")
    a = ap.parse_args(argv)
    prof = yaml.safe_load((ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml").read_text())
    data = ROOT / prof.get("data_dir", "example_data")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, per_scan = [], []
    for w in prof["wells"]:
        if a.wells and w["name"] not in a.wells:
            continue
        split = "dev" if w["name"] in DEV else "held-out"
        las = read_las(data / w["dir"] / w["las"])
        las_units = {}
        for c in las.curves[1:]:
            k = canon(c.mnemonic)
            if c.unit and k not in las_units:
                las_units[k] = normalize_las_unit(c.unit)
        las_set = {canon(c.mnemonic) for c in las.curves[1:]}
        scans = {}
        for r in w["runs"]:
            scans.setdefault(r["scan"], []).append(r)
        for scan, runs in scans.items():
            path = data / w["dir"] / scan
            t0 = time.time()
            try:
                H = propose_header(str(path), backend=a.backend)
                err = ""
            except Exception as ex:
                H = dict(curves=[], tracks={}, header_rows=None)
                err = f"{type(ex).__name__}: {ex}"
            dt = time.time() - t0
            props = H["curves"]
            truth_set = {canon(r["curve"]) for r in runs}
            ok_set = truth_set | las_set
            has_res = bool(ok_set & RES)
            tp = sum(1 for p in props if p["mnemonic"]["value"] in ok_set)
            tpf = sum(1 for p in props if p["mnemonic"]["value"] in ok_set or
                      (has_res and p["mnemonic"]["value"] in RES))
            per_scan.append(dict(well=w["name"], split=split, scan=scan, n_props=len(props), tp=tp, tp_family=tpf,
                                 runtime_s=round(dt, 2), header_rows=H.get("header_rows"), error=err,
                                 proposals=[dict(track=p["track"], mnemonic=p["mnemonic"]["value"],
                                                 unit=p["unit"]["value"], scale=[p["scale_left"]["value"],
                                                                                 p["scale_right"]["value"]],
                                                 type=p["scale_type"]["value"], backup=p["backup"]["value"],
                                                 conf=p["mnemonic"]["confidence"],
                                                 review=[k for k in ("mnemonic", "unit", "scale_left", "scale_right",
                                                                     "scale_type") if p[k]["needs_review"]])
                                            for p in props]))
            for r in runs:
                m = canon(r["curve"])
                cands = [p for p in props if p["mnemonic"]["value"] == m]
                same_track = [p for p in cands if p["track"] == r["track"]]
                fam = [p for p in props if p["track"] == r["track"] and m in RES and p["mnemonic"]["value"] in RES]
                # best matching proposal: same track & mnemonic, prefer one with a scale
                pool = same_track or cands or fam
                pool.sort(key=lambda p: p["scale_left"]["value"] is None)
                p = pool[0] if pool else None
                st = r.get("scale_type", prof.get("defaults", {}).get("scale_type", "linear"))
                lo, hi = r["scale"]
                row = dict(well=w["name"], split=split, scan=scan, curve=r["curve"], canon=m, track=r["track"],
                           scale_source=r.get("scale_source", ""), truth_scale=[lo, hi], truth_type=st,
                           truth_backup=r.get("backup"), las_unit=las_units.get(m),
                           found=bool(same_track or cands), found_family=bool(pool), found_track=bool(same_track))
                if p is not None:
                    pl, pr = p["scale_left"]["value"], p["scale_right"]["value"]
                    row.update(pred_mnemonic=p["mnemonic"]["value"], pred_unit=p["unit"]["value"],
                               pred_scale=[pl, pr], pred_type=p["scale_type"]["value"],
                               pred_backup=p["backup"]["value"], scale_proposed=pl is not None,
                               scale_ok=close(pl, lo, hi - lo) and close(pr, hi, hi - lo),
                               type_ok=p["scale_type"]["value"] == st,
                               unit_ok=units_match(p["unit"]["value"], las_units.get(m)) if las_units.get(m) else None,
                               scale_review=p["scale_left"]["needs_review"],
                               backup_ok=(r.get("backup") is not None and p["backup"]["value"] is not None and
                                          close(p["backup"]["value"][0], r["backup"][0], hi - lo) and
                                          close(p["backup"]["value"][1], r["backup"][1], hi - lo)))
                rows.append(row)
    # aggregate
    def agg(split):
        R = [r for r in rows if split in (None, r["split"])]
        S = [s for s in per_scan if split in (None, s["split"])]
        hdr = [r for r in R if r["scale_source"] == "header"]
        units = [r for r in R if r.get("las_unit") and r["found_family"]]
        bk = [r for r in hdr if r["truth_backup"]]
        n_props = sum(s["n_props"] for s in S)
        frac = lambda k, L: (sum(1 for r in L if r.get(k)), len(L))
        return dict(
            scans=len(S), runs=len(R),
            curve_recall=frac("found", R), curve_recall_same_track=frac("found_track", R),
            curve_recall_family=frac("found_family", R),
            precision=(sum(s["tp"] for s in S), n_props), precision_family=(sum(s["tp_family"] for s in S), n_props),
            unit_acc=frac("unit_ok", units),
            scale_acc_header=frac("scale_ok", hdr), scale_proposed_header=frac("scale_proposed", hdr),
            scale_acc_when_proposed=frac("scale_ok", [r for r in hdr if r.get("scale_proposed")]),
            wrong_scale_not_flagged=(sum(1 for r in hdr if r.get("scale_proposed") and not r.get("scale_ok")
                                         and not r.get("scale_review")),
                                     sum(1 for r in hdr if r.get("scale_proposed") and not r.get("scale_ok"))),
            correct_scale_flagged=(sum(1 for r in hdr if r.get("scale_ok") and r.get("scale_review")),
                                   sum(1 for r in hdr if r.get("scale_ok"))),
            backup_acc=frac("backup_ok", bk),
            type_acc_all=frac("type_ok", [r for r in R if r["found_family"]]),
            type_acc_header=frac("type_ok", [r for r in hdr if r["found_family"]]),
            runtime_s_mean=round(sum(s["runtime_s"] for s in S) / max(1, len(S)), 2),
            runtime_s_max=max([s["runtime_s"] for s in S] or [0]))
    summary = {k: agg(v) for k, v in (("dev", "dev"), ("held-out", "held-out"), ("all", None))}
    (out / "eval.json").write_text(json.dumps(dict(summary=summary, runs=rows, scans=per_scan), indent=1, default=str))
    keys = ["curve_recall", "curve_recall_same_track", "curve_recall_family", "precision", "precision_family",
            "unit_acc", "scale_proposed_header", "scale_acc_header", "scale_acc_when_proposed",
            "wrong_scale_not_flagged", "correct_scale_flagged", "backup_acc", "type_acc_all", "type_acc_header"]
    lines = ["| metric | dev | held-out | all |", "|---|---|---|---|"]
    for k in keys:
        cells = []
        for s in ("dev", "held-out", "all"):
            n, d = summary[s][k]
            cells.append(f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("| runtime s/scan mean (max) | " + " | ".join(
        f"{summary[s]['runtime_s_mean']} ({summary[s]['runtime_s_max']})" for s in ("dev", "held-out", "all")) + " |")
    lines += ["", "| well | split | scan | curve | truth scale (src) | proposed | scale | type | unit |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['well']} | {r['split']} | {r['scan']} | {r['curve']} | {r['truth_scale']} {r['truth_type']} "
                     f"({r['scale_source']}) | {r.get('pred_mnemonic', '-')} {r.get('pred_scale', '')} "
                     f"{r.get('pred_type', '')} | {'ok' if r.get('scale_ok') else ('-' if r['scale_source'] != 'header' else 'x')} "
                     f"| {'ok' if r.get('type_ok') else 'x'} | {r.get('pred_unit', '')} vs {r.get('las_unit')} |")
    lines += ["", "| scan | split | runtime s | proposals (track mnemonic unit scale type) |", "|---|---|---|---|"]
    for s in per_scan:
        pr = "; ".join(f"{p['track']} {p['mnemonic']} {p['unit']} {p['scale']} {p['type']}" for p in s["proposals"])
        lines.append(f"| {s['scan']} | {s['split']} | {s['runtime_s']} | {pr or s['error'] or '(none)'} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:len(keys) + 3]))
    return summary


if __name__ == "__main__":
    main()
