"""Automatic scale reading from the scan header (header OCR) with a confidence flag.

Order of sources for each configured run (Larry's rule):
  1. ocr          targeted scale-end OCR above the track grid (header_ocr.read_track_scales):
                  'high' or 'corrected' (backup pair joins) -> used
  2. ocr_header   header proposal (header_ocr.propose_header) with no field flagged for review
  3. manual       config scale_source: header (read off the print by a person)
  4. las_fallback only when 1-3 give nothing AND the reference passes ref_quality: scale ends from
                  the value fit (LAS ~ gain * digitized + offset) of a pipeline run (--pipeline CSV).
                  Labelled las_fallback and kept out of the honest accuracy score.
  otherwise       unresolved -> MANUAL FIX (never a silent guess)
Legacy confidence of the header proposal:
  high = proposal found, no field flagged needs_review, mnemonic matched, backup joins the primary
  low  = anything else -> MANUAL FIX (read the print header, set scale_source: header). Never a
         silent guess. The LAS is never read here; the config value is shown only for comparison.

    python scripts/header_scale_check.py --wells Chitwood_12 ELY_1 --out header_scales.csv
"""
import argparse, csv, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def pick(curves, track, mnem):
    c = [p for p in curves if p.get("track") == track]
    m = [p for p in c if (p["mnemonic"]["value"] or "").upper().startswith(mnem)]
    return (m or c or [None])[0], bool(m)


def confidence(p, matched):
    if p is None or p["scale_left"]["value"] is None:
        return "low", "no scale proposal"
    why = [k for k in ("scale_left", "scale_right", "scale_type", "backup") if p[k]["needs_review"]]
    if not matched:
        why.append("mnemonic not matched")
    b = p["backup"]["value"]
    if b is not None:
        l, r = p["scale_left"]["value"], p["scale_right"]["value"]
        if min(abs(b[0] - r), abs(b[1] - l)) > 1e-6 * max(1, abs(r - l)):
            why.append("backup does not join primary")
    return ("low" if why else "high"), ";".join(why)


def main(argv=None):
    from run_regression import jobs_from_config
    from pylogdigitizer import header_ocr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--wells", nargs="*"); ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pipeline", default=None, help="run_regression CSV (gain/offset) for the las_fallback")
    ap.add_argument("--no-targeted", action="store_true")
    a = ap.parse_args(argv)
    import numpy as np, pandas as pd
    from ref_quality import quality
    from pylogdigitizer.las_export import read_las
    pipe = pd.read_csv(a.pipeline) if a.pipeline else None
    rows, cache, tcache = [], {}, {}
    for j in jobs_from_config(pathlib.Path(a.config), a.wells, a.only):
        if j["scan_path"] not in cache:
            try:
                cache[j["scan_path"]] = header_ocr.propose_header(j["scan_path"])
            except Exception as e:      # noqa: BLE001 - reported as low confidence
                cache[j["scan_path"]] = {"curves": [], "error": f"{type(e).__name__}: {e}"}
        H = cache[j["scan_path"]]
        p, matched = pick(H["curves"], j.get("track", "track1"), j["curve"].split(":")[0].upper())
        conf, why = confidence(p, matched)
        if H.get("error"):
            why = H["error"][:120]
        ocr = None if p is None else [p["scale_left"]["value"], p["scale_right"]["value"]]
        bk = None if p is None else p["backup"]["value"]
        # ---- resolution order
        if not a.no_targeted and j["scan_path"] not in tcache:
            try:
                tcache[j["scan_path"]] = header_ocr.read_track_scales(j["scan_path"])
            except Exception as e:      # noqa: BLE001
                tcache[j["scan_path"]] = {"error": str(e)}
        tr = (tcache.get(j["scan_path"]) or {}).get(j.get("track", "track1")) or {}
        resolved, rsrc, rback = None, "unresolved", None
        if tr.get("confidence") in ("high", "corrected") and tr.get("scale"):
            resolved, rback, rsrc = list(tr["scale"]), (list(tr["backup"]) if tr.get("backup") else None), "ocr"
        elif conf == "high" and ocr and None not in ocr:
            resolved, rback, rsrc = ocr, bk, "ocr_header"
        elif j.get("scale_source") == "header":
            resolved, rback, rsrc = list(j["scale"]), j.get("backup"), "manual"
        else:
            ref = read_las(j["las_path"]).df()[j["curve"]].dropna(); ref = ref[ref > -999]
            q = quality(ref.values, j["scale"], j.get("scale_type", "linear"))
            prow = None if pipe is None else pipe[(pipe.well == j["well"]) & (pipe.scan == j["scan"]) & (pipe.curve == j["curve"])]
            if q["suspect"]:
                rsrc = "unresolved (reference suspect: no las_fallback)"
            elif prow is None or not len(prow) or not np.isfinite(prow.gain.iloc[0]) or j.get("scale_type") == "log":
                rsrc = "unresolved (las_fallback needs a linear pipeline run)"
            else:
                g, o = float(prow.gain.iloc[0]), float(prow.offset.iloc[0])
                resolved = [round(g * v + o, 1) for v in j["scale"]]; rsrc = "las_fallback"
        # legend: how many curves the track header names (curve proposals) and line-style samples
        names = [c for c in H["curves"] if c.get("track") == j.get("track", "track1")]
        n_names = len({(c["mnemonic"]["value"] or f"?{i}").upper() for i, c in enumerate(names)})
        n_style = 0
        if not a.no_targeted:
            try:
                key = ("leg", j["scan_path"], j.get("track", "track1"))
                if key not in tcache:
                    from pylogdigitizer import track_grid_detection as tgd
                    ink = header_ocr._load_top(j["scan_path"])
                    _, tracks = tgd.find_probe(ink, 800)
                    tt = [t for t in tracks if t["kind"] == "track"]
                    k = int(str(j.get("track", "track1")).replace("track", "") or 1) - 1
                    tcache[key] = header_ocr.legend_styles(ink, tt[k]["grid_x"][0], tt[k]["grid_x"][-1]) if k < len(tt) else []
                n_style = len(tcache[key])
            except Exception:           # noqa: BLE001 - legend is optional
                n_style = 0
        n_leg = n_names if n_names else (n_style if 1 <= n_style <= 3 else 0)
        agrees = bool(ocr and None not in ocr and [float(x) for x in ocr] == [float(x) for x in j["scale"]])
        r = dict(well=j["well"], scan=j["scan"], curve=j["curve"], ocr_scale=json.dumps(ocr), ocr_backup=json.dumps(bk),
                 confidence=conf, reason=why, config_scale=json.dumps(j["scale"]), config_backup=json.dumps(j.get("backup")),
                 config_source=j.get("scale_source", ""), agrees=agrees,
                 targeted_scale=json.dumps(tr.get("scale")), targeted_backup=json.dumps(tr.get("backup")),
                 targeted_conf=tr.get("confidence", ""), targeted_reason=str(tr.get("reason", ""))[:200],
                 resolved_scale=json.dumps(resolved), resolved_backup=json.dumps(rback), resolved_source=rsrc,
                 legend_names=n_names, legend_styles=n_style, legend_n_curves=n_leg)
        rows.append(r)
        flag = "  -> MANUAL FIX" if rsrc.startswith("unresolved") else ""
        print(f'{r["well"]:18s} {r["scan"]:11s} {r["curve"]:5s} targeted={tr.get("scale")} {tr.get("backup")} ({tr.get("confidence")}) '
              f'header={ocr} ({conf}) -> {rsrc} {resolved} {rback} | config={j["scale"]} {j.get("backup")} ({r["config_source"]}){flag}', flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
