# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Headless end-to-end run of the desktop app API vs notebook 01's conventional path (Section A).

    python scripts/app_equivalence.py [WELL] [SCAN]        (default USA_A_16 4342_crop_3260-4770.tif)

Writes only to a temporary folder; prints whether the LAS/CSV values are identical."""
import sys, pathlib, tempfile, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from fastapi.testclient import TestClient
from pylogdigitizer.app import create_app
from pylogdigitizer import workflows as wf
from pylogdigitizer.las_export import resample, write_las, read_las
from run_regression import jobs_from_config

well, scan = (sys.argv[1:3] + ["USA_A_16", "4342_crop_3260-4770.tif"][len(sys.argv[1:3]):])[:2]
tmp = pathlib.Path(tempfile.mkdtemp(prefix="lld_eq_"))
# --- notebook 01 Section A (cells 2, 4, 10, 14)
j = [j for j in jobs_from_config(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml", [well]) if j["scan"] == scan][0]
kw = {k: (tuple(j[k]) if k == "scale" else j[k]) for k in ("track", "scale", "backup", "wrap_side", "use_hatch", "first_depth", "scale_type") if k in j}
t = time.time(); L = wf.analyze_layout(j["scan_path"]); R = wf.digitize_curve(j["scan_path"], layout=L, **kw)
g_nb, v_nb = resample(R["depth"], R["value"], 0.5)
write_las(tmp / "notebook.las", g_nb, {j["curve"]: v_nb}, units={j["curve"]: "API"}); t_nb = time.time() - t
# --- app API
c = TestClient(create_app(str(ROOT)))
cat = [x for x in c.get("/app/api/info").json()["catalog"] if x["well"] == well and x["scan"].endswith(scan) and x["curve"] == j["curve"]][0]
t = time.time()
jid = c.post("/app/api/open", json=cat).json()["job"]["id"]
while (st := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
    time.sleep(0.2)
assert st["state"] == "done", st; sid = st["result"]["sid"]; t_app = time.time() - t
c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp / "app")})
w = c.post(f"/app/api/{sid}/export/all").json()["written"]
la = read_las(tmp / "app" / f"{pathlib.Path(w[0]).name}"); ln = read_las(tmp / "notebook.las")
da, va = la.index, la.curves[1].data; dn, vn = ln.index, ln.curves[1].data
csvv = np.loadtxt([p for p in w if p.endswith(".csv")][0], delimiter=",", skiprows=1)
same_arrays = np.array_equal(g_nb, csvv[:, 0]) and np.array_equal(v_nb, csvv[:, 1])
print(f"{well} {scan}: notebook {t_nb:.1f} s, app open {t_app:.1f} s; samples {len(dn)} vs {len(da)}")
print("LAS depth identical:", np.array_equal(da, dn), "| LAS values identical:", np.array_equal(va, vn),
      "| max |diff|:", float(np.nanmax(np.abs(va - vn))) if len(va) == len(vn) else "n/a")
print("CSV == notebook arrays bit-for-bit (float64 repr):", same_arrays)
data = lambda p: [l for l in open(p).read().split("~A", 1)[1].splitlines()]
print("LAS ~A data section byte-identical:", data(tmp / "notebook.las") == data(tmp / "app" / pathlib.Path(w[0]).name))
print("written:", [pathlib.Path(p).name for p in w], "| tmp:", tmp)
