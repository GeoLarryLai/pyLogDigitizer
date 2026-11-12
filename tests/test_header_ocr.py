# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""header_ocr: fuzzy mnemonic lookup, unit/number parsing, scale fitting, proposal assembly from
synthetic OCR words, and one integration test on a real example header (needs RapidOCR)."""
from pathlib import Path
import numpy as np
import pytest

from pylogdigitizer import header_ocr as ho

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "example_data" / "USA_A_16_15-055-20536" / "4342_crop_3260-4770.tif"


@pytest.mark.parametrize("text,want", [
    ("GAMMA RAY", "GR"), ("GAMMA RAY API UNITS", "GR"), ("GAMMA RAV", "GR"), ("GR", "GR"),
    ("SPONTANEOUS POTENTIAL", "SP"), ("SP", "SP"), ("CALIPER HOLE DIAM. IN INCHES", "CALI"), ("CALI", "CALI"),
    ("BULK DENSITY", "RHOB"), ("RH0B", "RHOB"), ("COMPENSATED NEUTRON POROSITY", "NPHI"),
    ("C0MPENSATED NEUTR0N P0ROSITY", "NPHI"), ("NPHl", "NPHI"),
    ("COMPENSATED FORMATION DENSITY POROSITY", "DPHI"), ("INTERVAL TRANSIT TIME", "DT"), ("DT", "DT"),
    ("AC", "DT"), ("SONIC", "DT"), ("ILD", "ILD"), ("DEEP INDUCTION", "ILD"), ("ILM", "ILM"), ("LLD", "LLD"),
    ("LLS", "LLS"), ("SFL", "SFL"), ("MSFL", "MSFL"), ("RES", "RES"), ("RESISTIVITY OHMS M2/M", "RES"),
    ("PE", "PE"), ("PHOTOELECTRIC FACTOR", "PE"), ("MEDIUM INDUCTION LOG", "ILM"),
])
def test_lookup_mnemonic(text, want):
    assert ho.lookup_mnemonic(text) == want


@pytest.mark.parametrize("text", ["Top Porosity Matrix Scale", "liquid Hole", "09-25-83 17:48", "", "TIME"])
def test_lookup_rejects_noise(text):
    assert ho.lookup_mnemonic(text) is None


@pytest.mark.parametrize("text,unit", [
    ("API UNITS", "GAPI"), ("MILLIVOLTS", "MV"), ("HOLE DIAM. IN INCHES", "IN"), ("G/CC", "G/CC"),
    ("GM/CC", "G/CC"), ("POROSITY INDEX %", "%"), ("MICROSECONDS PER FOOT", "US/F"), ("<uS/FT>", "US/F"),
    ("OHMS M2/M", "OHMM"), ("ohm-m", "OHMM"), ("MILLIMHOS/M", "MMHO/M"),
])
def test_parse_unit(text, unit):
    assert ho.parse_unit(text)[0] == unit


def test_units_match_and_las_units():
    assert ho.normalize_las_unit("USEC/FT") == "US/F" and ho.normalize_las_unit("G/C3") == "G/CC"
    assert ho.units_match("%", "V/V") and not ho.units_match("MV", "GAPI") and not ho.units_match(None, "MV")


@pytest.mark.parametrize("text,v", [("150", 150), ("-10", -10), ("0.2", 0.2), ("2.O", 2.0), ("150)", 150),
                                    (".150", 150), ("—10", -10), ("40,", 40), ("D", 0), ("45%", 45)])
def test_parse_number(text, v):
    assert ho.parse_number(text) == pytest.approx(v)


@pytest.mark.parametrize("text", ["|", "TIME", "I", "API", "3-1", "tbo"])
def test_parse_number_rejects(text):
    assert ho.parse_number(text) is None


def test_fit_scale_linear_reversed_log_and_outlier():
    f = ho.fit_scale([(110, 30), (850, 10), (1520, -10)], 100, 1600)        # NPHI 30 .. -10
    assert (f["left"], f["right"], f["type"]) == (30, -10, "linear")
    f = ho.fit_scale([(110, 100), (500, 70), (890, 40)], 100, 900)          # DT 100 -> 40 (reversed)
    assert (f["left"], f["right"], f["type"]) == (100, 40, "linear")
    f = ho.fit_scale([(105, 0.2), (400, 2), (700, 20), (1000, 200), (1290, 2000)], 100, 1300)
    assert (f["left"], f["right"], f["type"]) == (0.2, 2000, "log")
    f = ho.fit_scale([(110, 90), (850, 10), (1520, -10)], 100, 1600, mnemonic="NPHI")   # "30" misread as 90
    assert (f["left"], f["right"]) == (30, -10)
    f = ho.fit_scale([(400, 70), (800, 40)], 0, 800)                         # left label missing: extrapolate
    assert f["left"] == pytest.approx(100) and f["method"] == "edge+fit"
    assert ho.fit_scale([(400, 7)], 0, 800) is None


def _w(text, x, y, conf=95, w=None, h=20):
    return dict(text=text, conf=conf, x0=x, y0=y, x1=x + (w or 14 * len(text)), y1=y + h)


def test_propose_curves_from_words():
    words = [_w("GAMMA", 300, 100), _w("RAY", 400, 100), _w("API", 500, 130), _w("UNITS", 560, 130),
             _w("0", 102, 200), _w("150", 850, 200), _w("150", 102, 260), _w("300", 850, 260)]
    (p,) = ho.propose_curves(words, 100, 900, "linear", track="track1")
    assert p["status"] == "proposed" and p["track"] == "track1"
    assert p["mnemonic"]["value"] == "GR" and p["unit"]["value"] == "GAPI"
    assert (p["scale_left"]["value"], p["scale_right"]["value"]) == (0, 150)
    assert p["scale_type"]["value"] == "linear" and p["backup"]["value"] == [150, 300]
    sp = p["evidence"]["scale_points"]                 # OCR'd numbers placed in the track (grid-line values)
    assert [v for _, v in sp] == [0, 150] and 0 < sp[0][0] < 0.05 and 0.9 < sp[1][0] < 1.0
    for k in ("mnemonic", "unit", "scale_left", "scale_right", "scale_type", "backup"):
        assert set(p[k]) == {"value", "confidence", "needs_review"} and 0 <= p[k]["confidence"] <= 1


def test_propose_curves_backup_only_gamma_and_flags():
    # only the backup line 160..320 was read: GR main is inferred (0..160) and flagged for review
    words = [_w("GAMMA", 300, 100), _w("RAY", 400, 100), _w("160", 102, 200), _w("320", 850, 200)]
    (p,) = ho.propose_curves(words, 100, 900, track="track1")
    assert (p["scale_left"]["value"], p["scale_right"]["value"]) == (0, 160)
    assert p["scale_left"]["needs_review"] and p["backup"]["value"] == [160, 320]


def test_propose_curves_shared_log_scale_and_no_names():
    words = [_w("DEEP", 300, 100), _w("INDUCTION", 380, 100), _w("MEDIUM", 300, 140), _w("INDUCTION", 400, 140),
             _w("0.2", 102, 200), _w("2", 400, 200), _w("20", 700, 200), _w("200", 1000, 200), _w("2000", 1250, 200)]
    ps = {p["mnemonic"]["value"]: p for p in ho.propose_curves(words, 100, 1300, track="track2")}
    assert set(ps) == {"ILD", "ILM"}
    for p in ps.values():
        assert (p["scale_left"]["value"], p["scale_right"]["value"], p["scale_type"]["value"]) == (0.2, 2000, "log")
    assert ho.propose_curves([_w("REMARKS", 100, 10), _w("12", 300, 50)], 0, 900) == []
    assert ho.propose_curves([]) == []


def test_to_runs_and_workflow_requires_confirmation():
    from pylogdigitizer import workflows as wf
    words = [_w("GAMMA", 300, 100), _w("RAY", 400, 100), _w("0", 102, 200), _w("150", 850, 200),
             _w("150", 102, 260), _w("300", 850, 260)]
    props = ho.propose_curves(words, 100, 900, track="track1")
    runs = ho.to_runs(dict(curves=props, scan="x/4342_crop_3260-4770.tif"))
    assert runs[0]["curve"] == "GR" and runs[0]["scale"] == [0, 150] and runs[0]["confirmed"] is False
    assert runs[0]["scan"] == "4342_crop_3260-4770.tif" and runs[0]["scale_source"] == "header_ocr"
    with pytest.raises(ValueError, match="not confirmed"):
        wf.digitize_kwargs_from_proposal(props[0])
    kw = wf.digitize_kwargs_from_proposal(props[0], confirmed=True)
    assert kw == dict(track="track1", scale=(0, 150), scale_type="linear", backup=(150, 300), wrap_side="right")
    assert wf.digitize_kwargs_from_proposal(props[0], confirmed=True, scale=(0, 200))["scale"] == (0, 200)


def test_ocr_region_custom_backend():
    import numpy as np
    seen = {}

    def fake(img):
        seen["shape"] = img.shape
        return [dict(text="GR", conf=90, x0=10, y0=10, x1=40, y1=30)]
    w = ho.ocr_region(np.zeros((50, 80), bool), backend=fake, offset=(100, 1000))
    assert seen["shape"] == (70, 100)                      # 10 px white padding each side
    assert w[0]["x0"] == 100 and w[0]["y0"] == 1000


@pytest.mark.skipif(not EX.exists(), reason="example data not present")
def test_integration_usa_crop_header():
    from pylogdigitizer.depth_calibration import ocr_available
    if not ocr_available():
        pytest.skip("RapidOCR not installed")
    H = ho.propose_header(str(EX))
    assert H["status"] == "proposed" and H["runtime_s"] < 30
    got = {(c["track"], c["mnemonic"]["value"]): c for c in H["curves"]}
    gr = got[("track1", "GR")]
    assert (gr["scale_left"]["value"], gr["scale_right"]["value"]) == (0, 150)
    assert gr["backup"]["value"] == [150, 300] and gr["unit"]["value"] == "GAPI"
    cal = got[("track1", "CALI")]
    assert (cal["scale_left"]["value"], cal["scale_right"]["value"]) == (6, 16)
    assert all(c["scale_type"]["value"] == "linear" for c in H["curves"])


def test_scale_sanity_flags():
    from pylogdigitizer.header_ocr import scale_sanity
    ok = [("GR", 0, 150), ("GR", 0, 160), ("GR", 10, 110), ("DT", 140, 40), ("DT", 100, 40), ("RHOB", 2.0, 3.0),
          ("NPHI", 30, -10), ("GR", 0, 120), ("CALI", 6, 16)]
    for m, l, r in ok:
        assert scale_sanity(m, l, r, "linear") == [], (m, l, r)
    assert scale_sanity("DT", 100, 35, "linear")            # span 65: not a round chart span
    assert scale_sanity("GR", -1, 7, "linear")              # below the gamma range
    assert scale_sanity("GR", 50, 50, "linear")
    assert scale_sanity("ILD", 0.2, 2000, "log") == []      # log scales: span rule not applied


def test_word_edits_refit_scale_and_save(tmp_path):
    # OCR read "GAMMA RAY" fine but "200" as "100": editing the word re-fits the scale
    words = [_w("GAMMA", 300, 100), _w("RAY", 400, 100), _w("0", 102, 200), _w("100", 850, 200, conf=40)]
    h = dict(tracks={"track1": dict(x=(100, 900), grid="linear", words=words, curves=ho.propose_curves(words, 100, 900, "linear", track="track1"))})
    before = h["tracks"]["track1"]["curves"]
    assert not before or before[0]["scale_right"]["value"] != 200
    ho.apply_word_edits(h, "track1", [dict(i=3, text="200")])
    (p,) = h["curves"]
    assert p["mnemonic"]["value"] == "GR" and (p["scale_left"]["value"], p["scale_right"]["value"]) == (0, 200)
    w = h["tracks"]["track1"]["words"][3]
    assert w["conf"] == 100 and w["corrected"] and w["ocr_text"] == "100"
    img = np.full((400, 1000), 255, np.uint8)
    names = ho.save_word_corrections(tmp_path, img, h["tracks"]["track1"]["words"], scan="x/s.tif")
    assert names == ["s_0000.png"] and (tmp_path / "labels.tsv").read_text().split("\t")[1] == "200"
    with pytest.raises(IndexError):
        ho.apply_word_edits(h, "track1", [dict(i=9, text="x")])
