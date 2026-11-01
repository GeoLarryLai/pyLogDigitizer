# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""header_ocr: OCR the track-header boxes and PROPOSE curve mnemonics, units, left/right scale
values and linear vs log scale type. Proposals carry per-field confidences and ``needs_review``
flags and are never applied automatically; they are confirmed by hand (or by a workflow step).

Pipeline:
  1. load the top of the sheet, deskew (scan_cleanup) and find the first stable track layout
     (track_grid_detection.find_probe);
  2. locate the track-header box: the depth-column boundary lines run continuously up through the
     header box and stop at its top; the box ends where the regular grid of the log body starts;
  3. per track: remove straight grid/box lines, OCR words with boxes (RapidOCR, two resize
     variants merged);
  4. group words into lines, fuzzy-match lines against the bundled mnemonic table
     (header_ocr.mnemonics, CWLS LAS / SPWLA conventions) and unit spellings;
  5. numeric lines under a curve name give the scale: values are fitted against x (linear or log10)
     and evaluated at the track edges, so reversed scales (100 -> 40) and missing edge labels are
     handled; a second numeric line is proposed as the backup scale.
"""
from __future__ import annotations

import difflib
import os
import re
import time
from pathlib import Path

import numpy as np

from pylogdigitizer.header_ocr.mnemonics import CURVES, UNITS, LAS_UNIT_ALIASES, COMPATIBLE_UNITS

__all__ = ["ocr_region", "propose_curves", "lookup_mnemonic", "match_mnemonic", "parse_unit",
           "parse_number", "fit_scale", "find_header_box", "propose_header", "to_runs",
           "normalize_las_unit", "units_match", "REVIEW_THRESHOLD"]

REVIEW_THRESHOLD = 0.75          # field confidence below this -> needs_review
RAPID_VARIANTS = ((1.0, 1.0), (1.5, 1.0))   # RapidOCR header passes (resize, line-removal factor), merged
MIN_WORD_CONF = 30.0             # OCR word confidence floor (0-100)
# words that only appear in the service-company disclaimer / remarks above the header box
_STOP = {"INTERPRETATIONS", "OPINIONS", "INFERENCES", "SCHEDULE", "LIABLE", "RESPONSIBLE",
         "GUARANTEE", "ACCURACY", "CORRECTNESS", "NEGLIGENCE", "DAMAGES", "EXPENSES", "CUSTOMER",
         "AGENTS", "EMPLOYEES", "WILFUL", "WILLFUL", "PRICE", "CONDITIONS", "REMARKS", "EQUIPMENT",
         "MATRIX", "SETTING", "LIMESTONE", "SANDSTONE", "DOLOMITE"}
_ALPHA_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "|": "I", "$": "S", "6": "G"})
_NUM_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                          "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "—": "-",
                          "–": "-", "~": "-", ",": "."})

# typical end-value ranges per family (plausibility check only)
_RANGE = {"gamma": (0, 600), "sp": (-300, 300), "caliper": (0, 30), "density": (1.0, 3.5),
          "porosity": (-30, 100), "pe": (0, 20), "sonic": (20, 300), "resistivity": (0.05, 100000),
          "conductivity": (0, 10000)}


# ----------------------------------------------------------------------------- text matching
def _norm(text: str) -> str:
    t = text.upper().replace("—", "-").replace("–", "-")
    t = re.sub(r"[^A-Z0-9/%\-\. ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _alpha_tokens(text: str) -> list[str]:
    out = []
    for w in _norm(text).replace("/", " ").replace("-", " ").replace(".", " ").split():
        letters = sum(c.isalpha() for c in w)
        if letters == 0:
            continue
        if letters >= len(w) / 2:
            w = w.translate(_ALPHA_FIX)
        w = re.sub(r"[^A-Z]", "", w)
        if w:
            out.append(w)
    return out


def _phrase_table() -> list[tuple[str, str, bool]]:
    rows = []
    for m, d in CURVES.items():
        rows += [(a, m, False) for a in d["aliases"]]
        rows += [(p, m, True) for p in d["phrases"]]
    return rows


_PHRASES = _phrase_table()


def match_mnemonic(text: str, min_score: float = 0.8) -> tuple[str | None, float]:
    """Best canonical mnemonic for a header line and a 0..1 match score (fuzzy, OCR tolerant).
    Short aliases (<= 3 letters) must match a whole token exactly; longer aliases and phrases are
    compared with difflib ratios over windows of the line's tokens. Longer phrases win ties."""
    toks = _alpha_tokens(text)
    if not toks:
        return None, 0.0
    best, best_key = (None, 0.0), -1.0
    for phrase, mnem, is_phrase in _PHRASES:
        pt = phrase.split()
        n = len(pt)
        if n == 1 and len(phrase) <= 3:
            s = 1.0 if phrase in toks else 0.0
        elif n == 1 and len(phrase) == 4:       # one OCR substitution allowed (NPHl, RH0B)
            s = max((1.0 if w == phrase else 0.85 if len(w) == 4 and sum(a != b for a, b in zip(w, phrase)) == 1
                     else 0.0) for w in toks)
        else:
            s = 0.0
            target = " ".join(pt)
            for k in {max(1, n - 1), n, n + 1}:
                for i in range(max(1, len(toks) - k + 1)):
                    cand = " ".join(toks[i:i + k])
                    r = difflib.SequenceMatcher(None, target, cand).ratio()
                    s = max(s, r)
        if s < min_score:
            continue
        key = s + 0.03 * n + (0.01 if is_phrase else 0)
        if key > best_key:
            best, best_key = (mnem, round(float(s), 3)), key
    return best


def lookup_mnemonic(text: str) -> str | None:
    """Canonical curve mnemonic (GR, SP, CALI, RHOB, NPHI, DT, ILD, ...) for a printed header
    label or LAS mnemonic, or None. Tolerates OCR errors ("GAMMA RAV", "NPHl", "C0MPENSATED")."""
    return match_mnemonic(text)[0]


_UNIT_KEYS = sorted(UNITS, key=len, reverse=True)


def parse_unit(text: str) -> tuple[str | None, float]:
    """Canonical unit (GAPI, MV, IN, G/CC, %, US/F, OHMM, ...) printed in ``text`` and a score."""
    t = " " + _norm(text) + " "
    t2 = t.replace(" ", "")
    for k in _UNIT_KEYS:
        kk = _norm(k)
        if len(kk.replace(" ", "")) <= 3 and kk not in ("%",):
            if re.search(r"(?<![A-Z])" + re.escape(kk) + r"(?![A-Z])", t):
                return UNITS[k], 0.9
        elif kk.replace(" ", "") in t2:
            return UNITS[k], 1.0
    # fuzzy for long spellings
    words = _alpha_tokens(text)
    for k in _UNIT_KEYS:
        if len(k) >= 6:
            kt = re.sub(r"[^A-Z ]", " ", k).split()
            n = len(kt)
            for i in range(max(1, len(words) - n + 1)):
                if difflib.SequenceMatcher(None, " ".join(kt), " ".join(words[i:i + n])).ratio() >= 0.8:
                    return UNITS[k], 0.7
    return None, 0.0


def normalize_las_unit(u: str | None) -> str | None:
    if not u:
        return None
    k = u.strip().upper()
    return LAS_UNIT_ALIASES.get(k, k)


def units_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    return any(a in s and b in s for s in COMPATIBLE_UNITS)


_PUNCT = "|'\"‘’“”°()[]{}:;!_*,<>»«"


def parse_number(text: str) -> float | None:
    """Parse a scale label ("150", "-10", "0.2", "2.O", "1OO", "45%", "150)") or None. Words
    ("TIME") and lone glyphs that are usually line remnants ("|", "I", "l") are not numbers; a lone
    "O"/"D" is read as 0 (left-edge zeros are often misread)."""
    t = text.strip()
    if len(t) == 1 and not t.isdigit():
        return 0.0 if t in "ODo" else None
    if sum(c.isalpha() and c not in "OoDlIS" for c in t) > 0 and sum(c.isdigit() for c in t) < 2 \
            or sum(c.isalpha() for c in t) > sum(c.isdigit() for c in t) + 1:
        return None
    t = t.replace("—", "-").replace("–", "-").replace("~", "-")
    if sum(c.isdigit() for c in t) < 2:
        t = t.translate(_NUM_FIX)
    else:                                     # mostly digits: map only the classic confusions
        t = t.translate(str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "S": "5"}))
    t = t.strip(_PUNCT + " ").rstrip("%").strip(_PUNCT + " ")
    t = re.sub(r"^\.(?=\d{3})", "", t)            # ".150": stray dot before an integer label
    t = t.rstrip(".")
    if not re.fullmatch(r"-?\d*\.?\d+", t or ""):
        return None
    try:
        return float(t)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- OCR
_READERS: dict = {}     # lazily created optional OCR engines


def _easyocr_words(img: np.ndarray) -> list[dict]:     # optional backend
    import easyocr
    if "easyocr" not in _READERS:
        _READERS["easyocr"] = easyocr.Reader(["en"], gpu=False, verbose=False)
    out = []
    for box, text, conf in _READERS["easyocr"].readtext(img):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        for j, w in enumerate(text.split()):     # split phrases into approx words
            n = len(text.split()); x0, x1 = min(xs), max(xs); dx = (x1 - x0) / n
            out.append(dict(text=w, conf=100 * float(conf), x0=x0 + j * dx, y0=min(ys), x1=x0 + (j + 1) * dx,
                            y1=max(ys), pass_="easyocr"))
    return out


def _paddle_words(img: np.ndarray) -> list[dict]:      # optional backend
    from paddleocr import PaddleOCR
    if "paddle" not in _READERS:
        _READERS["paddle"] = PaddleOCR(lang="en", show_log=False)    # PaddleOCR 2.x API
    out = []
    for line in (_READERS["paddle"].ocr(np.dstack([img] * 3)) or [[]])[0] or []:
        box, (text, conf) = line
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        out.append(dict(text=text, conf=100 * float(conf), x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
                        pass_="paddle"))
    return out


def _iou_x(a, b) -> float:
    ix = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    if ix == 0 or iy == 0:
        return 0.0
    ua = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - ix * iy
    return ix * iy / ua


def _same_word(a, b) -> bool:
    """Same reading at (nearly) the same place: overlapping x and y ranges."""
    if a["text"].strip(".,'|") != b["text"].strip(".,'|"):
        return False
    ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    oy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return ox > 0 and oy > -0.5 * max(a["y1"] - a["y0"], b["y1"] - b["y0"])


def _merge_words(base: list[dict], extra: list[dict]) -> list[dict]:
    """Union of two word lists; overlapping boxes keep the higher-confidence reading (a reading
    that parses as the expected kind -- word vs number -- is preferred on ties)."""
    out = list(base)
    for d in extra:
        same = [i for i, w in enumerate(out) if _iou_x(w, d) > 0.3 or _same_word(w, d)]
        if not same:
            out.append(d)
            continue
        i = same[0]
        if d["conf"] > out[i]["conf"] + 5 and len(d["text"].strip("|_—-")) >= len(out[i]["text"].strip("|_—-")):
            out[i] = d
    return out


def ocr_region(image: np.ndarray, backend="rapidocr", upscale: float = 1, offset=(0, 0),
               remove_lines: bool = True, line_len: int = 81,
               variants=((0.5, 1.0), (1.0, 1.0), (1.0, 2.0), (1.5, 1.0))) -> list[dict]:
    """OCR one header region; returns words ``{text, conf (0-100), x0, y0, x1, y1}`` in the
    caller's coordinates (``offset`` = (x, y) of the region's top-left).

    The region is read under a few (resize factor, line-removal length factor) variants and the
    words are merged (overlapping boxes: higher confidence wins). ``backend``: 'rapidocr' (default,
    ``RAPID_VARIANTS``), 'easyocr', 'paddle', or a callable(gray_uint8) -> words (``variants``)."""
    words: list[dict] = []
    if backend == "rapidocr":
        variants = RAPID_VARIANTS
    else:
        variants = variants[1:2]
    for s, lf in variants:
        words = _merge_words(words, _ocr_once(image, backend, upscale * s, offset, remove_lines, int(line_len * lf)))
    return words


def _ocr_once(image: np.ndarray, backend="rapidocr", upscale: float = 1, offset=(0, 0),
              remove_lines: bool = True, line_len: int = 81) -> list[dict]:
    """OCR one header region. ``image``: bool ink mask (True = ink) or uint8 gray (255 = paper).
    Returns words ``{text, conf (0-100), x0, y0, x1, y1}`` in the caller's coordinates
    (``offset`` = (x, y) of the region's top-left). ``backend``: 'rapidocr' (default), 'easyocr', 'paddle', or a callable(gray_uint8) -> words."""
    import cv2
    from pylogdigitizer import scan_cleanup as scl
    if image.dtype == bool:
        ink = image
    else:
        ink = np.asarray(image) < 128
    if remove_lines and ink.size:
        ink = scl.remove_grid(ink, max(41, int(line_len)))[0]   # straight runs >= line_len px
    img = np.where(ink, 0, 255).astype(np.uint8)
    if upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC if upscale > 1 else cv2.INTER_AREA)
    img = np.ascontiguousarray(np.pad(img, 10, constant_values=255))
    if callable(backend):
        words = list(backend(img))
    elif backend == "rapidocr":
        from pylogdigitizer import ocr_engines
        words = ocr_engines.rapid_words(img, MIN_WORD_CONF)
    elif backend == "easyocr":
        words = _easyocr_words(img)
    elif backend == "paddle":
        words = _paddle_words(img)
    else:
        raise ValueError(f"unknown OCR backend {backend!r}")
    ox, oy = offset
    out = []
    for w in words:
        w = dict(w)
        for k, o in (("x0", ox), ("x1", ox), ("y0", oy), ("y1", oy)):
            w[k] = (float(w[k]) - 10) / upscale + o
        out.append(w)
    return out


# unit -> the curve it usually belongs to (fallback when the curve name itself is unreadable)
_UNIT_TO_CURVE = {"GAPI": "GR", "MV": "SP", "US/F": "DT", "G/CC": "RHOB", "IN": "CALI", "OHMM": "RES",
                  "MMHO/M": "COND", "B/E": "PE", "%": "NPHI"}


# ----------------------------------------------------------------------------- lines and scales
def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Cluster words into text lines by vertical centre (tolerance 0.6 x median word height)."""
    if not words:
        return []
    h = float(np.median([w["y1"] - w["y0"] for w in words])) or 10.0
    lines: list[list[dict]] = []
    centers: list[float] = []
    for w in sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2):
        c = (w["y0"] + w["y1"]) / 2
        j = int(np.argmin([abs(c - y) for y in centers])) if centers else -1
        if j >= 0 and abs(c - centers[j]) <= 0.6 * h:
            lines[j].append(w)
            centers[j] = float(np.mean([(v["y0"] + v["y1"]) / 2 for v in lines[j]]))
        else:
            lines.append([w]); centers.append(c)
    order = np.argsort(centers)
    return [sorted(lines[i], key=lambda w: w["x0"]) for i in order]


def _line_y(line) -> float:
    return float(np.mean([(w["y0"] + w["y1"]) / 2 for w in line]))


def _nice(v: float) -> float:
    if v == 0:
        return 0.0
    mag = 10 ** np.floor(np.log10(abs(v)))
    for step in (0.5, 1.0):
        r = round(v / (mag * step)) * mag * step
        if abs(r - v) <= 0.1 * max(abs(v), mag):
            return float(round(r, 6))
    return float(round(v, 3))


def _is_decades(vals) -> bool:
    v = np.asarray(sorted(vals), float)
    if len(v) < 3 or np.any(v <= 0):
        return False
    r = v[1:] / v[:-1]
    return bool(np.all(np.abs(r - 10) < 0.6))


def _drop_outlier(pts, mnemonic: str | None = None):
    """With >= 3 labels, drop one label inconsistent with a line (linear or log10) through the
    others -- OCR misreads such as 30 -> 90 or a stray "1". Candidates are ranked by how well the
    remaining labels fit, whether the implied end values are plausible for the curve and "round",
    and (weakly) a preference for dropping an outer label (outer labels touch the box lines)."""
    if len(pts) < 3:
        return pts
    xs = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])

    def models(v):
        return [v] + ([np.log10(v)] if np.all(v > 0) else [])

    def maxres(x, y):
        c = np.polyfit(x, y, 1)
        return np.max(np.abs(np.polyval(c, x) - y)) / (np.ptp(y) + 1e-9)

    if any(maxres(xs, y) <= 0.06 for y in models(vs)):
        return pts
    best = None
    for i in range(len(pts)):
        keep = np.array([j for j in range(len(pts)) if j != i])
        kv = vs[keep]
        if len(keep) > 2 and not any(maxres(xs[keep], y) <= 0.06 for y in models(kv)):
            continue
        fam_log = mnemonic in CURVES and CURVES[mnemonic]["scale_type"] == "log" and np.all(kv > 0)
        c = np.polyfit(xs[keep], np.log10(kv) if fam_log else kv, 1)
        ends = [np.polyval(c, xs[0]), np.polyval(c, xs[-1])]
        if fam_log:
            ends = [10 ** e for e in ends]
        nice = sum(abs(_nice(v) - v) <= 0.08 * max(1, abs(v)) for v in ends) / 2
        plaus = _plausible(mnemonic, min(ends), max(ends))
        edge = i in (0, len(pts) - 1)
        score = 2.0 * plaus + nice + 0.2 * edge + 0.5 * (len(set(np.sign(np.diff(kv)))) == 1)
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return pts
    return [p for j, p in enumerate(pts) if j != best[1]]


def fit_scale(points, x_left: float, x_right: float, grid_kind: str | None = None,
              prior_type: str | None = None, edge_frac: float = 0.15, mnemonic: str | None = None) -> dict | None:
    """Scale ends from numeric header labels.

    points: [(x_center, value), ...] (one printed scale line). Returns dict(left, right, type,
    method, residual) or None. Edge labels within ``edge_frac`` of the track width from an edge are
    taken as that edge's value; otherwise values are fitted against x (linear, or log10 when the
    labels are decades / span >= 2 decades / grid is log) and evaluated at the edges."""
    pts = sorted((float(x), float(v)) for x, v in points if v is not None)
    if not pts:
        return None
    pts = _drop_outlier(pts, mnemonic)
    W = float(x_right - x_left)
    xs = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
    pos = bool(np.all(vs > 0))
    span = (vs.max() / vs.min()) if pos and vs.min() > 0 else 1.0
    if pos and (_is_decades(vs) or (len(vs) >= 2 and span >= 100) or
                (grid_kind == "log" and span >= 9) or (len(vs) == 1 and grid_kind == "log")):
        kind = "log"
    else:
        kind = "linear"
    if kind == "linear" and len(vs) >= 3 and pos and span >= 9:
        # compare residuals of the two models
        A = np.c_[np.ones_like(xs), xs]
        rl = np.linalg.lstsq(A, vs, rcond=None)[1]
        rg = np.linalg.lstsq(A, np.log10(vs), rcond=None)[1]
        if rl.size and rg.size and rg[0] * np.var(vs) < rl[0] * np.var(np.log10(vs)):
            kind = "log"
    near_l = [(x, v) for x, v in pts if x - x_left <= edge_frac * W]
    near_r = [(x, v) for x, v in pts if x_right - x <= edge_frac * W]
    if len(pts) == 1:
        return None
    y = np.log10(vs) if kind == "log" else vs
    A = np.c_[np.ones_like(xs), xs]
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(res[0] / len(xs))) / (np.ptp(y) + 1e-9) if res.size else 0.0
    if abs(coef[1]) < 1e-12:
        return None
    ev = lambda x: coef[0] + coef[1] * x
    fl, fr = ev(x_left), ev(x_right)
    if kind == "log":
        fl, fr = 10 ** fl, 10 ** fr
    if not (np.isfinite(fl) and np.isfinite(fr)) or max(abs(fl), abs(fr)) > 1e6:
        return None
    left = near_l[0][1] if near_l else _nice(fl)
    right = near_r[-1][1] if near_r else _nice(fr)
    method = "edges" if (near_l and near_r) else ("edge+fit" if (near_l or near_r) else "fit")
    if left == right:
        return None
    if kind == "log" and (left <= 0 or right <= 0):
        kind = "linear"
    return dict(left=float(left), right=float(right), type=kind, method=method, residual=round(float(resid), 4),
                n_labels=len(pts))


def _plausible(mnem: str | None, left: float, right: float) -> bool:
    if not mnem or mnem not in CURVES:
        return True
    lo, hi = _RANGE.get(CURVES[mnem]["family"], (-1e9, 1e9))
    return lo <= min(left, right) and max(left, right) <= hi


# printed linear scales span 10 chart divisions of a "round" size, so the span's two leading
# digits are one of these (0-150 -> 15, 140-40 -> 10, 2.0-3.0 -> 10, 30..-10 -> 40, 0-160 -> 16)
_NICE_SPAN = {10, 12, 14, 15, 16, 18, 20, 24, 25, 30, 40, 45, 50, 60, 75, 80, 90}


def _nice_span(left: float, right: float) -> bool:
    span = abs(right - left)
    if span <= 0:
        return False
    m = span / 10 ** np.floor(np.log10(span))          # 1 <= m < 10
    return round(m * 10, 6) in _NICE_SPAN


def scale_sanity(mnem: str | None, left, right, scale_type: str | None) -> list[str]:
    """Reasons a proposed scale looks wrong (empty list = passes). Used to force ``needs_review``."""
    if left is None or right is None:
        return []
    why = []
    if not _plausible(mnem, left, right):
        why.append("outside the usual range for this curve type")
    if left == right:
        why.append("left == right")
    if scale_type != "log" and not _nice_span(left, right):
        why.append(f"span {abs(right - left):g} is not a round chart span")
    return why


def _numeric_line(line) -> list[tuple[float, float, float]] | None:
    """(x_center, value, conf) for a line whose words are mostly numbers, else None."""
    vals = [(0.5 * (w["x0"] + w["x1"]), parse_number(w["text"]), w["conf"]) for w in line]
    nums = [v for v in vals if v[1] is not None]
    alpha = sum(1 for w in line if sum(c.isalpha() for c in w["text"]) >= 2)
    if len(nums) >= 1 and alpha <= max(1, len(nums) // 2):
        return nums
    return None


def _field(value, conf: float) -> dict:
    conf = float(np.clip(conf, 0.0, 1.0))
    return dict(value=value, confidence=round(conf, 3), needs_review=bool(conf < REVIEW_THRESHOLD or value is None))


def propose_curves(ocr_tokens: list[dict], x_left: float | None = None, x_right: float | None = None,
                   grid_kind: str | None = None, track: str | None = None) -> list[dict]:
    """Turn OCR words of ONE track's header box into curve proposals.

    Each proposal: {track, mnemonic, unit, scale_left, scale_right, scale_type, backup} where every
    field is ``{value, confidence, needs_review}``, plus ``label`` (matched header text),
    ``status='proposed'`` and ``evidence``. Nothing is applied; callers must confirm."""
    words = [w for w in ocr_tokens if w.get("conf", 100) >= MIN_WORD_CONF]
    if not words:
        return []
    if x_left is None:
        x_left = min(w["x0"] for w in words)
    if x_right is None:
        x_right = max(w["x1"] for w in words)
    lines = _group_lines(words)
    hmed = float(np.median([w["y1"] - w["y0"] for w in words])) or 10.0
    names, nums = [], []
    for ln in lines:
        text = " ".join(w["text"] for w in ln)
        toks = set(_alpha_tokens(text))
        if toks & _STOP:
            continue
        m, s = match_mnemonic(" ".join(w["text"] for w in ln if parse_number(w["text"]) is None))
        if m is not None:
            oc = float(np.mean([w["conf"] for w in ln])) / 100
            names.append(dict(y=_line_y(ln), mnem=m, score=s, ocr=oc, text=text, line=ln))
            inline = [(0.5 * (w["x0"] + w["x1"]), parse_number(w["text"]), w["conf"]) for w in ln
                      if parse_number(w["text"]) is not None]
            if inline:                               # "150 GR API 300": scale printed on the name line
                nums.append(dict(y=_line_y(ln) + 0.01, pts=inline, text=text, inline=True))
            continue
        nl = _numeric_line(ln)
        if nl is not None:
            nums.append(dict(y=_line_y(ln), pts=nl, text=text))
    # unit-only labels (e.g. "<uS/FT>" under an unreadable delta-T glyph, "API UNITS" under an
    # unread "GAMMA RAY"): propose the unit's usual curve at reduced confidence
    for ln in lines:
        text = " ".join(w["text"] for w in ln)
        if set(_alpha_tokens(text)) & _STOP or _numeric_line(ln) is not None:
            continue
        y = _line_y(ln)
        if any(abs(y - n["y"]) < 2.6 * hmed for n in names):
            continue
        u, us = parse_unit(text)
        m = _UNIT_TO_CURVE.get(u)
        if m is not None and us >= 0.9:
            oc = float(np.mean([w["conf"] for w in ln])) / 100
            names.append(dict(y=y, mnem=m, score=0.6, ocr=oc, text=text, line=ln, from_unit=True))
    names.sort(key=lambda n: n["y"])
    # merge adjacent name lines that are one wrapped label ("COMPENSATED NEUTRON" / "POROSITY")
    merged = []
    for n in names:
        both_clear = n["mnem"] != (merged[-1]["mnem"] if merged else None) and \
            min(n["score"], merged[-1]["score"] if merged else 0) >= 0.95     # two distinct labels
        if merged and n["y"] - merged[-1]["y"] < 2.2 * hmed and not both_clear:
            m2, s2 = match_mnemonic(merged[-1]["text"] + " " + n["text"])
            if m2 is not None and s2 >= max(merged[-1]["score"], n["score"]) - 0.02:
                merged[-1].update(mnem=m2, score=s2, text=merged[-1]["text"] + " " + n["text"],
                                  line=merged[-1]["line"] + n["line"])
                continue
            if n["mnem"] == merged[-1]["mnem"]:
                continue
        if merged and n["mnem"] == merged[-1]["mnem"] and n["y"] - merged[-1]["y"] < 6 * hmed and \
                not any(merged[-1]["y"] < q["y"] < n["y"] for q in nums):
            if n["score"] * n["ocr"] > merged[-1]["score"] * merged[-1]["ocr"]:
                merged[-1] = n
            continue
        merged.append(n)
    names = merged
    # units: words on the name line or within ~2 line heights below / to the right
    props = []
    for i, n in enumerate(names):
        nxt = names[i + 1]["y"] if i + 1 < len(names) else np.inf
        near = [ln for ln in lines if n["y"] - 0.6 * hmed <= _line_y(ln) < min(nxt, n["y"] + 2.6 * hmed)]
        utext = " ".join(w["text"] for ln in near for w in ln if _numeric_line([w]) is None)
        unit, us = parse_unit(utext)
        d = CURVES[n["mnem"]]
        if unit is None:
            unit_f = _field(d["unit"], 0.35)
        else:
            agree = units_match(unit, d["unit"]) or (d["family"] == "porosity" and unit in ("%", "V/V"))
            unit_f = _field(unit, us * (0.95 if agree else 0.55))
        props.append(dict(track=track, n=n, unit=unit_f, nums=[]))
    # assign numeric lines to the nearest curve name (prefer the one above)
    for nl in nums:
        if not props:
            break
        best, bd = None, np.inf
        for p in props:
            dy = nl["y"] - p["n"]["y"]
            dist = dy if dy >= 0 else -2.5 * dy         # below the name is the usual layout
            if dist < bd:
                best, bd = p, dist
        if bd <= 18 * hmed:
            best["nums"].append(nl)
    out = []
    for p in props:
        n, d = p["n"], CURVES[p["n"]["mnem"]]
        name_conf = n["score"] * (0.5 + 0.5 * n["ocr"])
        scales = []
        for nl in sorted(p["nums"], key=lambda q: abs(q["y"] - n["y"])):
            f = fit_scale([(x, v) for x, v, _ in nl["pts"]], x_left, x_right, grid_kind, d["scale_type"],
                          mnemonic=n["mnem"])
            if f is not None:
                f["ocr"] = float(np.mean([c for _, _, c in nl["pts"]])) / 100
                f["y"] = nl["y"]; f["text"] = nl["text"]
                scales.append(f)
        # drop implausible readings (decorations, tick labels) when a plausible line exists
        rejected = [f["text"] for f in scales if not _plausible(n["mnem"], f["left"], f["right"])]
        scales = [f for f in scales if _plausible(n["mnem"], f["left"], f["right"])]
        # log-presented families (resistivity): a decade line beats a nearer linear line
        if d["scale_type"] == "log":
            scales.sort(key=lambda f: f["type"] != "log")
        main = scales[0] if scales else None
        backup = scales[1] if len(scales) > 1 else None
        backup_inferred = False
        if main is not None and backup is None:
            # a backup line often shows only the wrap value (e.g. "150" under GR 0..150 -> 150..300)
            span = main["right"] - main["left"]
            for nl in sorted(p["nums"], key=lambda q: abs(q["y"] - n["y"])):
                if nl["y"] == main["y"] or len(nl["pts"]) != 1:
                    continue
                x, v, c = nl["pts"][0]
                if abs(v - main["right"]) <= 1e-6 * max(1, abs(v)) and main["type"] == "linear":
                    left_side = x - x_left < 0.5 * (x_right - x_left)
                    backup = dict(left=v, right=v + span, ocr=c / 100) if left_side else \
                        dict(left=main["left"] - span, right=v, ocr=c / 100)
                    backup_inferred = True
                    break
        if main is None:
            sl = sr = _field(None, 0.0)
            st_type = d["scale_type"] if grid_kind not in ("linear", "log") else grid_kind
            st = _field(st_type, 0.4)
        else:
            base = {"edges": 0.95, "edge+fit": 0.8, "fit": 0.6}[main["method"]]
            base *= (0.6 + 0.4 * main["ocr"]) * (1 - min(0.5, 3 * main["residual"]))
            if not _plausible(n["mnem"], main["left"], main["right"]):
                base *= 0.5
            sl = _field(main["left"], base); sr = _field(main["right"], base)
            sanity = scale_sanity(n["mnem"], main["left"], main["right"], main["type"])
            if sanity:                          # any failed sanity check -> always needs review
                for f in (sl, sr):
                    f["needs_review"] = True; f["confidence"] = min(f["confidence"], 0.5)
            tconf = 0.6 + 0.2 * (main["type"] == d["scale_type"]) + 0.2 * (grid_kind == main["type"])
            if grid_kind in ("linear", "log") and grid_kind != main["type"]:
                tconf = 0.45
            st = _field(main["type"], tconf)
        if main is not None and backup is not None and not backup_inferred and main["type"] == "linear":
            # backups repeat the main span shifted by one span (wrap right: R..R+span, left: L-span..L);
            # snap a reading within 10% of either candidate to it (fixes one misread end)
            span = main["right"] - main["left"]
            cands = [(main["right"], main["right"] + span), (main["left"] - span, main["left"])]
            err = [abs(backup["left"] - a) + abs(backup["right"] - b) for a, b in cands]
            j = int(np.argmin(err))
            if 0 < err[j] <= 0.1 * 2 * abs(span):
                backup = dict(backup, left=cands[j][0], right=cands[j][1])
                backup_inferred = True
        bk = _field(None if backup is None else [backup["left"], backup["right"]],
                    0.0 if backup is None else (0.6 + 0.3 * backup["ocr"]) * (0.8 if backup_inferred else 1.0))
        if backup is None:
            bk["needs_review"] = False          # absence of a backup scale is the common case
        out.append(dict(track=track, label=n["text"], status="proposed",
                        mnemonic=_field(n["mnem"], name_conf), unit=p["unit"],
                        scale_left=sl, scale_right=sr, scale_type=st, backup=bk,
                        evidence=dict(name_y=round(n["y"], 1), scale_lines=[s["text"] for s in scales],
                                      scale_method=main["method"] if main else None, grid=grid_kind,
                                      rejected_scale_lines=rejected,
                                      scale_sanity=(scale_sanity(n["mnem"], main["left"], main["right"], main["type"])
                                                    if main else []),
                                      single_labels=[(round((nl["pts"][0][0] - x_left) / (x_right - x_left), 3),
                                                      nl["pts"][0][1]) for nl in p["nums"] if len(nl["pts"]) == 1])))
    return _share_track_scales(_merge_same_curve(out))


def _share_track_scales(props: list[dict]) -> list[dict]:
    """Track-level consistency rules (all results stay flagged for review):

    * gamma ray scales start near 0: a GR scale whose left value is >= half its span (e.g. 160..320
      read while the main 0..160 line was missed) is the backup line -> main = left-span..left;
    * curves of one family in one track (ILD/ILM/SFL on a shared log grid) share the scale line:
      a curve without a scale (or with a scale of the wrong type for its family) takes the scale of
      a same-family curve whose scale matches the family's usual type."""
    for p in props:
        l, r = p["scale_left"]["value"], p["scale_right"]["value"]
        fam = CURVES.get(p["mnemonic"]["value"], {}).get("family")
        if fam == "gamma" and l is not None and r is not None and r > l >= 0.5 * (r - l):
            span = r - l
            if p["backup"]["value"] is None:
                p["backup"] = _field([l, r], 0.6)
            p["scale_left"] = _field(_nice(l - span), 0.55)
            p["scale_right"] = _field(l, 0.55)
            p["evidence"]["scale_method"] = "gamma_backup_line_shifted"
    for p in props:
        m = p["mnemonic"]["value"]
        if m not in CURVES:
            continue
        fam, typ = CURVES[m]["family"], CURVES[m]["scale_type"]
        if p["scale_left"]["value"] is not None and p["scale_type"]["value"] == typ:
            continue
        donors = [q for q in props if q is not p and q["track"] == p["track"] and q["mnemonic"]["value"] in CURVES
                  and CURVES[q["mnemonic"]["value"]]["family"] == fam and q["scale_left"]["value"] is not None
                  and q["scale_type"]["value"] == typ and q["evidence"].get("scale_method") != "shared"]
        if donors:
            d = max(donors, key=lambda q: q["scale_left"]["confidence"])
            c = 0.8 * d["scale_left"]["confidence"]
            p["scale_left"] = _field(d["scale_left"]["value"], c)
            p["scale_right"] = _field(d["scale_right"]["value"], c)
            p["scale_type"] = _field(typ, max(0.5, 0.9 * d["scale_type"]["confidence"]))
            p["evidence"]["scale_method"] = "shared"
    return props


def _merge_same_curve(props: list[dict]) -> list[dict]:
    """Two header entries for the same mnemonic in one track: the later one whose left value equals
    the earlier one's right value is the backup scale (e.g. "0 GR 150" / "150 GR 300"). When the
    earlier entry only shows its right-edge value (the left "0" unread), it is inferred from the
    backup span. Plain duplicates are dropped (the one with a scale is kept)."""
    out: list[dict] = []
    for b in props:
        a = next((q for q in out if q["mnemonic"]["value"] == b["mnemonic"]["value"]), None)
        if a is None:
            out.append(b)
            continue
        al, ar = a["scale_left"]["value"], a["scale_right"]["value"]
        bl, br = b["scale_left"]["value"], b["scale_right"]["value"]
        if bl is not None and ar is not None and abs(bl - ar) < 1e-9 and a["backup"]["value"] is None:
            a["backup"] = _field([bl, br], 0.8 * min(b["scale_left"]["confidence"], 1.0))
        elif bl is not None and al is None:
            right_single = [v for f, v in a["evidence"].get("single_labels", []) if f > 0.7]
            if any(abs(v - bl) < 1e-9 for v in right_single):
                span = br - bl
                a["scale_left"] = _field(_nice(bl - span), 0.55)
                a["scale_right"] = _field(bl, 0.7)
                a["scale_type"] = b["scale_type"]
                a["backup"] = _field([bl, br], 0.7)
                a["evidence"]["scale_method"] = "inferred_from_backup"
            else:
                out[out.index(a)] = b
        # else: duplicate label line -> keep the first
    return out


# ----------------------------------------------------------------------------- header box + driver
def find_header_box(ink: np.ndarray, tracks: list[dict], probe: int, scale: float = 1.0) -> tuple[int, int]:
    """(top, bottom) rows of the track-header box around ``probe``.

    top: the depth-column boundaries are straight vertical lines that run up through the header
    box and stop at its top border; we follow their vertical-line mask (not just any ink, so table
    text above does not count) upwards from the probe row. bottom: the log body starts where the
    track's regular horizontal grid begins (>= 8 evenly spaced full-width lines)."""
    import cv2
    from pylogdigitizer import track_grid_detection as tgd
    H = ink.shape[0]
    dep = tgd.depth_column(tracks)
    lo = max(0, int(probe - 2500 * scale))
    top = max(0, int(probe - 350 * scale))
    if dep is not None:
        tops = []
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(40 * scale))))
        for x in (dep["grid_x"][0], dep["grid_x"][-1]):
            x = int(round(x))
            strip = ink[lo:probe + 1, max(0, x - 6):x + 7].astype(np.uint8)
            vm = cv2.morphologyEx(strip, cv2.MORPH_OPEN, k).astype(bool).any(1)[::-1]
            gap, tt = 0, probe
            for i, v in enumerate(vm):
                if v:
                    gap = 0; tt = probe - i
                else:
                    gap += 1
                    if gap > int(15 * scale) + 3:
                        break
            tops.append(tt)
        top = max(min(tops), int(probe - 1500 * scale))
        top = max(0, top - int(10 * scale))
    tr = [t for t in tracks if t["kind"] == "track"]
    bottom = min(H, int(max(top, probe) + 900 * scale))
    if tr:
        t = tr[0]
        a, b = top + int(60 * scale), min(H, int(max(probe, top) + 1800 * scale))
        if b - a > 50:
            c, _ = tgd.grid_rows(ink[a:b, int(t["grid_x"][0]):int(t["grid_x"][-1]) + 1], 0.6)
            c = np.asarray(c) + a
            for i in range(len(c) - 8):
                g = np.diff(c[i:i + 9])
                if g.max() < 130 * scale and g.min() > 0.75 * g.max():
                    bottom = int(c[i]) + 2
                    break
    return int(top), int(max(bottom, top + 100 * scale))


def _load_top(path, max_rows: int = 20000):
    from pylogdigitizer.scan_io import load_ink, read_info
    from pylogdigitizer import scan_cleanup as scl
    info = read_info(path)
    ink = load_ink(path, (0, 0, info["width"], min(info["height"], max_rows)))
    ink, _ = scl.deskew_blocks(ink, 2000)
    return ink


def propose_header(source, layout=None, backend="auto", max_workers: int = 4,
                   max_rows: int = 20000) -> dict:
    """OCR the track headers of a scan and return proposals (nothing is applied).

    source: scan path (or a bool ink array). layout: optional workflows.Layout to reuse its deskewed
    ink, tracks and probe row. Returns {scan, header_rows, tracks: {name: {x, grid, curves, words}},
    curves (flat list), runtime_s, backend, status='proposed'}. ``backend``: 'auto' / 'rapidocr'
    (default), 'easyocr', 'paddle' or a callable."""
    from pylogdigitizer import track_grid_detection as tgd
    if backend in (None, "auto"):
        from pylogdigitizer import ocr_engines
        backend = ocr_engines.resolve("auto")
        if backend is None:
            raise RuntimeError("no OCR engine installed: " + ocr_engines.HINTS["rapidocr"])
    t0 = time.time()
    y_off = x_off = 0
    if layout is not None:
        ink, tracks, probe = layout.ink, layout.tracks, int(layout.probe_row)
        x_off, y_off = int(layout.x_offset), int(layout.y_offset)
    else:
        ink = source if isinstance(source, np.ndarray) else _load_top(source, max_rows)
        probe, tracks = tgd.find_probe(ink, 800)
    s = ink.shape[1] / 2769.0
    up = 1.0
    top, bottom = find_header_box(ink, tracks, probe, s)
    names, k = [], 0
    for t in tracks:
        names.append("depth" if t["kind"] == "depth" else f"track{(k := k + 1)}")
    jobs = []
    for nm, t in zip(names, tracks):
        if t["kind"] != "track":
            continue
        xl, xr = float(t["grid_x"][0]), float(t["grid_x"][-1])
        m = int(0.04 * (xr - xl)) + 8
        a, b = max(0, int(xl) - m), min(ink.shape[1], int(xr) + m)
        jobs.append((nm, t, xl, xr, a, b))

    def run(job):
        a, b = job[4:]
        w = ocr_region(ink[top:bottom, a:b], backend=backend, upscale=up, offset=(a, top), line_len=int(120 * s))
        return w

    results = [run(j) for j in jobs]        # one shared RapidOCR model: sequential
    out_tracks, flat = {}, []
    for (nm, t, xl, xr, _a, _b), words in zip(jobs, results, strict=True):
        gk = tgd.classify_grid(t)
        cur = propose_curves(words, xl, xr, gk if gk != "unknown" else None, track=nm)
        for c in cur:
            c["evidence"]["name_y"] = c["evidence"]["name_y"] + y_off
        out_tracks[nm] = dict(x=(xl + x_off, xr + x_off), grid=gk, curves=cur,
                              words=[dict(w, x0=w["x0"] + x_off, x1=w["x1"] + x_off, y0=w["y0"] + y_off,
                                          y1=w["y1"] + y_off) for w in words])
        flat += cur
    return dict(scan=str(source) if not isinstance(source, np.ndarray) else None,
                header_rows=(top + y_off, bottom + y_off), probe_row=probe + y_off, tracks=out_tracks,
                curves=flat, runtime_s=round(time.time() - t0, 3),
                backend=backend if isinstance(backend, str) else getattr(backend, "__name__", "custom"),
                status="proposed")


def apply_word_edits(header: dict, track: str, edits: list[dict]) -> dict:
    """Correct OCR words of one track and re-fit its proposals (mnemonic, unit, scale) from them.

    edits: ``[{i, text}]`` (index into ``header['tracks'][track]['words']``); an empty text deletes the
    word. Typed text gets confidence 100 (the 0-100 OCR scale). Returns the header, updated in place,
    with ``corrected=True`` on edited words."""
    t = header["tracks"][track]
    words = [dict(w) for w in t["words"]]
    for e in edits:
        i = int(e["i"])
        if not 0 <= i < len(words):
            raise IndexError(f"word {i} not in track {track}")
        words[i].setdefault("ocr_text", words[i]["text"]); words[i].update(text=str(e["text"]).strip(), conf=100.0, corrected=True)
    t["words"] = words
    keep = [w for w in words if w["text"]]
    t["curves"] = propose_curves(keep, t["x"][0], t["x"][1], t["grid"] if t.get("grid") not in (None, "unknown") else None, track=track)
    header["curves"] = [c for nm in header["tracks"] for c in header["tracks"][nm]["curves"]]
    return header


def save_word_corrections(folder, ink_or_image: np.ndarray, words: list[dict], scan: str = "", offset=(0, 0), pad: int = 3) -> list[str]:
    """Keep (crop, text) pairs of hand-corrected words for later OCR evaluation: <folder>/<stem>_<n>.png
    + a line in <folder>/labels.tsv (file, text, original OCR text). offset = (x, y) subtracted from word
    boxes to index the image."""
    from PIL import Image
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); out = []
    img = ink_or_image
    if img.dtype == bool:
        img = np.where(img, 0, 255).astype(np.uint8)
    stem = Path(scan).stem or "scan"; n0 = len(list(folder.glob(f"{stem}_*.png")))
    with open(folder / "labels.tsv", "a", encoding="utf-8") as f:
        for k, w in enumerate(w for w in words if w.get("corrected")):
            x0, x1 = int(w["x0"] - offset[0]) - pad, int(w["x1"] - offset[0]) + pad
            y0, y1 = int(w["y0"] - offset[1]) - pad, int(w["y1"] - offset[1]) + pad
            crop = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if crop.size == 0:
                continue
            name = f"{stem}_{n0 + k:04d}.png"; Image.fromarray(crop).save(folder / name)
            f.write(f"{name}\t{w['text']}\t{w.get('ocr_text', '')}\n"); out.append(name)
    return out


def to_runs(header: dict, min_confidence: float = 0.0, scan: str | None = None) -> list[dict]:
    """Proposals -> dataset-profile style run dicts (curve, track, scale, scale_type, backup) with
    ``scale_source: header_ocr``, ``confirmed: False`` and the fields that need review."""
    runs = []
    for c in header["curves"]:
        if c["mnemonic"]["confidence"] < min_confidence:
            continue
        r = dict(curve=c["mnemonic"]["value"], track=c["track"], unit=c["unit"]["value"],
                 scale_type=c["scale_type"]["value"], scale_source="header_ocr", confirmed=False)
        if scan or header.get("scan"):
            r["scan"] = Path(scan or header["scan"]).name
        if c["scale_left"]["value"] is not None:
            r["scale"] = [c["scale_left"]["value"], c["scale_right"]["value"]]
        if c["backup"]["value"] is not None:
            r["backup"] = list(c["backup"]["value"])
        r["needs_review"] = [k for k in ("mnemonic", "unit", "scale_left", "scale_right", "scale_type", "backup")
                             if c[k]["needs_review"]]
        r["confidence"] = round(min(c[k]["confidence"] for k in ("mnemonic", "scale_left", "scale_type")), 3)
        runs.append(r)
    return runs


_BOTTOM_KEYS = ("btm", "bottom", "driller", "drillr", "logger", "welex", "schlum", "total depth", "last reading")
_TOP_KEYS = ("top log", "toplog", "top-log", "toptog", "first reading")


def header_depths(ink: np.ndarray, probe_row: int, rotations=(0, 1, 3)) -> dict:
    """Read the logged depth interval from the sheet header (above ``probe_row``): numbers on the
    lines naming 'Btm. log interval', 'Depth-Driller/Logger', 'Total depth' (-> ``bottom``) and
    'Top log interval' / 'First reading' (-> ``top``). Headers are often printed sideways, so a
    few rotations are read. Returns {bottom, top, lines}; values None when not found. Needs OCR."""
    import re
    from pylogdigitizer import ocr_engines
    out = dict(bottom=None, top=None, lines=[])
    if probe_row < 50 or ocr_engines.resolve("auto") is None:
        return out
    img = ink[:probe_row]
    img = np.where(img, 0, 255).astype(np.uint8) if img.dtype == bool else img
    num = re.compile(r"(?<![\d/@.,-])(\d{3,5})(?![\d/])")
    bots, tops = [], []
    for rot in rotations:
        words = ocr_engines.rapid_words(np.ascontiguousarray(np.rot90(img, rot)))
        for line in _group_lines(words):
            txt = " ".join(w["text"] for w in line)
            low = re.sub(r"[^a-z0-9 @/.\-]", " ", txt.lower())
            vals = [float(v) for v in num.findall(low)]
            vals = [v for v in vals if 100 <= v <= 40000]
            if not vals:
                continue
            if any(k in low for k in _TOP_KEYS):
                tops.append(min(vals)); out["lines"].append(txt)
            elif any(k in low for k in _BOTTOM_KEYS):
                bots.append(max(vals)); out["lines"].append(txt)
    if bots:
        out["bottom"] = float(np.median(bots))
    if tops:
        out["top"] = float(np.median(tops))
    return out


def depth_range_warning(top: float, base: float, header: dict, tol_ft: float = 100.0) -> str | None:
    """Plain-language warning when a digitized depth range clearly disagrees with the header."""
    hb, ht = header.get("bottom"), header.get("top")
    if hb and base > hb + max(tol_ft, 0.05 * hb):
        return (f"The traced depths ({top:.0f}-{base:.0f} ft) go deeper than the bottom depth written in the "
                f"header ({hb:.0f} ft). Check the depth numbers.")
    if ht and hb and base < ht - tol_ft:
        return (f"The traced depths ({top:.0f}-{base:.0f} ft) are above the logged interval written in the "
                f"header ({ht:.0f}-{hb:.0f} ft). Check the depth numbers.")
    return None
