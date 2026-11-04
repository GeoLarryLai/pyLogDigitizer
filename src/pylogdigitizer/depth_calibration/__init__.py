# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""depth_calibration: pixel row <-> depth.

Pipeline: find depth-label text blobs in the depth column -> read them (RapidOCR,
otherwise semi-automatic: you give the value of one label and the label interval) ->
snap labels to heavy grid lines -> optionally add every heavy grid line (e.g. every 10 ft) as an
anchor -> piecewise-linear map with leave-one-out residuals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import shutil
import numpy as np


@dataclass
class DepthMap:
    """Piecewise-linear row->depth map (linear extrapolation outside the anchors)."""
    rows: np.ndarray
    depths: np.ndarray
    info: dict = field(default_factory=dict)

    def __call__(self, row):
        r = np.asarray(row, float)
        d = np.interp(r, self.rows, self.depths)
        lo, hi = r < self.rows[0], r > self.rows[-1]
        if lo.any() or hi.any():
            s0 = (self.depths[1] - self.depths[0]) / (self.rows[1] - self.rows[0])
            s1 = (self.depths[-1] - self.depths[-2]) / (self.rows[-1] - self.rows[-2])
            d = np.where(lo, self.depths[0] + (r - self.rows[0]) * s0, d)
            d = np.where(hi, self.depths[-1] + (r - self.rows[-1]) * s1, d)
        return d

    def inverse(self, depth):
        d = np.asarray(depth, float)
        r = np.interp(d, self.depths, self.rows)
        lo, hi = d < self.depths[0], d > self.depths[-1]
        if lo.any() or hi.any():
            s0 = (self.rows[1] - self.rows[0]) / (self.depths[1] - self.depths[0])
            s1 = (self.rows[-1] - self.rows[-2]) / (self.depths[-1] - self.depths[-2])
            r = np.where(lo, self.rows[0] + (d - self.depths[0]) * s0, r)
            r = np.where(hi, self.rows[-1] + (d - self.depths[-1]) * s1, r)
        return r

    @property
    def px_per_ft(self) -> float:
        return float((self.rows[-1] - self.rows[0]) / (self.depths[-1] - self.depths[0]))


def find_label_rows(depth_col_ink: np.ndarray, min_chars: int = 3, char_h: tuple[int, int] = (12, 45),
                    char_w: tuple[int, int] = (4, 40), line_len: int = 41) -> list[dict]:
    """Detect text lines (depth labels) in the depth-column strip. Long straight lines are removed
    first; remaining character-sized components are grouped by row. Returns
    [{row, x0, x1, n_chars, y0, y1}] sorted by row."""
    import cv2
    from pylogdigitizer.scan_cleanup import line_masks
    hm, vm = line_masks(depth_col_ink, line_len)
    t = (depth_col_ink & ~hm & ~vm).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(t, 8)
    ok = ((st[:, 3] >= char_h[0]) & (st[:, 3] <= char_h[1]) & (st[:, 2] >= char_w[0]) & (st[:, 2] <= char_w[1]))
    ok[0] = False
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(cen[idx, 1])]
    groups, cur = [], []
    for i in idx:
        if cur and cen[i, 1] - cen[cur[-1], 1] >= char_h[0]:
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len(g) < min_chars:
            continue
        g = np.array(g)
        out.append(dict(row=float(cen[g, 1].mean()), x0=int(st[g, 0].min()), x1=int((st[g, 0] + st[g, 2]).max()),
                        y0=int(st[g, 1].min()), y1=int((st[g, 1] + st[g, 3]).max()), n_chars=len(g)))
    if len(out) >= 3:  # drop tick marks / stray symbols: labels share a typical text width
        wid = np.array([o["x1"] - o["x0"] for o in out])
        ref = np.median(wid[wid >= np.percentile(wid, 25)])
        out = [o for o, w_ in zip(out, wid) if w_ >= 0.6 * ref]
    return out


def find_label_blobs(depth_col_ink: np.ndarray, line_len: int = 41, merge: int = 25,
                     min_size: tuple[int, int] = (15, 15), max_size: tuple[int, int] = (300, 200)) -> list[dict]:
    """Orientation-independent label finder (rotated depth numbers, e.g. older Schlumberger
    prints): remove long lines, drop specks, dilate characters into word blobs, keep blobs of
    typical label size. Returns [{row, x0, x1, y0, y1, n_chars}] sorted by row."""
    import cv2
    from pylogdigitizer.scan_cleanup import line_masks
    hm, vm = line_masks(depth_col_ink, line_len)
    t = (depth_col_ink & ~hm & ~vm).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(t, 8)
    keep = (st[:, 4] >= 12); keep[0] = False
    t = keep[lab].astype(np.uint8)
    blob = cv2.dilate(t, np.ones((merge, merge), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(blob, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        h_, w_ = h - merge + 1, w - merge + 1
        if min_size[0] <= h_ <= max_size[0] and min_size[1] <= w_ <= max_size[1]:
            sub = t[y:y + h, x:x + w]
            nc = cv2.connectedComponents(sub)[0] - 1
            if nc >= 2:
                out.append(dict(row=float(y + h / 2), x0=int(x + merge // 2), x1=int(x + w - merge // 2),
                                y0=int(y + merge // 2), y1=int(y + h - merge // 2), n_chars=int(nc)))
    # no size-consensus filter here: header/curve junk can outnumber labels; the regular label
    # lattice (regular_rows) is the filter
    return sorted(out, key=lambda o: o["row"])


def lattice_spacing(rows) -> float:
    """Label spacing robust to missing labels: diffs are divided by their multiple of the
    smallest typical diff."""
    d = np.diff(np.sort(np.asarray(rows, float)))
    d = d[d > 0]
    if not len(d):
        return float("nan")
    d0 = np.percentile(d, 10)
    k = np.maximum(1, np.round(d / d0))
    return float(np.median(d / k))


def _chain(r: np.ndarray, s: float, rel_tol: float, max_gap: int) -> list[int]:
    n = len(r); best = []
    for i in range(n):
        chain = [i]
        while True:
            last = chain[-1]; nxt = None
            for j in range(last + 1, n):
                dd = r[j] - r[last]; k = round(dd / s)
                if k > max_gap:
                    break
                if k >= 1 and abs(dd - k * s) <= rel_tol * s * np.sqrt(k):
                    nxt = j; break
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) > len(best):
            best = chain
    return best


def regular_rows(rows, rel_tol: float = 0.04, min_spacing: float = 100.0, max_gap: int = 4,
                 sizes=None) -> np.ndarray:
    """Indices of the label lattice in ``rows``: the chain of rows whose consecutive differences
    are ~k * s (k = 1..max_gap, |diff - k*s| <= rel_tol*s*sqrt(k)). The spacing s is chosen
    among the observed differences as the one giving the longest chain (labels are nearly
    equally spaced - paper stretch allowed - while junk blobs are not)."""
    order = np.argsort(np.asarray(rows, float))
    r = np.asarray(rows, float)[order]; n = len(r)
    sz = np.asarray(sizes, float)[order] if sizes is not None else None
    if n < 3:
        return order
    cand = sorted({round(float(r[j] - r[i])) for i in range(n) for j in range(i + 1, min(n, i + 4))
                   if r[j] - r[i] >= min_spacing})
    # thin candidates to ~2 % bins
    thin = []
    for c in cand:
        if not thin or c > thin[-1] * (1 + rel_tol / 2):
            thin.append(c)
    best, best_cov = [], -1.0
    for c in thin:
        ch = _chain(r, c, rel_tol, max_gap)
        if len(ch) < 3:
            continue
        cov = r[ch[-1]] - r[ch[0]]
        score = len(ch) + cov / (r[-1] - r[0] + 1)     # most labels; coverage breaks ties
        if sz is not None:   # real labels share one size (text height x width), junk does not
            z = sz[ch]
            cv = float(np.mean(np.std(z, axis=0) / np.maximum(1, np.mean(z, axis=0))))
            score *= max(0.1, 1 - 2 * cv)
        if score > best_cov:
            best, best_cov = ch, score
    return np.sort(order[np.array(best, int)]) if best else np.array([], int)


def ocr_available(engine: str = "auto") -> bool:
    """True when the requested OCR engine (see :mod:`pylogdigitizer.ocr_engines`) can be used."""
    from pylogdigitizer import ocr_engines
    try:
        return ocr_engines.resolve(engine) is not None
    except RuntimeError:
        return False


def _ocr_crop(img: np.ndarray, rot: int, scale: int = 1, psm: int = 7, engine: str = "rapidocr") -> str:
    """Read one label crop. ``psm`` is kept for the variant tuples; ``psm=9`` adds a light blur
    after upscaling, which helps tiny, thin or broken digits (small rotated labels)."""
    from pylogdigitizer import ocr_engines
    c = np.rot90(img, rot) if rot else img
    if psm == 9:            # small-print read: upscale, blur, pad, then full detect + recognise
        import cv2
        c = cv2.resize(np.ascontiguousarray(c), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        c = np.pad(cv2.GaussianBlur(c, (5, 5), 0), 30, constant_values=255)
        words = ocr_engines.rapid_words(c, min_conf=0)
        return max((w["text"] for w in words), key=len, default="")
    c = np.pad(np.ascontiguousarray(c), 12 * scale, constant_values=255)
    if scale > 1:
        import cv2
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return ocr_engines.rapid_line(c)[0]


def _ocr_vote(crop: np.ndarray, rot: int, variants=((1, 7), (2, 7)), engine: str = "rapidocr") -> float | None:
    """Most common 2-5 digit read over a few scale / page-segmentation variants."""
    reads: list[float] = []
    cnt: dict[float, int] = {}
    n_var = len(variants)
    for vi, (sc_, psm) in enumerate(variants):
        d = "".join(ch for ch in _ocr_crop(crop, rot, sc_, psm, engine) if ch.isdigit())
        if 2 <= len(d) <= 5:
            val = float(d)
            reads.append(val)
            cnt[val] = cnt.get(val, 0) + 1
            rem = n_var - 1 - vi
            if rem > 0 and len(cnt) >= 1:
                best = max(cnt, key=lambda r_: (cnt[r_], -reads.index(r_)))
                bc = cnt[best]
                bi = reads.index(best)
                # Early exit when no remaining variant outcomes can change the winner
                if (bc > rem or (bc == rem and bi == 0)) and all(
                    bc > c_ + rem or (bc == c_ + rem and bi < reads.index(r_))
                    for r_, c_ in cnt.items() if r_ != best
                ):
                    return float(best)
    if not reads:
        return None
    # ties go to the earliest variant (the plain 1x / psm 7 read), not the smallest number
    return float(max(cnt, key=lambda r_: (cnt[r_], -reads.index(r_))))


def ocr_depth_labels(depth_col_image: np.ndarray, labels: list[dict] | None = None,
                     rotations: tuple[int, ...] | None = None, pad: int = 4,
                     max_workers: int = 4, engine: str = "auto") -> list[tuple[float, float | None]]:
    """OCR each label box (digits only). Returns [(row, value or None)].

    Labels may be printed rotated (older Schlumberger prints): with ``rotations=None`` the
    orientation (0, 90 or 270 deg) is chosen on the first few labels as the one giving the most
    3-5 digit reads. ``engine``: 'auto'/'rapidocr' or 'none'
    (see :mod:`pylogdigitizer.ocr_engines`). Without any OCR all values are None (use :func:`assign_depths`)."""
    if labels is None:
        ink = depth_col_image if depth_col_image.dtype == bool else depth_col_image < 128
        labels = find_label_rows(ink)
    from pylogdigitizer import ocr_engines
    engine = ocr_engines.resolve(engine)
    if engine is None:
        return [(l["row"], None) for l in labels]
    max_workers = 1                # deterministic recogniser, one shared model: one read per scale
    variants = ((1, 7), (2, 7))
    img = depth_col_image
    if img.dtype == bool:
        img = np.where(img, 0, 255).astype(np.uint8)
    crops = [img[max(0, l["y0"] - pad):l["y1"] + pad, max(0, l["x0"] - pad):l["x1"] + pad] for l in labels]
    if rotations is None:
        tall = np.median([c.shape[0] / max(1, c.shape[1]) for c in crops]) > 1.3 if crops else False
        cand = (1, 3) if tall else (0,)
        sample = crops[:: max(1, len(crops) // 6)][:6]
        def score(rot):
            return sum(3 <= len("".join(ch for ch in _ocr_crop(c, rot, engine=engine) if ch.isdigit())) <= 5 for c in sample)
        rotations = (max(cand, key=score),) if len(cand) > 1 else cand
    def _read_one(c):
        for rot in rotations:
            v = _ocr_vote(c, rot, variants, engine)
            if v is not None:
                return v
        return None
    if max_workers > 1 and len(crops) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            vals = list(ex.map(_read_one, crops))
    else:
        vals = [_read_one(c) for c in crops]
    if len(crops) >= 3 and infer_label_values([l["row"] for l in labels], vals) is None:
        # small-print fallback: tiny labels (e.g. rotated "0600") need a bigger, slightly blurred read
        tall = np.median([c.shape[0] / max(1, c.shape[1]) for c in crops]) > 1.3
        crops10 = [img[max(0, l["y0"] - 10):l["y1"] + 10, max(0, l["x0"] - 10):l["x1"] + 10] for l in labels]
        best = None
        for rot in ((1, 3) if tall else (0,)):
            v2 = [_ocr_vote(c, rot, ((3, 9), (4, 9)), engine) for c in crops10]
            inf = infer_label_values([l["row"] for l in labels], v2)
            n = inf[2] if inf else 0
            if n and (best is None or n > best[0]):
                best = (n, v2)
        if best is not None:
            vals = best[1]
    return [(l["row"], v) for l, v in zip(labels, vals)]


def infer_label_values(rows, ocr_values, intervals=(100, 50, 200, 500, 20, 10, 1000),
                       px_per_ft_range=(1.5, 60.0)) -> tuple[float, float, int] | None:
    """From OCR reads on a label lattice, infer (first_value, interval, n_agree): the label at
    lattice index k has value first + k*interval. Misreads are outvoted. None if < 3 agree."""
    r = np.asarray(rows, float)
    if len(r) < 3:
        return None
    sp = lattice_spacing(r)
    k = np.round((r - r[0]) / sp)
    best = None
    for iv in intervals:
        if not (px_per_ft_range[0] <= sp / iv <= px_per_ft_range[1]):
            continue            # interval implausible for this label spacing
        base = [v - kk * iv for v, kk in zip(ocr_values, k) if v is not None]
        if not base:
            continue
        vals, cnt = np.unique(np.round(base, 1), return_counts=True)
        j = int(np.argmax(cnt))
        if best is None or cnt[j] > best[2]:
            best = (float(vals[j]), float(iv), int(cnt[j]))
    return best if best and best[2] >= 3 else None


def choose_labels(depth_col_ink: np.ndarray, row_min: float = 0, edge_trim: int = 5) -> tuple[list[dict], str, dict]:
    """Pick the depth-label set: horizontal text lines (``find_label_rows``) or orientation-free
    blobs (``find_label_blobs``), whichever gives the longest regular lattice (``regular_rows``);
    ties go to the larger sheet coverage. Returns (labels on the lattice, mode, scores)."""
    t = depth_col_ink[:, edge_trim:-edge_trim] if depth_col_ink.shape[1] > 4 * edge_trim else depth_col_ink
    scores, best = {}, ([], "none")
    h = depth_col_ink.shape[0]
    for mode, fn in (("rows", find_label_rows), ("blobs", find_label_blobs)):
        try:
            labs = [l for l in fn(t) if l["row"] >= row_min]
        except Exception:
            labs = []
        if len(labs) < 3:
            scores[mode] = (0, 0.0); continue
        idx = regular_rows([l["row"] for l in labs], sizes=[(l["y1"] - l["y0"], l["x1"] - l["x0"]) for l in labs])
        sel = sorted((labs[i] for i in idx), key=lambda l: l["row"])
        cov = (sel[-1]["row"] - sel[0]["row"]) / max(1, h - row_min) if len(sel) >= 2 else 0.0
        scores[mode] = (len(sel), round(float(cov), 3))
        if (len(sel), cov) > (len(best[0]), scores.get(best[1], (0, 0))[1]):
            best = (sel, mode)
    return best[0], best[1], scores


class CalibrationError(ValueError):
    """Depth calibration failed or is implausible; the message says what to check."""


def validate_depth_map(dm: "DepthMap", anchors, residuals_ft, sheet_rows: int, dpi: float = 300.0,
                       px_per_ft_range: tuple[float, float] = (3.0, 30.0), max_residual_ft: float = 5.0,
                       min_labels: int = 3, min_coverage: float = 0.05, n_labels: int | None = None) -> dict:
    """Plausibility checks; raises CalibrationError with a hint. Returns the check values.
    At 300 dpi the usual depth scales (1:1200 .. 1:240 i.e. 1-5 in per 100 ft) give
    3-15 px/ft; 1:200 detail logs about 18 px/ft."""
    ppf = dm.px_per_ft * 300.0 / dpi
    a = np.array(sorted(anchors), float)
    cov = (a[-1, 0] - a[0, 0]) / max(1, sheet_rows) if len(a) >= 2 else 0.0
    res = float(np.nanmax(np.abs(residuals_ft))) if np.isfinite(residuals_ft).any() else 0.0
    chk = dict(px_per_ft=round(float(dm.px_per_ft), 3), coverage=round(float(cov), 3), loo_max_ft=round(res, 2),
               n_labels=n_labels)
    if not (px_per_ft_range[0] <= ppf <= px_per_ft_range[1]):
        raise CalibrationError(f"implausible depth scale {dm.px_per_ft:.2f} px/ft (expected {px_per_ft_range} at 300 dpi): "
                               "labels were probably not depth labels or the label interval is wrong - check the "
                               "depth column detection or give label_interval / first_depth")
    if n_labels is not None and n_labels < min_labels:
        raise CalibrationError(f"only {n_labels} depth labels found (need {min_labels})")
    if cov < min_coverage:
        raise CalibrationError(f"depth anchors cover only {cov:.0%} of the sheet - labels likely from one short "
                               "section or header; give rows= or check label detection")
    if res > max_residual_ft:
        raise CalibrationError(f"depth anchors inconsistent (leave-one-out residual {res:.1f} ft > {max_residual_ft}); "
                               "a label was misread or a splice/repeat section is mixed in")
    return chk


def assign_depths(label_rows, first_depth: float, interval: float = 100.0, first_index: int = 0) -> list[tuple[float, float]]:
    """Semi-automatic labelling: label ``first_index`` has value ``first_depth``; following labels
    increase by ``interval`` per label spacing (gaps of ~k spacings = k-1 missing labels)."""
    r = np.asarray(label_rows, float)[first_index:]
    d = np.diff(r)
    sp = lattice_spacing(r)
    k = np.r_[0, np.cumsum(np.maximum(1, np.round(d / sp)))]
    return [(float(a), float(first_depth + interval * b)) for a, b in zip(r, k)]


def check_ocr_sequence(pairs: list[tuple[float, float | None]], interval: float = 100.0) -> list[tuple[float, float]]:
    """Keep OCR'd labels consistent with a linear row->depth trend (rejects misreads)."""
    good = [(r, v) for r, v in pairs if v is not None]
    if len(good) < 3:
        return good
    r, v = np.array(good).T
    A = np.c_[r, np.ones_like(r)]
    coef = np.linalg.lstsq(A, v, rcond=None)[0]
    res = v - A @ coef
    return [g for g, e in zip(good, res) if abs(e) < interval / 2]


def snap_to_lines(anchors: list[tuple[float, float]], line_rows: np.ndarray, max_dist: float = 25) -> list[tuple[float, float]]:
    """Move each anchor row to the nearest grid line within ``max_dist`` px."""
    lr = np.asarray(line_rows, float)
    out = []
    for r, d in anchors:
        if len(lr):
            j = np.argmin(abs(lr - r))
            if abs(lr[j] - r) <= max_dist:
                r = lr[j]
        out.append((float(r), d))
    return out


def densify_with_grid(anchors: list[tuple[float, float]], line_rows: np.ndarray, ft_per_line: float,
                      tol: float = 0.3) -> list[tuple[float, float]]:
    """Add every grid line (spaced ``ft_per_line``) between anchors as an anchor, assigning the
    nearest multiple of ``ft_per_line`` predicted by the label anchors (drops ambiguous lines)."""
    a = sorted(anchors)
    rows = np.array([x[0] for x in a]); deps = np.array([x[1] for x in a])
    lr = np.asarray(line_rows, float)
    lr = lr[(lr > rows[0]) & (lr < rows[-1])]
    pred = np.interp(lr, rows, deps)
    q = np.round(pred / ft_per_line) * ft_per_line
    ok = abs(pred - q) < tol * ft_per_line
    extra = dict(zip(q[ok], lr[ok]))
    for r, d in a:
        extra[d] = r
    return sorted((float(r), float(d)) for d, r in extra.items())


def fit_depth_map(anchors: list[tuple[float, float]]) -> DepthMap:
    """Piecewise-linear map through (row, depth) anchors (needs >= 2)."""
    a = np.array(sorted(anchors), float)
    if len(a) < 2:
        raise ValueError("need at least two depth anchors")
    m = DepthMap(a[:, 0], a[:, 1])
    m.info["px_per_ft"] = m.px_per_ft
    return m


def residuals(anchors: list[tuple[float, float]], depth_map: DepthMap | None = None) -> np.ndarray:
    """Leave-one-out residuals (ft) of each interior anchor; large values flag misread labels or
    stretch. Ends get NaN."""
    a = np.array(sorted(anchors), float)
    out = np.full(len(a), np.nan)
    for i in range(1, len(a) - 1):
        out[i] = a[i, 1] - np.interp(a[i, 0], np.r_[a[:i, 0], a[i + 1:, 0]], np.r_[a[:i, 1], a[i + 1:, 1]])
    return out


def calibrate_labels(label_rows, heavy_rows, values: list | None = None, first_depth: float | None = None,
                     interval: float | None = None, line_ft: float | None = None, snap_px: float = 25) -> tuple:
    """Depth map from labels on a lattice. Label values come from OCR reads (``values``, outvoted
    by :func:`infer_label_values`) or, semi-automatically, from ``first_depth`` + ``interval``
    per lattice step. Labels are snapped to ``heavy_rows`` (heavy grid lines) and every heavy line
    becomes an anchor. Returns (DepthMap, anchors, loo_residuals, info). Raises CalibrationError."""
    rows = np.asarray(label_rows, float)
    if len(rows) < 3:
        raise CalibrationError(f"only {len(rows)} depth labels on a regular lattice (need 3): the depth column "
                               "was not found or labels are unreadable - check the detection figure")
    sp_init = lattice_spacing(rows)
    if np.isfinite(sp_init) and sp_init > 0:
        keep_idx = [0]
        for i in range(1, len(rows)):
            if rows[i] - rows[keep_idx[-1]] >= 0.65 * sp_init:
                keep_idx.append(i)
        if len(keep_idx) >= 3 and len(keep_idx) < len(rows):
            rows = rows[keep_idx]
            if values is not None:
                values = [values[i] for i in keep_idx]
    info = {}
    inferred = infer_label_values(rows, values) if values is not None else None
    if inferred is not None:
        d0, iv, nag = inferred
        user_first = first_depth
        if interval is not None and iv != interval:
            info["ocr_interval"] = iv
        first_depth, interval = d0, iv
        info.update(source="ocr", ocr_agree=nag, ocr_read=sum(v is not None for v in values))
    else:
        if first_depth is None:
            raise CalibrationError(
                "The depth numbers on the scan could not be read. Tell it one known depth: pass "
                "first_depth=<depth of the first depth number below the header> (and label_interval=<step "
                "between numbers, default 100>) to digitize_curve()/calibrate_depth(), or set first_depth "
                "in the wells config. In the app, type them under 'Or: first label + step'.")
        info["source"] = "first_depth"
    interval = interval or 100.0
    pairs = assign_depths(rows, first_depth, interval)
    if info.get("source") == "ocr":
        # keep only labels whose own read confirms the lattice value: unread / misread blobs at the
        # lattice ends (header text, junk) would otherwise shift the k-indexing
        good = [v is not None and abs(v - p_[1]) < 1e-6 for p_, v in zip(pairs, values)]
        bad = [v is not None and not g for v, g in zip(values, good)]
        idx = [i for i, g in enumerate(good) if g]
        lo, hi = (idx[0], idx[-1]) if idx else (0, -1)
        # misreads inside the lattice keep their lattice value; labels before the first
        # confirmed read are dropped when unread or contradicted (header text, junk blobs),
        # and trailing repeat-section labels after a gap/contradiction beyond hi are dropped
        sp_ok = lattice_spacing(rows)
        end_i = len(pairs) - 1
        if hi >= lo:
            end_i = hi
            for i in range(hi + 1, len(pairs)):
                if pairs[i][0] - pairs[i - 1][0] > 1.35 * sp_ok:
                    break
                if values[i] is not None and values[i] >= 100 and values[i] < pairs[i][1] - 1.5 * interval:
                    break
                end_i = i
        ok = [
            p_ for i, p_ in enumerate(pairs)
            if i <= end_i and (i >= lo or not (bad[i] or values[i] is None))
        ]
        if len(ok) >= 3:
            info["n_ocr_dropped"] = len(pairs) - len(ok)
            pairs = ok
        # ``first_depth`` means the first depth number printed below the header: compare it with the
        # first label OCR confirmed, not with lattice index 0 (often header text or a junk blob)
        if user_first is not None and pairs and abs(user_first - pairs[0][1]) > 1e-6:
            info["ocr_disagrees_with_first_depth"] = (float(pairs[0][1]), user_first)
    heavy = np.asarray(heavy_rows, float)
    anchors = snap_to_lines(pairs, heavy, max_dist=snap_px)
    if line_ft is None and len(heavy) > 3:
        line_ft = estimate_line_ft(anchors, heavy)
    if line_ft:
        anchors = densify_with_grid(anchors, heavy, line_ft)
    anchors, dropped = prune_anchors(anchors)
    dm = fit_depth_map(anchors)
    info.update(first_depth=first_depth, interval=interval, line_ft=line_ft, n_labels=len(rows), n_pruned=dropped)
    dm.info.update(info)
    return dm, anchors, residuals(anchors), info


def estimate_line_ft(anchors, heavy_rows, candidates=(5, 10, 20, 25, 50)) -> float | None:
    """Heavy grid-line interval in ft: the candidate that best explains the spacings of detected
    heavy lines (curves hide some lines, so multiples are allowed); None if no candidate
    explains at least half of them."""
    a = np.array(sorted(anchors), float)
    px_ft = np.median(np.diff(a[:, 0]) / np.diff(a[:, 1]))
    d = np.diff(np.sort(np.asarray(heavy_rows, float))) / px_ft
    d = d[d > 0.5]
    if len(d) < 3:
        return None
    best, best_score = None, 0.0
    for c in candidates:
        k = np.round(d / c)
        ok = (k >= 1) & (np.abs(d - k * c) < 0.15 * c)
        score = ok.sum() / k[ok].sum() if ok.any() else 0      # prefer few missing lines
        frac = ok.mean()
        if frac >= 0.5 and score * frac > best_score:
            best, best_score = float(c), score * frac
    return best


def prune_anchors(anchors, max_res_ft: float = 1.5, max_frac: float = 0.2) -> tuple[list, int]:
    """Drop the worst leave-one-out outliers (a heavy line hidden by a curve and the next line
    taken instead, a mis-snapped label) one at a time while the worst residual > ``max_res_ft``,
    at most ``max_frac`` of the anchors. Returns (anchors, n_dropped)."""
    a = sorted(anchors)
    n0 = len(a)
    while len(a) > 4 and n0 - len(a) < max_frac * n0:
        res = residuals(a)
        if not np.isfinite(res).any():
            break
        i = int(np.nanargmax(np.abs(res)))
        if abs(res[i]) <= max_res_ft:
            break
        a.pop(i)
    while len(a) >= 6 and n0 - len(a) < max_frac * n0:
        arr = np.array(a, float)
        med = float(np.median(np.diff(arr[:, 0]) / np.diff(arr[:, 1])))
        if med <= 0:
            break
        tail_err = abs((a[-1][0] - a[-2][0]) / med - (a[-1][1] - a[-2][1]))
        if tail_err > max(5.0, 3.0 * max_res_ft):
            a.pop(-1)
            continue
        head_err = abs((a[1][0] - a[0][0]) / med - (a[1][1] - a[0][1]))
        if head_err > max(5.0, 3.0 * max_res_ft):
            a.pop(0)
            continue
        break
    return a, n0 - len(a)
