# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""curve_tracing: follow one curve through a track.

1. ``build_cost``: thick-stroke ink -> cost 0, paper -> 1, detected grid lines -> neutral (so the
   path neither follows nor avoids them); block-min downsampling for speed.
2. ``build_ml_cost``: optional 2D U-Net cost-map refinement (`cost_mode="ml"` or `"blend"`) that
   penalizes non-target distractor ink (such as stationary Caliper / Bit Size / SP curves) on
   multi-curve tracks while keeping the same dynamic-programming tracer.
3. ``trace``: dynamic programming, one column per row. Step cost =
   cost[y, x] + smooth*|dx| + seg_weight * mean(min(cost[y], cost[y-1])[span(px..x)]): a big horizontal jump is
   cheap only if ink connects it (horizontal excursions of the curve), so the path does not hop
   between curves. Anchors pin the path to user points (rows) within ``anchor_tol`` columns.
Backends: Rust ``curve_tracing_rs`` (if built with maturin) > numba > NumPy reference.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

try:
    try:
        from pylogdigitizer import curve_tracing_rs as _rs  # type: ignore[attr-defined]
    except ImportError:
        import curve_tracing_rs as _rs  # maturin wheel installs a top-level module
    HAVE_RUST = hasattr(_rs, "trace_dp_lab")
except ImportError:
    _rs, HAVE_RUST = None, False

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    numba, HAVE_NUMBA = None, False


if not HAVE_RUST:
    import warnings
    warnings.warn("pyLogDigitizer: the compiled Rust tracer is missing, so tracing uses the "
                  + ("numba" if HAVE_NUMBA else "pure-Python (much slower)") + " fallback. "
                  "Install a built wheel (pip install .) or run `maturin develop --release` in the project folder.",
                  RuntimeWarning, stacklevel=2)


def backend() -> str:
    return "rust" if HAVE_RUST else ("numba" if HAVE_NUMBA else "python")


def cost_from_ink(binary_ink: np.ndarray) -> np.ndarray:
    """Ink mask (True=ink) -> float cost (0 on ink, 1 off ink)."""
    return np.where(binary_ink, 0.0, 1.0).astype(np.float64)


def build_cost(track_ink: np.ndarray, grid_cols: list[tuple[int, int]] | None = None,
               grid_rows_mask: np.ndarray | None = None, min_width: int = 3, neutral: float = 0.6, line_neutral: float = 0.85,
               fy: int = 3, fx: int = 2, line_frac: float = 0.6, min_area: int = 0,
               straight_len: int = 0, return_labels: bool = False, line_row_cost: float = 0.8):
    """Cost image for a track window (rows x cols), downsampled by (fy, fx).

    - ink thinner than ``min_width`` (thin grid lines, specks) is dropped (morphological opening);
    - components smaller than ``min_area`` px are dropped (isolated dots, text specks);
    - horizontal grid lines (rows with >= ``line_frac`` ink, or ``grid_rows_mask``), the
      ``grid_cols`` column ranges and perfectly vertical runs longer than ``straight_len`` px
      (heavy grid lines, flat curves such as an in-gauge caliper) become ``neutral``, so the path
      can cross them but is not attracted to them.
    Masks are applied after downsampling so that neutral areas survive the block reduction.
    With ``return_labels`` also returns stroke component ids per cell (curve-identity term).
    """
    from pylogdigitizer.scan_cleanup import thick_strokes, line_masks
    if min_width == 0:   # thin-curve mode: remove long straight grid lines instead of opening
        hm, vm = line_masks(track_ink, max(61, track_ink.shape[1] // 4))
        thick = track_ink & ~hm & ~vm
    else:
        thick = thick_strokes(track_ink, min_width) if min_width > 1 else track_ink
    if grid_rows_mask is None:
        grid_rows_mask = track_ink.mean(axis=1) >= line_frac
    heavy = thick.mean(axis=1) >= line_frac
    rows_all = grid_rows_mask | heavy
    # curve ink: grid-row bands removed and bridged where a stroke continues above and below
    curve = bridge_rows(thick, rows_all)
    if min_area:
        curve = remove_small_components(curve, min_area)
    line_ink = thick & rows_all[:, None] & ~curve
    h, w = thick.shape
    h2, w2 = h // fy, w // fx
    ds = lambda m: m[:h2 * fy, :w2 * fx].reshape(h2, fy, w2, fx).any(axis=(1, 3))
    ink = ds(curve)
    cost = np.where(ink, 0.0, 1.0)
    cost[ds(line_ink) & ~ink] = line_row_cost
    thick = curve
    if straight_len:
        import cv2
        L = max(3, straight_len // fy)
        wide = cv2.dilate(ink.astype(np.uint8), np.ones((1, 3), np.uint8))   # tolerate 1-col jitter
        vr = cv2.morphologyEx(wide, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        vr = cv2.dilate(vr, np.ones((1, 5), np.uint8)).astype(bool)
        cost[vr & ink] = line_neutral
    for x0, x1 in grid_cols or []:
        a, b = max(0, x0 // fx), min(w2, -(-x1 // fx))
        sl = cost[:, a:b]
        sl[sl == 0.0] = line_neutral
    if not return_labels:
        return cost
    lab = stroke_labels(thick, fy, fx)
    lab[cost != 0.0] = 0      # only real curve ink carries an identity
    return cost, lab


def build_ml_cost(
    cost: np.ndarray,
    prob_map: np.ndarray,
    v61: np.ndarray | None = None,
    rel_x: np.ndarray | None = None,
    mode: str = "blend",
    ml_weight: float = 0.15,
    caliper_weight: float = 0.55,
) -> np.ndarray:
    """Combine conventional DP cost matrix with 2D U-Net curve probability map.

    On curve ink cells (`cost == 0.0`), computes row-normalized disfavor from `prob_map`
    so the highest-probability curve on each row remains near 0.0 while competing
    distractor curves pay a soft penalty (capped at 0.48 so ink continuity holds).
    """
    if mode == "conventional":
        return np.asarray(cost, dtype=np.float64)
    if mode not in ("ml", "blend"):
        raise ValueError(f"Unsupported cost_mode={mode!r}; expected 'conventional', 'ml', or 'blend'")

    c0 = cost == 0.0
    prob = np.asarray(prob_map, dtype=np.float64)
    p_max = np.where(c0, prob, -1.0).max(axis=1, keepdims=True)
    p_min = np.where(c0, prob, 2.0).min(axis=1, keepdims=True)
    disfavor = np.where(
        c0 & (p_max > p_min + 0.01),
        np.clip((p_max - prob) / np.maximum(p_max - p_min, 0.05), 0.0, 1.0),
        0.0,
    )
    pen = float(ml_weight) * disfavor
    if v61 is not None and rel_x is not None and caliper_weight > 0.0:
        stat_cal = (v61 > 0.30) & (rel_x < 0.46) & (disfavor > 0.25)
        pen = pen + np.where(stat_cal, float(caliper_weight) * disfavor, 0.0)
    if mode == "ml":
        pen = pen + 0.05 * (1.0 - np.clip(prob / np.maximum(p_max, 0.05), 0.0, 1.0))
    pen = np.clip(pen, 0.0, 0.48)
    return np.where(c0, pen, cost).astype(np.float64)


def ml_available() -> bool:
    """True if the optional torch package is installed (the ML cost map needs it)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def ml_fallback(cost_mode: str, ml_model: Any = None) -> str:
    """Return ``cost_mode``, or ``"conventional"`` (with a warning) when torch or a model checkpoint is missing."""
    if cost_mode == "conventional":
        return cost_mode
    import warnings
    if not ml_available():
        warnings.warn(f"cost_mode={cost_mode!r} needs torch (pip install torch); using the conventional cost map",
                      RuntimeWarning, stacklevel=3)
        return "conventional"
    try:
        _resolve_default_ml_model(ml_model)
    except FileNotFoundError as e:
        warnings.warn(f"{e} Using the conventional cost map.", RuntimeWarning, stacklevel=3)
        return "conventional"
    return cost_mode


def _resolve_default_ml_model(ml_model: Any) -> Any:
    if ml_model is not None:
        return ml_model
    repo_root = Path(__file__).resolve().parents[2]
    for cand in (
        Path("models/kansas_gr_unet.pt"),
        Path("models/kansas_kgs/unet.pt"),
        repo_root / "models" / "kansas_gr_unet.pt",
        repo_root / "models" / "kansas_kgs" / "unet.pt",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No ML model checkpoint found at models/kansas_gr_unet.pt; pass ml_model=... or train one first."
    )


def stroke_widths(panel: np.ndarray, sample_rows: int = 6000) -> tuple[float, float]:
    """(curve stroke width, vertical grid-line width) in px, estimated on a sample of the panel.
    Curve width = median over non-grid ink of min(horizontal run, vertical run) through the pixel."""
    from pylogdigitizer import track_grid_detection as tgd
    from pylogdigitizer.scan_cleanup import line_masks
    s = panel[len(panel) // 3: len(panel) // 3 + sample_rows]
    if s.shape[0] < 100:
        s = panel
    vl = tgd.vertical_lines(s, 0.5)
    gw = float(np.median([l["x1"] - l["x0"] + 1 for l in vl])) if vl else 1.0
    hm, vm = line_masks(s, max(61, s.shape[1] // 4))
    ink = s & ~hm & ~vm
    def runs(a):
        d = np.diff(np.c_[np.zeros((a.shape[0], 1), np.int8), a.astype(np.int8), np.zeros((a.shape[0], 1), np.int8)], axis=1)
        st = np.argwhere(d == 1); en = np.argwhere(d == -1)
        out = np.zeros(a.shape, np.int32)
        for (r, a0), (_, a1) in zip(st, en):
            out[r, a0:a1] = a1 - a0
        return out
    hr = runs(ink); vr = runs(ink.T).T
    w = np.minimum(hr, vr)[ink]
    return (float(np.median(w)) if w.size else 0.0), gw


def stroke_labels(thick: np.ndarray, fy: int, fx: int, chunk: int = 20000) -> np.ndarray:
    """Connected-component id of the ink in each (fy x fx) cell (max id in the cell), computed in
    row chunks (ids made unique across chunks). 0 = no ink."""
    import cv2
    h, w = thick.shape
    h2, w2 = h // fy, w // fx
    out = np.zeros((h2, w2), np.int32)
    off = 0
    step = (chunk // fy) * fy
    for s in range(0, h2 * fy, step):
        e = min(h2 * fy, s + step)
        n, lab = cv2.connectedComponents(thick[s:e, :w2 * fx].astype(np.uint8), connectivity=8)
        lab = lab.astype(np.int32)
        lab[lab > 0] += off
        off += n
        out[s // fy:e // fy] = lab.reshape((e - s) // fy, fy, w2, fx).max(axis=(1, 3))
    return out


def bridge_rows(ink: np.ndarray, rows_mask: np.ndarray, max_band: int = 9) -> np.ndarray:
    """Remove the ink of full-width line rows (``rows_mask``) and reconnect strokes across each
    removed band: a pixel in the band is set if there is ink directly above and below the band
    (within 2 px). A solid curve crossing grid lines stays one component; dots/dashes of a
    dotted curve are not joined (their gaps are not at grid lines)."""
    out = ink & ~rows_mask[:, None]
    idx = np.flatnonzero(rows_mask)
    if not len(idx):
        return out
    bands = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    h = ink.shape[0]
    for b in bands:
        a, e = b[0], b[-1] + 1
        if e - a > max_band:
            continue
        up = out[max(0, a - 2):a].any(axis=0) if a > 0 else np.zeros(ink.shape[1], bool)
        dn = out[e:min(h, e + 2)].any(axis=0) if e < h else np.zeros(ink.shape[1], bool)
        out[a:e] |= (up & dn)[None, :]
    return out


def remove_small_components(mask: np.ndarray, min_area: int, chunk: int = 20000) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` px (dots/dashes of dotted curves,
    specks). Processed in overlapping row chunks to bound memory."""
    import cv2
    out = np.zeros_like(mask)
    h = mask.shape[0]
    ov = 200
    for s in range(0, h, chunk):
        a, b = max(0, s - ov), min(h, s + chunk + ov)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask[a:b].astype(np.uint8), 8)
        keep = st[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        part = keep[lab]
        out[s:min(h, s + chunk)] = part[s - a:s - a + min(chunk, h - s)]
    return out


def _apply_anchors(cost: np.ndarray, anchors, tol: int) -> np.ndarray:
    if not anchors:
        return cost
    c = cost.copy()
    w = c.shape[1]
    for r, x in anchors:
        r, x = int(round(r)), int(round(x))
        if 0 <= r < c.shape[0]:
            band = np.full(w, 1e6)
            band[max(0, x - tol):min(w, x + tol + 1)] = 0
            c[r] = c[r] + band
    return c


def trace_dp_python(cost: np.ndarray, max_step: int = 3, smooth: float = 0.1, seg_weight: float = 0.0,
                    labels: np.ndarray | None = None, switch_cost: float = 0.0) -> np.ndarray:
    """Reference NumPy DP (slow; for tests). Same objective as the compiled backends.

    Curve identity: with ``labels`` (stroke component id per cell, 0 = no ink), entering a cell
    whose id differs from the previous cell's id costs ``switch_cost`` - following one
    continuous stroke is free, hopping between dashes/dots of another curve is not."""
    h, w = cost.shape
    lab = labels if labels is not None else np.zeros((h, w), np.int32)
    both = np.minimum(cost, np.r_[cost[:1], cost[:-1]])  # span on row y or y-1 (diagonal strokes)
    pref = np.c_[np.zeros(h), np.cumsum(both, axis=1)]
    acc = cost[0].astype(np.float64).copy()
    back = np.zeros((h, w), dtype=np.int32)
    xs = np.arange(w)
    for y in range(1, h):
        best = np.full(w, np.inf); arg = np.zeros(w, dtype=np.int32)
        for dx in range(-max_step, max_step + 1):
            px = xs - dx
            ok = (px >= 0) & (px < w)
            c = np.full(w, np.inf)
            lo = np.minimum(px[ok], xs[ok]); hi = np.maximum(px[ok], xs[ok])
            seg = (pref[y, hi + 1] - pref[y, lo]) / (hi - lo + 1)
            c[ok] = acc[px[ok]] + smooth * abs(dx) + seg_weight * seg * (abs(dx) > 0)
            if switch_cost:
                lx = lab[y, xs[ok]]
                c[ok] += switch_cost * ((lx > 0) & (lx != lab[y - 1, px[ok]]))
            m = c < best; best[m] = c[m]; arg[m] = px[m]
        acc = best + cost[y]; back[y] = arg
    path = np.zeros(h, dtype=np.int32); path[-1] = int(np.argmin(acc))
    for y in range(h - 1, 0, -1):
        path[y - 1] = back[y, path[y]]
    return path


if HAVE_NUMBA:
    @numba.njit(cache=True)
    def _dp_numba(cost, max_step, smooth, seg_weight, labels, switch_cost):  # pragma: no cover - compiled
        h, w = cost.shape
        pref = np.zeros(w + 1)
        acc = cost[0].copy()
        nxt = np.empty(w)
        back = np.zeros((h, w), dtype=np.int32)
        for y in range(1, h):
            pref[0] = 0.0
            for x in range(w):
                pref[x + 1] = pref[x] + min(cost[y, x], cost[y - 1, x])
            for x in range(w):
                best = np.inf; arg = x
                lo = max(0, x - max_step); hi = min(w - 1, x + max_step)
                for px in range(lo, hi + 1):
                    d = abs(x - px)
                    v = acc[px] + smooth * d
                    if d > 0:
                        a = min(px, x); b = max(px, x)
                        v += seg_weight * (pref[b + 1] - pref[a]) / (b - a + 1)
                    if switch_cost != 0.0 and labels[y, x] > 0 and labels[y, x] != labels[y - 1, px]:
                        v += switch_cost
                    if v < best:
                        best = v; arg = px
                nxt[x] = best + cost[y, x]; back[y, x] = arg
            for x in range(w):
                acc[x] = nxt[x]
        path = np.zeros(h, dtype=np.int32)
        bx = 0
        for x in range(w):
            if acc[x] < acc[bx]:
                bx = x
        path[h - 1] = bx
        for y in range(h - 1, 0, -1):
            path[y - 1] = back[y, path[y]]
        return path


def trace(cost: np.ndarray, max_step: int = 3, smooth: float = 0.1, seg_weight: float = 0.0,
          anchors: list[tuple[int, int]] | None = None, anchor_tol: int = 2, use: str | None = None,
          labels: np.ndarray | None = None, switch_cost: float = 0.0) -> np.ndarray:
    """Best path (column per row) through ``cost``. ``use`` forces 'rust'|'numba'|'python'.
    ``labels``/``switch_cost``: curve-identity term (see :func:`trace_dp_python`)."""
    c = np.ascontiguousarray(_apply_anchors(np.asarray(cost, np.float64), anchors, anchor_tol))
    lab = np.ascontiguousarray(labels if labels is not None else np.zeros(c.shape, np.int32), dtype=np.int32)
    use = use or backend()
    if use == "rust" and HAVE_RUST and hasattr(_rs, "trace_dp_lab"):
        return np.asarray(_rs.trace_dp_lab(c, lab, int(max_step), float(smooth), float(seg_weight), float(switch_cost)))
    if use in ("rust", "numba") and HAVE_NUMBA:
        return _dp_numba(c, int(max_step), float(smooth), float(seg_weight), lab, float(switch_cost))
    return trace_dp_python(c, max_step, smooth, seg_weight, lab, switch_cost)


def confidence(cost: np.ndarray, path: np.ndarray, window: int = 15) -> np.ndarray:
    """Per-row confidence in [0, 1]: running mean of (1 - cost) along the path (1 = on ink)."""
    v = 1.0 - cost[np.arange(len(path)), path]
    k = np.ones(window) / window
    return np.clip(np.convolve(v, k, mode="same"), 0, 1)


def low_confidence_intervals(conf: np.ndarray, threshold: float = 0.5, min_len: int = 5) -> list[tuple[int, int]]:
    """Row intervals where confidence < threshold (for the review queue)."""
    m = conf < threshold
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def trace_track(panel: np.ndarray, row0: int = 0, fy: int | None = None, fx: int = 2, max_step: int | None = None,
                smooth: float = 0.02, seg_weight: float = 3.0, min_width: int | None = None, min_area: int = 400,
                straight_len: int = 121, anchors_px: list[tuple[float, float]] | None = None,
                exclude_cols: list[tuple[int, int]] | None = None, offscale_cost: float | None = 0.8,
                grid_frac: float = 0.5, grid_pad: int = 2, px_per_ft: float | None = None,
                switch_cost: float = 0.5, edge_frac: float = 0.02, use: str | None = None,
                cost_mode: str = "conventional", ml_model: Any = None, ml_weight: float = 0.15,
                caliper_weight: float = 0.55, prob_map: np.ndarray | None = None, mask_other: bool = False) -> dict:
    """Trace the dominant continuous thick curve in a rectified track panel.

    Supports ``cost_mode="conventional" | "ml" | "blend"``. When ``cost_mode`` is ``"ml"`` or
    ``"blend"``, evaluates ``CurveUNet`` (via ``ml_model`` or ``models/kansas_gr_unet.pt``)
    to disfavor competing curves (such as Caliper/SP) before running the DP tracer.
    ``mask_other``: drop text / stamps / handwriting (separation.other_mask) before tracing.
    """
    from pylogdigitizer import track_grid_detection as tgd
    if mask_other:
        from pylogdigitizer import separation
        panel = separation.curve_layer(panel)
    if fy is None:
        fy = int(np.clip(round(px_per_ft * 0.2), 1, 4)) if px_per_ft else 3
    if max_step is None:
        max_step = max(5, panel.shape[1] // (2 * fx))
    if min_width is None:
        min_width = 3   # opening width; 0 = thin-curve mode (line removal), see build_cost
    vl = tgd.vertical_lines(panel[: min(len(panel), 20000)], grid_frac)
    gcols = [(l["x0"] - grid_pad, l["x1"] + grid_pad) for l in vl] + list(exclude_cols or [])
    cost_conv, lab = build_cost(panel, gcols, min_width=min_width, fy=fy, fx=fx, min_area=min_area,
                                straight_len=straight_len, return_labels=True)
    cost = cost_conv
    v61, rel_x = None, None
    if prob_map is None:
        cost_mode = ml_fallback(cost_mode, ml_model)
    if cost_mode != "conventional":
        import cv2
        from pylogdigitizer import ml_models
        H_dp, W_dp = cost_conv.shape
        if fy < 3 and H_dp >= 24:
            k_ml = max(1, 3 // fy)
            h3 = H_dp // k_ml
            cost3 = cost_conv[: h3 * k_ml].reshape(h3, k_ml, W_dp).min(axis=1)
            feat3 = ml_models.extract_track_features(panel, cost3, fy=fy * k_ml, fx=fx)
            if prob_map is None:
                resolved_model = _resolve_default_ml_model(ml_model)
                prob3 = ml_models.predict_track_probabilities(
                    panel, cost3, model=resolved_model, fy=fy * k_ml, fx=fx, feat=feat3
                )
                prob_map = cv2.resize(prob3, (W_dp, H_dp), interpolation=cv2.INTER_LINEAR)
            v61 = cv2.resize(feat3[2], (W_dp, H_dp), interpolation=cv2.INTER_LINEAR)
            rel_x = np.broadcast_to(np.linspace(0.0, 1.0, W_dp, dtype=np.float32)[None, :], (H_dp, W_dp))
        else:
            feat = ml_models.extract_track_features(panel, cost_conv, fy=fy, fx=fx)
            if prob_map is None:
                resolved_model = _resolve_default_ml_model(ml_model)
                prob_map = ml_models.predict_track_probabilities(
                    panel, cost_conv, model=resolved_model, fy=fy, fx=fx, feat=feat
                )
            v61, rel_x = feat[2], feat[6]
        cost = build_ml_cost(
            cost_conv,
            prob_map,
            v61=v61,
            rel_x=rel_x,
            mode=cost_mode,
            ml_weight=ml_weight,
            caliper_weight=caliper_weight,
        )
    pad = 1 if offscale_cost is not None else 0
    if pad:  # virtual off-scale columns: the path parks there when the curve leaves the track
        cost = np.pad(cost, ((0, 0), (1, 1)), constant_values=offscale_cost)
        lab = np.pad(lab, ((0, 0), (1, 1)))
    anc = [((r - row0) / fy, x / fx + pad) for r, x in anchors_px] if anchors_px else None
    path = trace(cost, max_step=max_step, smooth=smooth, seg_weight=seg_weight, anchors=anc,
                 labels=lab, switch_cost=switch_cost, use=use)
    T = finish_trace(cost, path, row0, fy, fx, pad, edge_frac)
    if prob_map is not None:
        p_along = prob_map[np.arange(len(T["path"])), T["path"]]
        k = np.ones(15) / 15.0
        T["ml_conf"] = np.clip(np.convolve(p_along, k, mode="same"), 0.0, 1.0)
        T["prob_map"] = prob_map
    T.update(
        labels=lab,
        cost_conv=cost_conv,
        cost_mode=cost_mode,
        prob_map=prob_map,
        ml_prob=prob_map,
        _v61=v61,
        _rel_x=rel_x,
        params=dict(
            max_step=max_step,
            smooth=smooth,
            seg_weight=seg_weight,
            switch_cost=switch_cost,
            use=use,
            edge_frac=edge_frac,
            cost_mode=cost_mode,
            ml_weight=ml_weight,
            caliper_weight=caliper_weight,
        ),
    )
    return T


def finish_trace(cost: np.ndarray, path_padded: np.ndarray, row0: int, fy: int, fx: int, pad: int,
                 edge_frac: float = 0.02) -> dict:
    """DP path in padded cost columns -> trace dict (rows, x, conf, off_left/right, ...).
    Shared by :func:`trace_track` and the assisted re-trace (``assisted_tracing``)."""
    path_padded = np.asarray(path_padded, np.int32)
    conf = confidence(cost, path_padded)
    wcols = cost.shape[1] - 2 * pad
    e = max(1, int(round(edge_frac * wcols)))
    off_left = (path_padded < pad + e)
    off_right = (path_padded >= cost.shape[1] - pad - e)
    path = np.clip(path_padded - pad, 0, wcols - 1)
    rows = row0 + np.arange(len(path)) * fy + fy / 2
    x = path * fx + fx / 2
    conf[off_left | off_right] = np.minimum(conf[off_left | off_right], 0.5)
    return dict(rows=rows, x=x, conf=conf, cost=cost, path=path, fy=fy, fx=fx,
                off_left=off_left, off_right=off_right, path_padded=path_padded, pad=pad, row0=row0)
