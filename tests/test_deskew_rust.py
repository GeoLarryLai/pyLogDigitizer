# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""The Rust deskew kernel gives exactly the same result as the pure-Python version."""
import numpy as np, pytest
from pylogdigitizer import scan_cleanup as sc


@pytest.mark.skipif(sc._rs is None, reason="Rust extension not built")
def test_rust_deskew_matches_python():
    rng = np.random.default_rng(0)
    h, w = 9000, 600
    ink = np.zeros((h, w), bool)
    for i, y in enumerate(range(50, h, 120)):             # tilted grid lines, drifting skew
        t = np.tan(np.deg2rad(0.3 + 0.2 * np.sin(i / 7)))
        xs = np.arange(w); ys = np.clip(np.round(y + xs * t).astype(int), 0, h - 1); ink[ys, xs] = True
    ink |= rng.random((h, w)) < 0.002
    o_rs, a_rs = sc.deskew_blocks(ink, block=2000)
    rs, nb = sc._rs, sc.HAVE_NUMBA
    try:
        sc._rs, sc.HAVE_NUMBA = None, False
        o_py, a_py = sc.deskew_blocks(ink, block=2000)
    finally:
        sc._rs, sc.HAVE_NUMBA = rs, nb
    assert np.array_equal(a_rs, a_py) and np.array_equal(o_rs, o_py) and o_rs.dtype == bool
