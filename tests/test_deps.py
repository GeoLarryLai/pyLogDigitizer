# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Dependency trim: in-house Sauvola matches the formula; dropped packages are not imported by the package."""
import pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sauvola_matches_bruteforce():
    from pylogdigitizer.scan_cleanup import threshold_sauvola, binarize
    rng = np.random.default_rng(0)
    img = (rng.random((40, 50)) * 255).astype(np.uint8); img[10:14] = 5
    w = 7; T = threshold_sauvola(img, w)
    pad = np.pad(img.astype(float), w // 2, mode="reflect"); r = 0.5 * (img.max() - img.min())
    for y, x in [(0, 0), (12, 20), (39, 49), (25, 3)]:
        win = pad[y:y + w, x:x + w]
        assert abs(T[y, x] - win.mean() * (1 + 0.2 * (win.std() / r - 1))) < 1e-6
    assert binarize(img, "sauvola")[11].mean() > 0.9


def test_dropped_dependencies_not_imported():
    deps = re.search(r"dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.S).group(1)
    for name in ("scipy", "scikit-image", "tifffile"):
        assert f'"{name}"' not in deps
    pat = re.compile(r"^\s*(import|from)\s+(scipy|skimage|tifffile)\b", re.M)
    hits = [str(p) for p in (ROOT / "src").rglob("*.py") if pat.search(p.read_text())]
    assert not hits, hits


def test_runtime_without_numba_torch_matplotlib():
    """Trim 2: numba and torch are not dependencies, matplotlib loads lazily. The package imports and
    traces with all three blocked, and importing it does not load them."""
    import subprocess, sys
    code = ("import sys\n"
            "for m in ('numba', 'torch', 'matplotlib', 'scipy', 'skimage', 'tifffile'): sys.modules[m] = None\n"
            "import numpy as np, warnings\n"
            "warnings.simplefilter('ignore')\n"
            "import pylogdigitizer.workflows, pylogdigitizer.app.routes\n"
            "from pylogdigitizer import curve_tracing as ct, grid_rectify as gr\n"
            "ink, _, _ = gr.synthetic_grid(H=600, W=200, nv=5, nh=6, margin=10, amp=0, shear=0, seed=0)\n"
            "v = np.arange(0, 600, 0.25); gr._stamp(ink, 100 + 50 * np.sin(v / 60), v, 3)\n"
            "T = ct.trace_track(ink, 0, px_per_ft=10)\n"
            "assert np.isfinite(T['x']).all()\n"
            "print(ct.backend())\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    probe = subprocess.run([sys.executable, "-c", "import sys, pylogdigitizer.workflows, pylogdigitizer.app.routes;"
                            "print(sorted(m for m in ('numba','torch','matplotlib','scipy') if m in sys.modules))"],
                           capture_output=True, text=True, timeout=120)
    assert probe.stdout.strip() == "[]", probe.stdout + probe.stderr[-1000:]
    deps = re.search(r"dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.S).group(1)
    assert '"numba"' not in deps and '"torch"' not in deps
