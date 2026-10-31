# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Use the installed package (pip install . / maturin develop) when it has the compiled Rust module;
fall back to ./src only when pylogdigitizer is not importable at all. A src/ copy without the Rust
module would otherwise silently run the slow pure-Python tracer."""
import importlib.util, pathlib, sys, warnings

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if importlib.util.find_spec("pylogdigitizer") is None:
    sys.path.insert(0, str(SRC))
try:
    from pylogdigitizer import curve_tracing_rs  # noqa: F401
except ImportError:
    warnings.warn("pyLogDigitizer's Rust module is not built: tests use the slow Python fallback. "
                  "Build it with `maturin develop --release` (or `pip install .`).", stacklevel=1)
