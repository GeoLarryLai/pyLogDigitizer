# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""web_editor: core HTTP API + vendored OpenSeadragon viewer used by the app (``pylogdigitizer-app``).

    pylogdigitizer-app [PROJECT_DIR]     (``pylogdigitizer editor`` is an alias)

Binds to 127.0.0.1 only; no data leaves the machine. All digitization logic is in
``pylogdigitizer.assisted_tracing`` - this module only exposes it over HTTP.
"""
from __future__ import annotations

__all__ = ["create_app", "serve", "HOST"]
HOST = "127.0.0.1"


def create_app(project_dir: str = "."):
    from pylogdigitizer.web_editor.app import build_app
    return build_app(project_dir)


def serve(project_dir: str = ".", port: int = 8765, open_browser: bool = False) -> None:
    """Run the editor on http://127.0.0.1:<port> (local only)."""
    from pylogdigitizer.app import main
    main([str(project_dir), "--port", str(port)] + ([] if open_browser else ["--no-browser"]))
