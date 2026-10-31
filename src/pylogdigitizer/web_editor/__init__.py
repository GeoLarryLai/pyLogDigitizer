# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""web_editor: local web UI (FastAPI backend + OpenSeadragon viewer, vendored, works offline).

    pylogdigitizer editor [PROJECT_DIR] [--port 8765]     ->  http://127.0.0.1:8765

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
    import uvicorn
    app = create_app(project_dir)
    if open_browser:
        import threading, webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{port}")).start()
    print(f"pyLogDigitizer editor: http://{HOST}:{port}  (local only, Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
