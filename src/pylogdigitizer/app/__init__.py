# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""app: pyLogDigitizer desktop app - a local, step-by-step browser UI following notebook 01.

    pip install pylogdigitizer
    pylogdigitizer-app [PROJECT_DIR]          # opens http://127.0.0.1:<free port>/

Local only (127.0.0.1), works offline (vendored JS), writes nothing until an Export button is
pressed. Layout: ``shell/`` is the project-independent frame (step rail, jobs + progress, undo,
export guard, styles); ``routes.py`` + ``static/`` are the pyLogDigitizer steps, on top of the
existing editor API (``web_editor``), which is mounted at /core (API + static only; its own page is not served) and shares its sessions.
"""
from __future__ import annotations
from pathlib import Path
import argparse

__all__ = ["create_app", "main"]
STATIC = Path(__file__).parent / "static"


LOCAL_HOSTS = ("127.0.0.1", "localhost")


def create_app(project_dir: str = ".", allowed_hosts=LOCAL_HOSTS):
    """The app for ``project_dir``. Requests must name a local host (``allowed_hosts``), so a web page
    cannot reach the app through DNS rebinding; writes must also come from the app's own page."""
    from fastapi import FastAPI
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pylogdigitizer.web_editor.app import build_app
    from pylogdigitizer.app.shell import JobRunner, mount_shell
    from pylogdigitizer.app import routes
    root = Path(project_dir).expanduser().resolve()
    app = FastAPI(title="pyLogDigitizer", docs_url="/app/api/docs", redoc_url=None)
    app.state.root = root
    app.state.editor = build_app(str(root))
    app.state.runner = JobRunner(error_types=routes.EXPECTED)
    mount_shell(app, app.state.runner)
    routes.install(app, root, app.state.runner)
    app.mount("/core", app.state.editor)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="app_static")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi import Response
        return Response(status_code=204)
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pylogdigitizer-app", description="pyLogDigitizer desktop app (local browser UI)")
    ap.add_argument("project", nargs="?", default=".", help="project folder with example_data/ and dataset_profiles/ (default: .)")
    ap.add_argument("--port", type=int, default=None, help="preferred port (a free one is picked if busy)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args(argv)
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        print("The app needs FastAPI and uvicorn: pip install fastapi uvicorn"); return 1
    from pylogdigitizer.app.shell import serve
    serve(create_app(a.project), "pyLogDigitizer", a.port, not a.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
