# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""shell: a small, project-independent frame for local step-by-step browser apps.

Copy this folder into another project (e.g. pyPetrograph) to reuse it. It knows nothing about
well logs. It provides:

* ``JobRunner`` (jobs.py)     - background threads with ``n / N (xx%)`` progress, polled by the page
* ``ExportGuard`` (export.py) - the only way the app writes files: explicit export presses only
* ``UndoStack`` (undo.py)     - bounded snapshot stack (the app decides what a snapshot holds)
* ``mount_shell`` / ``serve`` (server.py) - /shell routes (jobs, quit, open folder), free port on
  127.0.0.1, browser auto-open, clean Ctrl+C / Quit shutdown
* static/shell.css + static/shell.js - step rail, highlighted main button, help bar, progress
  bar, toasts, ``api()`` helper, ``runJob()`` poller. Vanilla JS/CSS, no build step, no CDN.
"""
from .jobs import JobRunner, Job
from .export import ExportGuard
from .undo import UndoStack
from .server import mount_shell, serve, free_port, HOST

__all__ = ["JobRunner", "Job", "ExportGuard", "UndoStack", "mount_shell", "serve", "free_port", "HOST"]
