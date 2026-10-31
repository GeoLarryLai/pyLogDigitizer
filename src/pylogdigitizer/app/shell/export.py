# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""export: the app's only file writer. Nothing is written until an export button is pressed."""
from __future__ import annotations
from pathlib import Path


class ExportGuard:
    """Keeps the export folder and a log of what was written. ``write(name, writer)`` creates the
    folder on first use and calls ``writer(path)``; it refuses names that escape the folder."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser().resolve(); self.written: list[str] = []

    def set_folder(self, folder: str | Path) -> Path:
        self.folder = Path(folder).expanduser().resolve(); return self.folder

    def path(self, name: str) -> Path:
        p = (self.folder / Path(name).name).resolve()
        if p.parent != self.folder:
            raise ValueError(f"bad export name {name!r}")
        return p

    def write(self, name: str, writer) -> Path:
        p = self.path(name); self.folder.mkdir(parents=True, exist_ok=True)
        out = writer(p); out = Path(out) if out else p
        self.written.append(str(out)); return out

    def state(self) -> dict:
        return dict(folder=str(self.folder), exists=self.folder.is_dir(), written=self.written[-20:])
