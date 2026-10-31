# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""undo: bounded stack of (label, snapshot) pairs."""
from __future__ import annotations


class UndoStack:
    def __init__(self, limit: int = 30):
        self.items: list[tuple[str, object]] = []; self.limit = limit

    def push(self, label: str, snap) -> None:
        self.items.append((label, snap)); del self.items[:-self.limit]

    def pop(self):
        return self.items.pop() if self.items else None

    def labels(self) -> list[str]:
        return [l for l, _ in self.items]

    def clear(self) -> None:
        self.items.clear()
