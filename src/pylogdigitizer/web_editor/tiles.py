# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""tiles: Deep Zoom (DZI) pyramid of a working sheet image, built lazily in memory."""
from __future__ import annotations
import io, math
import numpy as np


class TilePyramid:
    """DZI tiles from a bool ink image (True = ink) or uint8 gray image. Levels are built on demand
    by 2x area-averaging the next finer level (thin ink stays visible as gray)."""

    def __init__(self, image: np.ndarray, tile_size: int = 512):
        self.tile = tile_size
        self.h, self.w = image.shape[:2]
        self.max_level = int(math.ceil(math.log2(max(self.h, self.w, 1))))
        self._full = image
        self._levels: dict[int, np.ndarray] = {}
        self._cache: dict[tuple, bytes] = {}

    def dzi_xml(self) -> str:
        return ('<?xml version="1.0" encoding="UTF-8"?>'
                f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="png" Overlap="0" TileSize="{self.tile}">'
                f'<Size Width="{self.w}" Height="{self.h}"/></Image>')

    def level(self, lv: int) -> np.ndarray:
        if lv in self._levels:
            return self._levels[lv]
        if lv >= self.max_level:
            im = self._full
            img = np.where(im, 0, 255).astype(np.uint8) if im.dtype == bool else im
        else:
            import cv2
            up = self.level(lv + 1)
            img = cv2.resize(up, (max(1, -(-up.shape[1] // 2)), max(1, -(-up.shape[0] // 2))), interpolation=cv2.INTER_AREA)
        if lv < self.max_level or self._full.dtype != bool:
            self._levels[lv] = img
        else:
            self._levels[lv] = img   # full-res uint8 copy kept (fast tile cuts)
        return img

    def tile_png(self, lv: int, col: int, row: int) -> bytes:
        key = (lv, col, row)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image
        img = self.level(min(lv, self.max_level))
        t = img[row * self.tile:(row + 1) * self.tile, col * self.tile:(col + 1) * self.tile]
        if t.size == 0:
            raise KeyError(key)
        b = io.BytesIO(); Image.fromarray(t).save(b, "PNG", compress_level=1); data = b.getvalue()
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = data
        return data
