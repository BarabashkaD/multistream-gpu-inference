# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Geometry for the display overlay: box outlines, track labels and a bitmap font atlas with per-character quads."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

_FONT, _SCALE, _THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
_CELL_WIDTH, _CELL_HEIGHT, _BASELINE = 22, 28, 20
_FIRST, _COUNT = 32, 95
_LABEL_CACHE = 2048


def _colours_for(classes: np.ndarray) -> np.ndarray:
    """Return one RGB colour per class id, stable across frames."""
    # Same palette as annotate() in the current demo, converted from BGR to RGB.
    classes = classes.astype(np.int64)
    red = (97 * classes + 40) % 255
    green = (17 * classes + 150) % 255
    blue = (37 * classes + 80) % 255
    return np.stack((red, green, blue), axis=1).astype(np.uint8)


def box_geometry(tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn track rows into line-segment vertices (eight per box, four edges) and a colour per vertex.

    Returns empty arrays when there are no tracks. Colours follow the track's class.
    """
    if len(tracks) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 3), np.uint8)
    x1, y1, x2, y2 = tracks[:, 0], tracks[:, 1], tracks[:, 2], tracks[:, 3]
    edges = np.stack((x1, y1, x2, y1, x2, y1, x2, y2, x2, y2, x1, y2, x1, y2, x1, y1), axis=1)
    return edges.reshape(-1, 2).astype(np.float32), np.repeat(_colours_for(tracks[:, 6]), 8, axis=0)


def track_label(row: np.ndarray, names: dict[int, str]) -> str:
    """Format a track row as its class name (the number if unnamed), score and track id."""
    class_id = int(row[6])
    return f"{names.get(class_id, str(class_id))} {row[5]:.2f} id={int(row[4])}"


class FontAtlas:
    """Every printable ASCII glyph, rendered once, for one textured quad per character."""

    def __init__(self):
        """Render each glyph into its own cell of one image and record its advance width."""
        self.width = _CELL_WIDTH * _COUNT
        self.height = _CELL_HEIGHT
        self.pixels = np.zeros((self.height, self.width), np.uint8)
        self.advance = {}
        for index in range(_COUNT):
            character = chr(_FIRST + index)
            (width, _), _ = cv2.getTextSize(character, _FONT, _SCALE, _THICKNESS)
            self.advance[character] = max(width, 8)
            cv2.putText(self.pixels, character, (index * _CELL_WIDTH + 1, _BASELINE), _FONT, _SCALE, 255,
                        _THICKNESS, cv2.LINE_AA)
        self._quads = lru_cache(maxsize=_LABEL_CACHE)(self._label_quads)

    def _label_quads(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the quad vertices and texture coordinates for a string, laid out from the origin.

        Characters outside the atlas are skipped. The result is cached per string by the constructor.
        """
        vertices, uvs = [], []
        unit = 1.0 / _COUNT
        x = 0.0
        for character in text:
            code = ord(character) - _FIRST
            if not 0 <= code < _COUNT:
                continue
            left, right = code * unit, (code + 1) * unit
            vertices += [x, -_BASELINE, x + _CELL_WIDTH, -_BASELINE,
                         x + _CELL_WIDTH, _CELL_HEIGHT - _BASELINE, x, _CELL_HEIGHT - _BASELINE]
            uvs += [left, 0.0, right, 0.0, right, 1.0, left, 1.0]
            x += self.advance[character]
        return np.array(vertices, np.float32).reshape(-1, 2), np.array(uvs, np.float32).reshape(-1, 2)

    def text_geometry(self, labels: list[tuple[str, float, float, tuple[int, int, int]]]):
        """Combine labels of (text, x, y, colour) into one set of quad vertices, texture coordinates and colours.

        Each text is moved to its position; the layout of each distinct string is cached. Returns empty arrays for
        no labels.
        """
        if not labels:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros((0, 3), np.uint8)
        vertices, uvs, colours = [], [], []
        for text, x, y, colour in labels:
            quad_vertices, quad_uvs = self._quads(text)
            vertices.append(quad_vertices + np.array((x, y), np.float32))
            uvs.append(quad_uvs)
            colours.append(np.full((len(quad_vertices), 3), colour, np.uint8))
        return np.concatenate(vertices), np.concatenate(uvs), np.concatenate(colours)
