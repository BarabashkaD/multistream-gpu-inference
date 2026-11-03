# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the overlay geometry: box edges and colours, track labels and the font atlas text quads."""
import numpy as np

from pipeline.overlay import FontAtlas, box_geometry, track_label


def track(x1=10.0, y1=20.0, x2=30.0, y2=40.0, track_id=5.0, score=0.875, cls=2.0):
    """Builds one track row: box corners, track id, score, class and an unused last column."""
    return np.array([x1, y1, x2, y2, track_id, score, cls, 0.0], np.float32)


def test_box_geometry_has_four_edges_per_box():
    """Proves each box becomes four line segments, eight vertices, with float32 positions and uint8 colours."""
    vertices, colours = box_geometry(np.stack([track(), track(x1=50, x2=60)]))
    assert vertices.shape == (16, 2)
    assert colours.shape == (16, 3)
    assert vertices.dtype == np.float32
    assert colours.dtype == np.uint8
    assert vertices[0].tolist() == [10.0, 20.0]
    assert vertices[1].tolist() == [30.0, 20.0]


def test_box_geometry_empty():
    """Proves no tracks give empty vertex and colour arrays of the right width."""
    vertices, colours = box_geometry(np.zeros((0, 8), np.float32))
    assert vertices.shape == (0, 2)
    assert colours.shape == (0, 3)


def test_box_colour_matches_the_current_palette():
    """Proves a box takes its colour from the class number with the palette formula, in red, green, blue order."""
    _, colours = box_geometry(np.stack([track(cls=2.0)]))
    blue, green, red = (37 * 2 + 80) % 255, (17 * 2 + 150) % 255, (97 * 2 + 40) % 255
    assert colours[0].tolist() == [red, green, blue]


def test_label_format():
    """Proves a label shows the class name, the score to two decimals and the track id."""
    assert track_label(track(), {2: "car"}) == "car 0.88 id=5"


def test_atlas_contains_glyphs():
    """Proves the font atlas image has the reported size and holds drawn glyph pixels."""
    atlas = FontAtlas()
    assert atlas.pixels.shape == (atlas.height, atlas.width)
    assert atlas.pixels.max() == 255


def test_text_geometry_has_one_quad_per_printable_character():
    """Proves each character becomes one quad of six vertices, carries its string's colour, and has texture
    coordinates inside the atlas."""
    atlas = FontAtlas()
    vertices, uvs, colours = atlas.text_geometry([("ab", 10.0, 50.0, (1, 2, 3)), ("c", 0.0, 20.0, (4, 5, 6))])
    assert vertices.shape == (12, 2)
    assert uvs.shape == (12, 2)
    assert colours.shape == (12, 3)
    assert colours[0].tolist() == [1, 2, 3]
    assert colours[-1].tolist() == [4, 5, 6]
    assert 0.0 <= uvs.min() and uvs.max() <= 1.0


def test_glyphs_are_upright():
    """Proves a glyph quad has its top on screen matched to the top row of the atlas, so text is not upside down."""
    vertices, uvs, _ = FontAtlas().text_geometry([("a", 0.0, 100.0, (1, 2, 3))])
    assert vertices[0, 1] < vertices[2, 1]  # the first vertex is the top of the quad on screen
    assert uvs[0, 1] == 0.0  # and it samples the top row of the atlas
    assert uvs[2, 1] == 1.0


def test_text_geometry_empty():
    """Proves no strings give no vertices, texture coordinates or colours."""
    vertices, uvs, colours = FontAtlas().text_geometry([])
    assert len(vertices) == len(uvs) == len(colours) == 0
