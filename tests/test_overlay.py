# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.overlay: box geometry, track labels and the bitmap font atlas used by the display."""
import numpy as np
import pytest

from pipeline.overlay import FontAtlas, box_geometry, track_label


def track(x1=10.0, y1=20.0, x2=30.0, y2=40.0, track_id=5.0, score=0.875, cls=2.0):
    """Build one track row of eight float32 values: box corners, id, score, class and a zero pad."""
    return np.array([x1, y1, x2, y2, track_id, score, cls, 0.0], np.float32)


def test_box_geometry_has_four_edges_per_box():
    """Two boxes give 16 line vertices and 16 colours of the right dtypes, with edges starting at the corners."""
    vertices, colours = box_geometry(np.stack([track(), track(x1=50, x2=60)]))
    assert vertices.shape == (16, 2)
    assert colours.shape == (16, 3)
    assert vertices.dtype == np.float32
    assert colours.dtype == np.uint8
    assert vertices[0].tolist() == [10.0, 20.0]
    assert vertices[1].tolist() == [30.0, 20.0]


def test_box_geometry_empty():
    """No tracks give empty vertex and colour arrays that keep their column counts."""
    vertices, colours = box_geometry(np.zeros((0, 8), np.float32))
    assert vertices.shape == (0, 2)
    assert colours.shape == (0, 3)


def test_box_colour_matches_the_current_palette():
    """A box's colour comes from its class index through the fixed palette formula, in red, green, blue order."""
    _, colours = box_geometry(np.stack([track(cls=2.0)]))
    blue, green, red = (37 * 2 + 80) % 255, (17 * 2 + 150) % 255, (97 * 2 + 40) % 255
    assert colours[0].tolist() == [red, green, blue]


def test_label_format():
    """A label is the class name, the score to two decimals and the track id."""
    assert track_label(track(), {2: "car"}) == "car 0.88 id=5"


def test_atlas_contains_glyphs():
    """The font atlas is a non-empty pixel image of its declared size with at least one fully lit pixel."""
    atlas = FontAtlas()
    assert atlas.pixels.shape == (atlas.height, atlas.width)
    assert atlas.pixels.max() == 255


def test_text_geometry_has_one_quad_per_printable_character():
    """Each printable character becomes one quad of six vertices, coloured per label, with UVs inside the atlas."""
    atlas = FontAtlas()
    vertices, uvs, colours = atlas.text_geometry([("ab", 10.0, 50.0, (1, 2, 3)), ("c", 0.0, 20.0, (4, 5, 6))])
    assert vertices.shape == (12, 2)
    assert uvs.shape == (12, 2)
    assert colours.shape == (12, 3)
    assert colours[0].tolist() == [1, 2, 3]
    assert colours[-1].tolist() == [4, 5, 6]
    assert 0.0 <= uvs.min() and uvs.max() <= 1.0


def test_glyphs_are_upright():
    """A glyph quad's first vertex is its top on screen and samples the top row of the atlas."""
    vertices, uvs, _ = FontAtlas().text_geometry([("a", 0.0, 100.0, (1, 2, 3))])
    assert vertices[0, 1] < vertices[2, 1]  # the first vertex is the top of the quad on screen
    assert uvs[0, 1] == 0.0  # and it samples the top row of the atlas
    assert uvs[2, 1] == 1.0


def test_text_geometry_empty():
    """No labels give empty vertex, UV and colour arrays."""
    vertices, uvs, colours = FontAtlas().text_geometry([])
    assert len(vertices) == len(uvs) == len(colours) == 0

def test_text_scale_grows_glyphs_from_the_label_origin():
    """A larger scale stretches the text away from the label's origin and leaves its left edge in place."""
    atlas = FontAtlas()
    small, _, _ = atlas.text_geometry([("car", 100.0, 50.0, (255, 0, 0))])
    large, _, _ = atlas.text_geometry([("car", 100.0, 50.0, (255, 0, 0))], scale=4.0)
    assert (large[:, 0].max() - 100.0) == pytest.approx((small[:, 0].max() - 100.0) * 4.0)
    assert large[:, 0].min() == pytest.approx(small[:, 0].min())
