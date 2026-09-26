# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Field.x and the other column views: zero-copy, read-only, and guarded.

A view points into the field's own arrays, so whatever would reallocate or
take those arrays has to refuse while one is alive -- otherwise the view is
left reading freed memory. These tests pin down which operations refuse and
which do not.
"""

import gc
import sys

import numpy as np
import pysif
import pytest

N = 1000
BOX = 10.0


@pytest.fixture
def columns():
    rng = np.random.default_rng(0)
    return [rng.random(N) * BOX for _ in range(3)]


@pytest.fixture
def field(columns):
    f = pysif.Field()
    f.from_numpy(*columns, vx=columns[0], vy=columns[1], vz=columns[2],
                 weights=np.full(N, 2.0))
    return f


def test_views_match_and_are_read_only(field, columns):
    for name, want in zip(["x", "y", "z", "vx", "vy", "vz"], columns * 2):
        view = getattr(field, name)
        assert view.shape == (N,)
        assert not view.flags.writeable
        np.testing.assert_allclose(view, want, rtol=1e-6)
    assert np.all(field.weights == 2.0)

    with pytest.raises(ValueError):
        field.x[0] = 1.0


def test_absent_columns_are_none(columns):
    f = pysif.Field()
    assert f.x is None
    f.from_numpy(*columns)
    assert f.vx is None and f.vy is None and f.vz is None and f.weights is None


def test_view_is_zero_copy_and_sees_in_place_changes(field, columns):
    x = field.x
    field.translate([1.0, 0.0, 0.0])
    np.testing.assert_allclose(x, columns[0] + 1.0, rtol=1e-6)


def test_view_outlives_the_field(field):
    x = field.x
    want = x.copy()
    del field
    gc.collect()
    np.testing.assert_array_equal(x, want)


@pytest.mark.parametrize("action", [
    lambda f, c: f.sort_morton(),
    lambda f, c: f.from_numpy(*c),
    lambda f, c: pysif.Octree(f),  # unsorted, so building sorts it
    lambda f, c: pysif.ChainMesh(4, BOX, f, consume_field=True),
], ids=["sort_morton", "from_numpy", "octree", "chain-mesh-consume"])
def test_reallocation_refused_while_viewed(field, columns, action):
    x = field.x
    want = x.copy()
    with pytest.raises(BufferError, match="1 live array view"):
        action(field, columns)
    np.testing.assert_array_equal(x, want)  # the field was left alone

    del x
    gc.collect()
    action(field, columns)  # and goes through once the view is gone


def test_harmless_operations_allowed_while_viewed(field):
    field.sort_morton()
    w = field.weights
    pysif.Octree(field)             # already sorted: nothing reallocated
    pysif.ChainMesh(4, BOX, field)  # copies the field, leaves it be
    field.wrap(BOX)
    assert np.all(w == 2.0)


def test_views_are_released(field):
    refs = sys.getrefcount(field)
    for _ in range(1000):
        field.x
    gc.collect()
    assert sys.getrefcount(field) == refs
    field.sort_morton()  # no export left behind
