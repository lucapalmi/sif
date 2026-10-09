# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""What Field, Grid, ChainMesh and Catalogue raise when they refuse: the
library's own reason as the message -- which part is wrong, and the values --
and a class that says what kind of failure it was. Each case is one a caller
can actually reach, and most used to share a message with several others."""

import numpy as np
import pysif
import pytest

R = pysif.real
BOX = 100.0


def field(x, y, z):
    return pysif.field_from_numpy(np.asarray(x, R), np.asarray(y, R),
                                  np.asarray(z, R))


@pytest.fixture
def inside():
    return field([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])


@pytest.fixture
def outside():
    # Particle 1 is past the box on x; particle 2 is NaN on z.
    return field([1.0, 150.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, np.nan])


# --- Field ---

def test_empty_arrays():
    e = np.array([], dtype=R)
    with pytest.raises(ValueError, match=r"field: 0 particles \(none were given"):
        pysif.field_from_numpy(e, e, e)


def test_wrap_needs_a_positive_box(inside):
    with pytest.raises(ValueError, match=r"box_length: 0 \(expected > 0\)"):
        inside.wrap(0.0)


def test_wrap_reports_non_finite_coordinates():
    f = field([np.nan, 150.0, 1.0], [1.0, 1.0, np.inf], [1.0, 1.0, 1.0])
    with pytest.warns(RuntimeWarning, match=r"field: 2 coordinates not finite"):
        counts = f.wrap(BOX)
    assert counts == {"boundary": 0, "wrapped": 1, "nonfinite": 2}


def test_wrap_of_a_clean_field_is_silent(inside):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert inside.wrap(BOX)["nonfinite"] == 0


@pytest.mark.parametrize("action", ["refresh_bounds", "sort_morton"])
def test_non_finite_positions_are_named(action):
    f = field([1.0, np.nan], [1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match=r"1 of 2 positions not finite "
                                         r"\(first: particle 1 at \(nan, 2, 2\)\)"):
        getattr(f, action)()


# --- Grid ---

@pytest.mark.parametrize("n_cells, box, message", [
    (0, BOX, r"n_cells: 0 \(expected > 0\)"),
    (8, -1.0, r"box_length: -1 \(expected > 0\)"),
])
def test_grid_geometry(n_cells, box, message):
    # ValueError, not the MemoryError a bad geometry used to raise.
    with pytest.raises(ValueError, match=message):
        pysif.Grid(n_cells=n_cells, box_length=box)


def test_deposit_names_the_first_particle_outside(outside):
    with pytest.raises(ValueError) as e:
        pysif.Grid(8, BOX).assign_cic(outside)
    msg = str(e.value)
    assert "2 of 3 particles outside the box [0, 100)" in msg
    assert "first: particle 1 at (150, 2, 2)" in msg
    assert "Field.wrap(box_length)" in msg  # the Python remedy


def test_rounding_at_the_faces_needs_no_wrap():
    # A coordinate stored as exactly box_length, or a hair below 0, is
    # rounding: deposited and binned as its periodic fold, no wrap() needed.
    up = np.nextafter(R(BOX), R(np.inf))
    down = np.nextafter(R(0.0), R(-np.inf))
    f = field([BOX, up, down], [1.0, BOX, 2.0], [3.0, 4.0, 5.0])
    pysif.Grid(8, BOX).assign_cic(f)
    pysif.ChainMesh(4, BOX, f)
    assert f.x[0] == R(BOX)  # the field itself is left as it was


def test_deposit_of_a_consumed_field(inside):
    pysif.ChainMesh(2, 10.0, inside, consume_field=True)
    with pytest.raises(ValueError, match="consuming chain mesh build"):
        pysif.Grid(8, 10.0).assign_cic(inside)


@pytest.mark.parametrize("prepare, message", [
    (lambda g, f: None, r"mean density 0 .*nothing has been deposited"),
    (lambda g, f: (g.assign_cic(f), g.to_density_contrast()),
     r"already a density contrast"),
])
def test_density_contrast_says_which_refusal(inside, prepare, message):
    # These two used to be told apart by a check in the binding; the library
    # now says which it was itself.
    g = pysif.Grid(8, 10.0)
    prepare(g, inside)
    with pytest.raises(ValueError, match=message):
        g.to_density_contrast()


# --- ChainMesh ---

def test_mesh_geometry(inside):
    with pytest.raises(ValueError, match=r"n_cells: 0 \(expected > 0\)"):
        pysif.ChainMesh(0, BOX, inside)


def test_mesh_names_the_first_particle_outside(outside):
    with pytest.raises(ValueError, match=r"first: particle 1 at \(150, 2, 2\)"):
        pysif.ChainMesh(4, BOX, outside)


def test_mesh_refusal_leaves_a_consumed_field_alone(outside):
    with pytest.raises(ValueError, match="outside the box"):
        pysif.ChainMesh(4, BOX, outside, consume_field=True)
    assert outside.n_particles == 3


# --- Catalogue ---

@pytest.fixture
def catalogue():
    return pysif.catalogue_from_numpy(*(np.array([v], R) for v in (1, 2, 3, 4)))


@pytest.mark.parametrize("key, message", [
    ("my-key", r"key has '-' at character 3"),
    ("1abc", r"key starts with a digit"),
    ("a" * 70, r"key is 70 characters long \(at most 64\)"),
    ("TTYPE3", r"key is reserved"),
])
def test_metadata_key_says_what_is_wrong(catalogue, key, message):
    with pytest.raises(ValueError, match=message):
        catalogue.set_metadata(key, 1)


@pytest.mark.parametrize("value, message", [
    (float("nan"), r"nan is not finite"),
    ('say "hi"', r"contains a double quote at character 5"),
])
def test_metadata_value_says_what_is_wrong(catalogue, value, message):
    with pytest.raises(ValueError, match=message):
        catalogue.set_metadata("ok", value)


def test_to_sky_names_the_first_centre():
    c = pysif.catalogue_from_numpy(*(np.array([v], R)
                                     for v in (1.0, 2.0, np.inf, 3.0)))
    with pytest.raises(ValueError, match=r"first: void 0 at \(1, 2, inf\)"):
        c.to_sky(omega_m=0.3)
