# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.model.comoving_distance, Field.units and
Field.convert_sky_coordinates: that the arguments reach the C side as meant,
and that a field still holding sky coordinates is refused, with a message
that says so, by everything that reads positions as lengths. The distances
themselves are tested in C (tests/test_cosmology.c)."""

import numpy as np
import pysif
import pytest

DH = 2997.92458  # Mpc/h


def eds(z):
    return 2 * DH * (1 - 1 / np.sqrt(1 + z))


# --- distances ---

def test_closed_form():
    z = np.array([0.0, 0.5, 1.0, 3.0])
    d = pysif.model.comoving_distance(z, omega_m=1.0, omega_de=0.0)
    np.testing.assert_allclose(d, eds(z), rtol=1e-12)


def test_scalar_in_scalar_out_and_shape_kept():
    assert isinstance(pysif.model.comoving_distance(1.0, 0.3), float)
    assert pysif.model.comoving_distance(np.ones((2, 3)), 0.3).shape == (2, 3)


def test_flat_by_default():
    flat = pysif.model.comoving_distance(1.0, 0.31)
    explicit = pysif.model.comoving_distance(1.0, 0.31, omega_de=0.69)
    open_ = pysif.model.comoving_distance(1.0, 0.31, omega_de=0.5)
    assert flat == explicit and open_ != flat


@pytest.mark.parametrize("kwargs, message", [
    (dict(z=-0.5, omega_m=0.3), "redshift"),
    (dict(z=float("nan"), omega_m=0.3), "redshift"),
    (dict(z=1.0, omega_m=-0.3), "negative"),
    (dict(z=1.0, omega_m=0.3, w0=float("inf")), "finite"),
    (dict(z=3.0, omega_m=0.0, omega_de=3.0), "expansion history"),
])
def test_distance_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pysif.model.comoving_distance(**kwargs)


# --- units and conversion ---

def sky(ra, dec, z, **kwargs):
    return pysif.field_from_numpy(
        ra=np.asarray(ra, dtype=pysif.real), dec=np.asarray(dec, dtype=pysif.real),
        z=np.asarray(z, dtype=pysif.real), **kwargs)


def test_units_default_and_relabel():
    f = pysif.field_from_numpy(*np.ones((3, 4), dtype=pysif.real))
    assert f.units == "cartesian"
    f.units = "sky"
    assert f.units == "sky"
    with pytest.raises(ValueError):
        f.units = "galactic"


def test_names_say_what_the_positions_are():
    ones = np.ones(4, dtype=pysif.real)
    assert pysif.field_from_numpy(ra=ones, dec=ones, z=ones).units == "sky"
    with pytest.raises(ValueError, match="not some of each"):
        pysif.field_from_numpy(x=ones, dec=ones, z=ones)
    with pytest.raises(ValueError, match="ra, dec and z"):
        pysif.field_from_numpy(ra=ones, dec=ones)
    with pytest.raises(ValueError, match="x, y and z"):
        pysif.field_from_numpy(ones, ones)


def test_conversion_follows_pyrecon():
    rng = np.random.default_rng(3)
    ra, dec = rng.uniform(0, 360, 500), rng.uniform(-90, 90, 500)
    z = rng.uniform(0, 2, 500)
    w = rng.uniform(0.5, 1.5, 500)
    f = sky(ra, dec, z, weights=w)
    f.convert_sky_coordinates(omega_m=0.31)

    # pyrecon.utils.sky_to_cartesian, written out
    d = pysif.model.comoving_distance(z.astype(pysif.real), omega_m=0.31)
    r, t = np.radians(ra.astype(pysif.real)), np.radians(dec.astype(pysif.real))
    want = d * np.array([np.cos(t) * np.cos(r), np.cos(t) * np.sin(r), np.sin(t)])

    assert f.units == "cartesian"
    np.testing.assert_allclose(np.array([f.x, f.y, f.z]), want, atol=1e-3)
    np.testing.assert_array_equal(f.weights, w.astype(pysif.real))


def test_conversion_refusals_leave_the_field():
    f = sky([10.0, 20.0], [0.0, 95.0], [0.5, 0.6])
    with pytest.raises(ValueError, match="declination"):
        f.convert_sky_coordinates(0.3)
    assert f.units == "sky" and f.y[1] == np.float32(95.0)

    g = pysif.field_from_numpy(*np.ones((3, 2), dtype=pysif.real))
    with pytest.raises(ValueError, match="already holds Cartesian"):
        g.convert_sky_coordinates(0.3)


# --- refusals of a sky field ---

@pytest.fixture
def sky_field():
    return sky([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("action", [
    lambda f: f.wrap(100.0),
    lambda f: f.translate((1.0, 1.0, 1.0)),
    lambda f: f.sort_morton(),
    lambda f: f.refresh_bounds(),
    lambda f: pysif.Grid(16, 100.0).assign_cic(f),
    lambda f: pysif.ChainMesh(4, 100.0, f),
    lambda f: pysif.Octree(f),
    lambda f: pysif.finders.survey_box(f, np.array([5.0], pysif.real), 32),
], ids=["wrap", "translate", "sort_morton", "refresh_bounds", "assign_cic",
        "ChainMesh", "Octree", "survey_box"])
def test_sky_field_refused(sky_field, action):
    with pytest.raises(ValueError, match="sky coordinates"):
        action(sky_field)
