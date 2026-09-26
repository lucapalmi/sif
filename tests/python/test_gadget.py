# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.io.read_gadget, gadget_header and inspect_gadget, on the fixtures in
tests/data/gadget -- see make_fixtures.py there for what they hold.

The formats themselves are tested in C (tests/test_gadget.c). Here: that the
keyword arguments map onto the right C options, that every snapshot reaches
Python with the right values, and that each kind of failure raises the
exception it is documented to.
"""

import numpy as np
import pysif
import pytest
from conftest import GADGET, HAS_HDF5, requires_hdf5

N_TOTAL = {0: 40, 1: 100, 2: 30}

SNAPSHOTS = [
    pytest.param("f1_legacy/snap_005", id="f1-legacy"),
    pytest.param("f1_g4/snap_005", id="f1-g4-snapdir"),
    pytest.param("f1_g4_u32/snap_005", id="f1-g4-u32"),
    pytest.param("f2_swapped/snap_005.1", id="f2-swapped"),
    pytest.param("hdf5_g4/snap_005", id="hdf5-g4", marks=requires_hdf5),
    pytest.param("hdf5_legacy/snap_005", id="hdf5-legacy", marks=requires_hdf5),
]


def path(rel):
    return str(GADGET / rel)


def expected(ptype):
    g = np.arange(N_TOTAL[ptype])
    x = 100.0 * ptype + 0.5 * g + 0.125
    vx = 10.0 * ptype + 0.25 * g
    return x, vx, 1.0 + g / 1024.0


# --- reading ---

@pytest.mark.parametrize("snap", SNAPSHOTS)
@pytest.mark.parametrize("ptype", [0, 1, 2])
def test_values(snap, ptype):
    field, box = pysif.io.read_gadget(
        path(snap), ptype=ptype, velocities="raw", masses=True)
    x, vx, m = expected(ptype)
    dt = field.x.dtype

    assert field.n_particles == N_TOTAL[ptype]
    assert box == 10.0  # 10 000 kpc/h
    np.testing.assert_array_equal(field.x, (x * 1e-3).astype(dt))
    np.testing.assert_array_equal(field.y, ((x + 1000) * 1e-3).astype(dt))
    np.testing.assert_array_equal(field.z, ((x + 2000) * 1e-3).astype(dt))
    np.testing.assert_array_equal(field.vx, vx.astype(dt))
    np.testing.assert_array_equal(field.vy, (-vx).astype(dt))
    np.testing.assert_array_equal(field.vz, (vx + 1).astype(dt))

    if ptype == 1:  # one mass for the whole type, in the table
        assert field.weights is None
    else:
        np.testing.assert_array_equal(field.weights, m.astype(dt))


def test_defaults_skip_velocities_and_masses():
    field, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), ptype=0)
    assert field.has_velocities is False and field.vx is None
    assert field.weights is None


def test_peculiar_velocities():
    # u * sqrt(a), with a = 0.25
    field, _ = pysif.io.read_gadget(
        path("f1_legacy/snap_005"), velocities="peculiar")
    _, vx, _ = expected(1)
    np.testing.assert_array_equal(field.vx, (0.5 * vx).astype(field.vx.dtype))


def test_length_mpc_is_not_converted():
    field, box = pysif.io.read_gadget(path("f1_legacy/snap_005"), length="mpc")
    x, _, _ = expected(1)
    assert box == 10000.0
    np.testing.assert_array_equal(field.x, x.astype(field.x.dtype))


@requires_hdf5
@pytest.mark.parametrize("snap", ["hdf5_g4/snap_005", "hdf5_legacy/snap_005"])
def test_length_auto_from_hdf5(snap):
    field, box = pysif.io.read_gadget(path(snap), length="auto")
    x, _, _ = expected(1)
    assert box == pytest.approx(10.0, rel=1e-12)
    np.testing.assert_allclose(field.x, x * 1e-3, rtol=1e-6)


@pytest.mark.parametrize("fmt", [1, "1"])
def test_explicit_format(fmt):
    field, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), format=fmt)
    assert field.n_particles == 100


# --- subsampling ---

def test_subsample_exact_ordered_reproducible():
    a, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), fraction=0.3, seed=7)
    b, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), fraction=0.3, seed=7)
    c, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), fraction=0.3, seed=8)

    assert a.n_particles == 30
    assert np.all(np.diff(a.x) > 0)  # file order kept
    np.testing.assert_array_equal(a.x, b.x)
    assert not np.array_equal(a.x, c.x)


@pytest.mark.parametrize("snap", SNAPSHOTS)
def test_subsample_same_across_formats(snap):
    ref, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), fraction=0.3, seed=7)
    other, _ = pysif.io.read_gadget(path(snap), fraction=0.3, seed=7)
    np.testing.assert_array_equal(ref.x, other.x)


# --- header ---

def test_header_binary():
    h = pysif.io.gadget_header(path("f2_swapped/snap_005"))
    assert h["format"] == 2
    assert h["swapped"] is True and h["legacy_header"] is True
    assert h["precision"] == 4 and h["n_files"] == 3
    assert h["n_part_total"][:3] == [40, 100, 30]
    assert h["n_part_file"][:3] == [15, 30, 0]
    assert h["mass_table"][:3] == [0.0, 0.5, 0.0]
    assert h["time"] == 0.25 and h["redshift"] == 3.0 and h["box_size"] == 10000.0
    assert h["cosmology"] == {"omega0": 0.3, "omega_lambda": 0.7, "hubble_param": 0.7}
    assert h["unit_length_in_cm"] is None
    assert h["blocks"] == ["HEAD", "POS", "VEL", "ID", "MASS", "U"]


def test_header_format_1_is_unlabelled():
    h = pysif.io.gadget_header(path("f1_g4/snap_005"))
    assert h["format"] == 1 and h["legacy_header"] is False
    assert h["precision"] == 8 and h["cosmology"] is None
    assert h["n_blocks"] == 6 and h["blocks"] is None


@requires_hdf5
def test_header_hdf5():
    h = pysif.io.gadget_header(path("hdf5_g4/snap_005.2.hdf5"))
    assert h["format"] == "hdf5"
    assert h["swapped"] is None and h["legacy_header"] is None
    assert h["unit_length_in_cm"] == 3.085678e21
    assert {"Coordinates", "Velocities", "Masses"} <= set(h["blocks"])


def test_inspect_prints_to_sys_stdout(capsys):
    pysif.io.inspect_gadget(path("f1_g4/snap_005"))
    out = capsys.readouterr().out
    assert "SnapFormat 1, GADGET-4 header" in out
    assert "double" in out and "per particle" in out


# --- failures ---

@pytest.mark.parametrize("kwargs, message", [
    (dict(ptype=6), "ptype"),
    (dict(ptype=-1), "ptype"),
    (dict(velocities="comoving"), "velocities"),
    (dict(velocities=True), "velocities"),
    (dict(length="pc"), "length"),
    (dict(format=3), "format"),
    (dict(format="gadget"), "format"),
    (dict(fraction=0.0), "fraction"),
    (dict(fraction=1.5), "fraction"),
    (dict(fraction=float("nan")), "fraction"),
])
def test_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pysif.io.read_gadget(path("f1_legacy/snap_005"), **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(ptype=4),          # no particles of that type
    dict(fraction=0.001),   # keeps none of 100
    dict(length="auto"),    # a binary file records no unit
])
def test_requests_the_snapshot_cannot_satisfy(kwargs):
    with pytest.raises(ValueError):
        pysif.io.read_gadget(path("f1_legacy/snap_005"), **kwargs)


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        pysif.io.read_gadget(path("f1_legacy/snap_005"), 1)


@pytest.mark.parametrize("rel, kwargs", [
    ("nothing/snap_005", {}),
    ("make_fixtures.py", {}),
    ("f1_legacy/snap_005", dict(format=2)),
])
def test_io_errors(rel, kwargs):
    with pytest.raises(OSError):
        pysif.io.read_gadget(path(rel), **kwargs)


@pytest.mark.skipif(HAS_HDF5, reason="pysif built with HDF5")
def test_hdf5_without_support():
    with pytest.raises(RuntimeError, match="HDF5"):
        pysif.io.read_gadget(path("hdf5_g4/snap_005"))


# --- a dark-matter-only GADGET-4 run ---

NTYPES2 = "f1_g4_ntypes2/snapdir_005/snap_005"


def test_ntypes2_header():
    h = pysif.io.gadget_header(path(NTYPES2))
    assert h["n_types"] == 2
    assert h["n_part_total"] == [0, 100] and h["mass_table"] == [0.0, 0.5]
    assert h["n_blocks"] == 4  # HEAD POS VEL ID: no MASS, no gas


def test_ntypes2_values():
    field, box = pysif.io.read_gadget(path(NTYPES2), velocities="raw", masses=True)
    x, vx, _ = expected(1)
    assert box == 10.0 and field.weights is None
    np.testing.assert_array_equal(field.x, (x * 1e-3).astype(field.x.dtype))
    np.testing.assert_array_equal(field.vx, vx.astype(field.vx.dtype))

    ref, _ = pysif.io.read_gadget(path("f1_legacy/snap_005"), fraction=0.3, seed=7)
    sub, _ = pysif.io.read_gadget(path(NTYPES2), fraction=0.3, seed=7)
    np.testing.assert_array_equal(ref.x, sub.x)


@pytest.mark.parametrize("ptype", [0, 2])  # empty, and beyond NTYPES
def test_ntypes2_missing_types(ptype):
    with pytest.raises(ValueError):
        pysif.io.read_gadget(path(NTYPES2), ptype=ptype)
