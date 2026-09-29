# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.io.read_hdf5, on a snapshot written here with h5py in the layout
IllustrisTNG uses. The reader itself is tested in C (tests/test_hdf5.c).
"""

import numpy as np
import pysif
import pytest
from conftest import requires_hdf5

h5py = pytest.importorskip("h5py")
pytestmark = requires_hdf5

N = 500
I = np.arange(N)
POS = np.stack([I + 0.5, 2 * I + 0.5, 3 * I + 0.5], axis=1).astype(np.float32)
MASS = 1 + I / 1024


def snapshot(path, rows=slice(None)):
    with h5py.File(path, "w") as f:
        f["PartType1/Coordinates"] = POS[rows]
        f["PartType1/Masses"] = MASS[rows]
        f["PartType1/ParticleIDs"] = (1000 + I[rows]).astype(np.int32)
    return path


COLS = dict(x="PartType1/Coordinates[0]", y="PartType1/Coordinates[1]",
            z="PartType1/Coordinates[2]", w="PartType1/Masses")


def test_read(tmp_path):
    f = pysif.io.read_hdf5(snapshot(tmp_path / "s.hdf5"), length_scale=1e-3,
                           **COLS)
    assert f.n_particles == N and f.units == "cartesian"
    np.testing.assert_allclose(f.x, POS[:, 0] * 1e-3, rtol=1e-6)
    np.testing.assert_array_equal(f.weights, MASS.astype(pysif.real))


def test_several_files(tmp_path):
    whole = snapshot(tmp_path / "s.hdf5")
    parts = [snapshot(tmp_path / "s.0.hdf5", slice(0, 200)),
             snapshot(tmp_path / "s.1.hdf5", slice(200, None))]
    a = pysif.io.read_hdf5(whole, fraction=0.4, seed=3, **COLS)
    b = pysif.io.read_hdf5(parts, fraction=0.4, seed=3, **COLS)
    np.testing.assert_array_equal(a.x, b.x)


def test_sky(tmp_path):
    f = pysif.io.read_hdf5(snapshot(tmp_path / "s.hdf5"),
                           ra="PartType1/ParticleIDs", dec="PartType1/Masses",
                           z="PartType1/Masses")
    assert f.units == "sky" and f.x[3] == 1003


@pytest.mark.parametrize("kwargs", [
    dict(x="Nope", y="Nope", z="Nope"),
    dict(x="PartType1/Coordinates", y="PartType1/Masses", z="PartType1/Masses"),
    dict(x="PartType1/Coordinates[3]", y="PartType1/Masses", z="PartType1/Masses"),
])
def test_bad_requests(tmp_path, kwargs):
    with pytest.raises(ValueError):
        pysif.io.read_hdf5(snapshot(tmp_path / "s.hdf5"), **kwargs)


def test_refusals(tmp_path):
    path = snapshot(tmp_path / "s.hdf5")
    with pytest.raises(ValueError, match="not some of each"):
        pysif.io.read_hdf5(path, x="a", dec="b", z="c")
    with pytest.raises(ValueError, match="length_scale"):
        pysif.io.read_hdf5(path, ra="a", dec="b", z="c", length_scale=2)
    with pytest.raises(FileNotFoundError):
        pysif.io.read_hdf5(tmp_path / "missing.hdf5", **COLS)
    text = tmp_path / "t.txt"
    text.write_text("not hdf5")
    with pytest.raises(OSError, match="not an HDF5 file"):
        pysif.io.read_hdf5(text, **COLS)
