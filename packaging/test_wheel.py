# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Smoke test of a built wheel, run by cibuildwheel against the installed wheel.

Not the test suite -- that runs on the C library, in CI -- but a check that
the wheel is whole: that every library it bundles loads and works on a
machine that has none of them. So it touches each one:

  - FFTW and OpenMP, through a finder run (the smoothing is FFT-based, the
    rest is OpenMP-parallel);
  - HDF5, through a file round trip;
  - HDF5 again, next to h5py's own copy in the same process: pysif writes, h5py
    rewrites a dataset compressed, pysif reads it back -- which also checks
    that the deflate filter was built in.
"""

import os
import tempfile
import warnings

import h5py
import numpy as np
import pysif

warnings.simplefilter("error")  # a text fallback would mean HDF5 is missing

pysif.init()

rng = np.random.default_rng(1)
N, L = 100_000, 200.0
x, y, z = (rng.random(N) * L for _ in range(3))
w = np.ones(N)
for cx, cy, cz, r in [(50, 50, 50, 26), (150, 140, 60, 21), (70, 160, 150, 18)]:
    w[(x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r**2] = 0.0

field = pysif.Field()
field.from_numpy(x, y, z, weights=w)
grid = pysif.Grid(64, L)
grid.assign_cic(field)
grid.to_density_contrast()

radii = [30.0, 25.0, 20.0, 16.0]
mesh = pysif.ChainMesh(pysif.finders.suggest_mesh_cells(N, L, radii[0]), L, field)
cat = pysif.finders.exodus(grid, mesh, radii, -0.7, 0.0)
assert cat.n_voids >= 3, f"expected the three planted voids, found {cat.n_voids}"

path = os.path.join(tempfile.mkdtemp(), "wheel.h5")
pysif.io.write_catalog_hdf5(path, cat)
pysif.io.set_hdf5_attr(path, "origin", "wheel test")
back = pysif.io.read_catalog_hdf5(path)
assert np.array_equal(back.radii, cat.radii)
assert pysif.io.get_hdf5_attr(path, "origin") == "wheel test"

with h5py.File(path, "r+") as f:
    assert f.attrs["sif_format"] == "sif"
    radii_data = f["catalog/radii"][...]
    del f["catalog/radii"]
    f["catalog"].create_dataset("radii", data=radii_data, compression="gzip")

back = pysif.io.read_catalog_hdf5(path)
assert np.array_equal(back.radii, cat.radii), "compressed dataset read wrong"

print(f"pysif wheel OK: {cat.n_voids} voids, HDF5 round trip, h5py interop")
