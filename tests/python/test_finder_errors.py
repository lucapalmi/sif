# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""What the finders raise when they refuse: the library's own reason as the
message, and a class that says what kind of failure it was. Which inputs are
refused is tested in C (tests/test_finders.c, tests/test_survey.c)."""

import numpy as np
import pysif
import pytest

BOX, N_GRID, N = 100.0, 32, 20_000


def radii(*r):
    return np.array(r, dtype=pysif.real)


@pytest.fixture(scope="module")
def tracers():
    rng = np.random.default_rng(1)
    return pysif.field_from_numpy(*(rng.random((3, N)) * BOX).astype(pysif.real))


@pytest.fixture
def grid(tracers):
    g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    g.assign_cic(field=tracers)
    g.to_density_contrast()
    return g


@pytest.fixture(scope="module")
def mesh(tracers):
    return pysif.ChainMesh(pysif.finders.suggest_mesh_cells(N, BOX), BOX, tracers)


@pytest.mark.parametrize("args, kwargs, message", [
    ((radii(8, -3), -0.7), {}, r"radii\[1\] = -3: not a positive radius"),
    ((radii(8), 0.5), {}, r"threshold 0.5: not a density contrast in \(-1, 0\)"),
    ((radii(8), -0.7), {"overlap_fraction": 2},
     r"overlap_fraction 2: not in \[0, 1\]"),
])
def test_bad_arguments(grid, mesh, args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pysif.finders.exodus(grid, mesh, *args, **kwargs)
    with pytest.raises(ValueError, match=message):
        pysif.finders.spherical(grid, *args, **kwargs)


def test_box_mismatch(mesh):
    g = pysif.Grid(n_cells=N_GRID, box_length=2 * BOX)
    with pytest.raises(ValueError, match="mesh: box of 100, grid: box of 200"):
        pysif.finders.exodus(g, mesh, radii(8), -0.7)


def test_failed_run_raises(grid, mesh):
    # The first rung fails: its search sphere does not fit the mesh. That used
    # to come back as an empty catalogue.
    with pytest.raises(ValueError, match="wider than the mesh"):
        pysif.finders.exodus(grid, mesh, radii(80), -0.7)


def test_survey_without_padding(tracers, mesh):
    data = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    data.assign_cic(field=tracers)
    rand = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    rand.assign_cic(field=tracers)
    rmesh = pysif.ChainMesh(16, BOX, tracers)
    with pytest.raises(ValueError, match="of padding on every side"):
        pysif.finders.exodus_survey(data, rand, mesh, rmesh, radii(8), -0.7)


def test_survey_box_refusals(tracers):
    with pytest.raises(ValueError, match="n_cells 8: a survey box needs at least 16"):
        pysif.finders.survey_box(tracers, radii(8), 8)
    bad = pysif.field_from_numpy(radii(1, np.nan), radii(1, 2), radii(1, 2))
    with pytest.raises(ValueError, match="randoms: 1 of 2 with a coordinate"):
        pysif.finders.survey_box(bad, radii(8), 64)
