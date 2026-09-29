# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.finders.suggest_mesh_cells_survey: that it reaches the C helper with
its arguments, and that each way it can refuse raises what it says. The
sizing itself is tested in C (tests/test_survey.c)."""

import numpy as np
import pysif
import pytest

N_GRID, RADIUS = 64, 12.0


def ball(n, radius, centre, seed):
    """n points uniform in a ball."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p *= (radius * rng.random(n) ** (1 / 3) / np.linalg.norm(p, axis=1))[:, None]
    return p + centre


@pytest.fixture(scope="module")
def survey():
    """Randoms in a ball, placed in a box by survey_box(), and their grid."""
    pts = ball(200_000, 60.0, np.array([500.0, -40.0, 12.0]), seed=1)
    randoms = pysif.field_from_numpy(*pts.T.astype(pysif.real))

    radii = np.array([RADIUS], dtype=pysif.real)
    offset, box = pysif.finders.survey_box(randoms, radii, N_GRID)
    randoms.translate(offset)

    grid = pysif.Grid(n_cells=N_GRID, box_length=box)
    grid.assign_cic(field=randoms)
    return grid, box


def test_finer_than_the_box_average(survey):
    grid, box = survey
    n = 200_000
    cells = pysif.finders.suggest_mesh_cells_survey(n, grid, max_radius=RADIUS)
    plain = pysif.finders.suggest_mesh_cells(n, box, max_radius=RADIUS)
    assert cells > plain


def test_refuses_a_density_contrast(survey):
    grid, box = survey
    contrast = pysif.Grid(n_cells=N_GRID, box_length=box)
    contrast.assign_cic(field=_one_point(box))
    contrast.to_density_contrast()
    with pytest.raises(ValueError, match="density contrast"):
        pysif.finders.suggest_mesh_cells_survey(100, contrast)


def test_refuses_an_empty_grid(survey):
    _, box = survey
    with pytest.raises(ValueError, match="empty"):
        pysif.finders.suggest_mesh_cells_survey(100, pysif.Grid(N_GRID, box))


@pytest.mark.parametrize("n, radius", [(0, RADIUS), (100, 1e6)])
def test_refuses_what_suggest_mesh_cells_does(survey, n, radius):
    grid, _ = survey
    with pytest.raises(ValueError):
        pysif.finders.suggest_mesh_cells_survey(n, grid, max_radius=radius)


def _one_point(box):
    half = np.array([box / 2], dtype=pysif.real)
    return pysif.field_from_numpy(half, half, half)
