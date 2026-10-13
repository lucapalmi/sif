# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The volume= argument of the sizing helpers: that it reaches the C side,
and that each way it can be wrong raises what it says. The sizing itself is
tested in C (tests/test_finders.c, tests/test_survey.c)."""

import numpy as np
import pysif
import pytest

N, BOX, RADIUS = 200_000, 400.0, 12.0
FOOTPRINT = 4 / 3 * np.pi * 60.0**3  # a ball of radius 60 in the box


def test_mesh_finer_inside_the_footprint():
    plain = pysif.finders.suggest_mesh_cells(N, BOX, max_radius=RADIUS)
    survey = pysif.finders.suggest_mesh_cells(N, BOX, max_radius=RADIUS,
                                              volume=FOOTPRINT)
    assert survey > plain
    # None and the box's own volume are the same thing.
    assert pysif.finders.suggest_mesh_cells(N, BOX, RADIUS, BOX**3) == plain


def test_grid_finer_inside_the_footprint():
    # Tracers in an eighth of the box: half the separation, twice the cells.
    assert pysif.finders.grid_cells_from_size(256**3, 1000.0, 1.0) == 256
    assert pysif.finders.grid_cells_from_size(
        256**3, 1000.0, 1.0, volume=1000.0**3 / 8) == 512


@pytest.mark.parametrize("volume, error, message", [
    (0.0, ValueError, r"volume: 0, expected positive and finite \(None for "),
    (-5.0, ValueError, "volume: -5"),
    (float("nan"), ValueError, "volume: nan"),
    (float("inf"), ValueError, "volume: inf"),
    ("big", TypeError, None),
])
def test_bad_volume(volume, error, message):
    with pytest.raises(error, match=message):
        pysif.finders.suggest_mesh_cells(N, BOX, volume=volume)
    with pytest.raises(error, match=message):
        pysif.finders.grid_cells_from_size(N, BOX, 1.0, volume=volume)
