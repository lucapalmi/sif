# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.measure.survey_volume(): the arguments and what comes back. Its
accuracy, the separability warning and the refusals are tested in C
(tests/test_survey_volume.c)."""

import numpy as np
import pysif
import pytest

R, N = 100.0, 100_000


@pytest.fixture(scope="module")
def sphere():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(N, 3))
    v *= (R * rng.random(N) ** (1 / 3) / np.linalg.norm(v, axis=1))[:, None]
    return v.astype(pysif.real)


def test_sphere(sphere):
    volume = pysif.measure.survey_volume(pysif.field_from_numpy(*sphere.T))
    assert volume == pytest.approx(4 / 3 * np.pi * R**3, rel=0.01)


def test_observer(sphere):
    at_origin = pysif.measure.survey_volume(pysif.field_from_numpy(*sphere.T))
    shift = np.array([300.0, 250.0, 200.0], dtype=pysif.real)
    moved = pysif.field_from_numpy(*(sphere + shift).T)
    for observer in (shift, tuple(shift.tolist()), list(shift.tolist())):
        assert pysif.measure.survey_volume(moved, observer=observer) == \
            pytest.approx(at_origin, rel=1e-3)


@pytest.mark.parametrize("observer, error, message", [
    ((0.0, 0.0), ValueError, "observer: 2 coordinates, expected 3"),
    (5.0, TypeError, "observer: expected 3 coordinates"),
    ((0.0, "a", 0.0), TypeError, None),
])
def test_bad_observer(sphere, observer, error, message):
    with pytest.raises(error, match=message):
        pysif.measure.survey_volume(pysif.field_from_numpy(*sphere.T),
                                    observer=observer)


def test_sky_coordinates_refused():
    ones = np.ones(10, dtype=pysif.real)
    sky = pysif.field_from_numpy(ra=ones, dec=ones, z=ones)
    with pytest.raises(ValueError, match="sky coordinates, expected Cartesian"):
        pysif.measure.survey_volume(sky)
