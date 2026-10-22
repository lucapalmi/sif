# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Grid.assign(), its record on the grid, and deconvolution= on the finders
and pysif.measure: that each reaches the C side and comes back as it says.
The deposits and the corrections themselves are tested in C
(tests/test_field_grid.c, tests/test_fft.c)."""

import numpy as np
import pysif
import pysif.measure
import pytest

BOX, N_GRID, N = 200.0, 32, 40_000
RADII = np.array([30, 25, 20, 16], dtype=pysif.real)


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(5)
    pts = rng.random((3, 2 * N)) * BOX
    keep = (pts[0] - 100)**2 + (pts[1] - 100)**2 + (pts[2] - 100)**2 >= 35**2
    return pysif.field_from_numpy(*pts[:, keep][:, :N].astype(pysif.real))


@pytest.fixture(scope="module")
def mesh(field):
    cells = pysif.finders.suggest_mesh_cells(N, BOX, max_radius=float(RADII[0]))
    return pysif.ChainMesh(cells, BOX, field)


def contrast(field, **kw):
    g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    g.assign(field, **kw)
    g.to_density_contrast()
    return g


def test_fresh_grid_records_nothing():
    g = pysif.Grid(n_cells=8, box_length=BOX)
    assert g.assignment is None and g.interlaced is False


@pytest.mark.parametrize("scheme", ["ngp", "cic", "tsc", "pcs"])
@pytest.mark.parametrize("interlace", [False, True])
def test_assign_records_the_scheme(field, scheme, interlace):
    g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    g.assign(field, scheme, interlace=interlace)
    assert g.assignment == scheme and g.interlaced is interlace


def test_assign_defaults_to_cic(field):
    a, b = pysif.Grid(16, BOX), pysif.Grid(16, BOX)
    a.assign(field)
    b.assign_cic(field)
    assert a.assignment == b.assignment == "cic"
    a.to_density_contrast()
    b.to_density_contrast()
    cat_a = pysif.finders.spherical(a, RADII, -0.7)
    cat_b = pysif.finders.spherical(b, RADII, -0.7)
    assert np.array_equal(cat_a.radii, cat_b.radii)


def test_assign_refusals(field):
    g = pysif.Grid(n_cells=N_GRID, box_length=BOX)
    with pytest.raises(ValueError, match="'ngp', 'cic', 'tsc' or 'pcs'"):
        g.assign(field, "sph")
    with pytest.raises(TypeError):
        g.assign(field, "tsc", True)  # interlace is keyword-only
    with pytest.raises(ValueError, match="at least 4"):
        pysif.Grid(n_cells=3, box_length=BOX).assign(field, "tsc")


@pytest.mark.parametrize("deconvolution", ["full", "aliasing", "none"])
def test_finders_take_deconvolution(field, mesh, deconvolution):
    g = contrast(field, scheme="tsc", interlace=True)
    cat = pysif.finders.exodus(g, mesh, RADII, -0.7, checkpoint=False,
                               deconvolution=deconvolution)
    assert cat.n_voids > 0
    cat = pysif.finders.spherical(g, RADII, -0.7, deconvolution=deconvolution)
    assert cat.n_voids > 0


def test_deconvolution_changes_the_field(field):
    g = contrast(field)
    full = pysif.measure.delta_moments_grid(g, RADII, 0)
    alias = pysif.measure.delta_moments_grid(g, RADII, 0, deconvolution="aliasing")
    none = pysif.measure.delta_moments_grid(g, RADII, 0, deconvolution="none")
    kept = pysif.measure.delta_moments_grid(g, RADII, 0, keep_cic_window=True)
    s = [np.asarray(m.sigma)[0] for m in (full, alias, none, kept)]
    # Each correction boosts the small scales less than the last.
    assert np.all(s[0] > s[1]) and np.all(s[1] > s[2])
    assert np.array_equal(s[2], s[3])


def test_deconvolution_refusals(field, mesh):
    g = contrast(field)
    with pytest.raises(ValueError, match="'full', 'aliasing' or 'none'"):
        pysif.finders.spherical(g, RADII, -0.7, deconvolution="partial")
    with pytest.raises(ValueError, match="'full', 'aliasing' or 'none'"):
        pysif.measure.delta_moments_grid(g, RADII, 0, deconvolution="partial")
    with pytest.raises(ValueError, match="keep_cic_window=True"):
        pysif.measure.delta_moments_grid(g, RADII, 0, keep_cic_window=True,
                                         deconvolution="aliasing")



@pytest.mark.parametrize("centres", [None, "legacy", "minima", "optimised"])
def test_exodus_takes_centres(field, mesh, centres):
    g = contrast(field)
    kw = {} if centres is None else {"centres": centres}
    cat = pysif.finders.exodus(g, mesh, RADII, -0.7, checkpoint=False, **kw)
    assert cat.n_voids > 0


def test_centres_refusals(field, mesh):
    g = contrast(field)
    with pytest.raises(ValueError, match="'legacy', 'minima' or 'optimised'"):
        pysif.finders.exodus(g, mesh, RADII, -0.7, checkpoint=False,
                             centres="refined")
    with pytest.raises(TypeError):
        pysif.finders.exodus(g, mesh, RADII, -0.7, checkpoint=False,
                             centre_is_minimum=True)
