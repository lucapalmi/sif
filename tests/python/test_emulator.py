# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.model.VarianceKernel, Walk and ep_multiplicity_function_emu: that
the arguments reach the C side as meant, what comes back and in what shape,
and what is refused. The kernel's accuracy and the emulator itself are tested
in C (tests/test_ep_emu.c)."""

import numpy as np
import pysif
import pytest

M = pysif.model


def cdm(k):
    """A smooth CDM-like P(k), n_s = 0.96 through the BBKS transfer, at
    sigma_8 ~ 0.8 so the barriers below land where the emulator was
    trained."""
    k = np.asarray(k, dtype=float)
    q = k / 0.21
    t = (np.log(1 + 2.34 * q) / (2.34 * q)
         * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3
            + (6.71 * q) ** 4) ** -0.25)
    return 1.468e6 * k ** 0.96 * t * t


@pytest.fixture(scope="module")
def kernel():
    return M.VarianceKernel()


@pytest.fixture(scope="module")
def walk(kernel):
    return kernel(cdm(kernel.k))


def smt(walk):
    return M.ep_barrier_smt(walk.sigma, 0.096, 1.15, 0.43)


# --- the kernel and the walk ---

def test_default_kernel(kernel):
    assert kernel.k.size == 600
    np.testing.assert_allclose(kernel.k[[0, -1]], [1e-4, 100.0], rtol=1e-6)
    r = kernel.radii
    assert r[0] == pytest.approx(0.5) and r[-1] <= 400.0
    np.testing.assert_allclose(np.diff(np.log(r)), 0.02, rtol=1e-4)
    assert kernel.dln_r == pytest.approx(0.02) and kernel.window == "top_hat"


def test_walk_views(walk, kernel):
    assert np.interp(np.log(8.0), np.log(walk.radii), walk.sigma) == \
        pytest.approx(0.8, rel=0.05)
    assert walk.n_radii == kernel.radii.size
    assert walk.sigma.dtype == pysif.real
    assert walk.deriv_variance.dtype == np.float64
    np.testing.assert_array_equal(walk.radii, kernel.radii)
    np.testing.assert_array_equal(walk.k, kernel.k)
    assert np.all(np.diff(walk.sigma) < 0)
    with pytest.raises(ValueError):
        walk.sigma[0] = 1.0


def test_walk_is_linear_in_pk(kernel, walk):
    twice = kernel(2 * cdm(kernel.k))
    np.testing.assert_allclose(twice.sigma, np.sqrt(2) * walk.sigma, rtol=1e-6)
    np.testing.assert_allclose(twice.deriv_variance, walk.deriv_variance / 2,
                               rtol=1e-6)


def test_sigma_matches_the_table_quadrature(kernel, walk):
    k = np.geomspace(1e-4, 100.0, 20000)
    r = walk.radii[::40]
    _, sigma, _, _ = M.delta_covariance_pk(k, cdm(k), r)
    np.testing.assert_allclose(walk.sigma[::40], sigma, rtol=1e-5)


def test_own_nodes_and_window():
    k = np.geomspace(1e-3, 50.0, 300)
    kn = M.VarianceKernel(k, r_min=2.0, r_max=100.0, dln_r=0.05,
                          window="gaussian")
    np.testing.assert_allclose(kn.k, k, rtol=1e-6)
    assert kn.window == "gaussian" and kn(cdm(k)).window == "gaussian"
    assert kn.radii[0] == pytest.approx(2.0) and kn.radii[-1] <= 100.0


@pytest.mark.parametrize("kwargs", [
    dict(k=[1.0, 2.0, 3.0]),
    dict(k=[1.0, 2.0, 2.0, 3.0]),
    dict(r_min=2.0, r_max=1.0),
    dict(r_min=1.0, r_max=1.05),
    dict(dln_r=0.0),
    dict(window="sharp_k"),
])
def test_kernel_refuses(kwargs):
    with pytest.raises(ValueError):
        M.VarianceKernel(**kwargs)


def test_walk_refuses(kernel):
    with pytest.raises(ValueError, match="one per node"):
        kernel(np.ones(10))
    pk = cdm(kernel.k)
    pk[3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        kernel(pk)


# --- the emulator ---

def test_emulator_at_radii_and_binned(walk):
    radii = np.geomspace(10.0, 60.0, 20)
    f = M.ep_multiplicity_function_emu(walk, smt(walk), radii)
    fb = M.ep_multiplicity_function_emu(walk, smt(walk), radii, binned=True)
    assert f.shape == (20,) and fb.shape == (19,)
    assert np.all(f > 0) and np.all(fb > 0)
    lo, hi = np.minimum(f[:-1], f[1:]), np.maximum(f[:-1], f[1:])
    assert np.all((fb >= lo * (1 - 1e-6)) & (fb <= hi * (1 + 1e-6)))


def test_emulator_domain(walk):
    f, dom = M.ep_multiplicity_function_emu(walk, smt(walk), [10.0, 20.0],
                                            return_domain=True)
    assert set(dom) == {"in_domain", "n_bins_outside", "nu_origin",
                        "first_step_mass", "expected_error"}
    assert dom["nu_origin"] >= 4.5


def test_collapse_scalar_or_per_radius(walk):
    radii = np.geomspace(2.0, 10.0, 15)
    barrier = M.ep_barrier_smt(walk.sigma, 0.5, 0.3, 0.8)
    single = M.ep_multiplicity_function_emu(walk, barrier, radii)
    scalar = M.ep_multiplicity_function_emu(walk, barrier, radii,
                                            collapse=1.686)
    array = M.ep_multiplicity_function_emu(
        walk, barrier, radii, collapse=np.full(walk.n_radii, 1.686))
    np.testing.assert_array_equal(scalar, array)
    assert np.all(scalar <= single * (1 + 1e-6)) and np.any(scalar < single)


def test_emulator_refuses(walk, kernel):
    b = smt(walk)
    with pytest.raises(TypeError, match="Walk"):
        M.ep_multiplicity_function_emu(kernel, b, [10.0])
    with pytest.raises(ValueError, match="one per lattice radius"):
        M.ep_multiplicity_function_emu(walk, b[:10], [10.0])
    with pytest.raises(ValueError, match="lattice"):
        M.ep_multiplicity_function_emu(walk, b, [0.1, 10.0])
    with pytest.raises(ValueError, match="increasing"):
        M.ep_multiplicity_function_emu(walk, b, [20.0, 10.0])
    with pytest.raises(ValueError, match="at least 2"):
        M.ep_multiplicity_function_emu(walk, b, [10.0], binned=True)
    with pytest.raises(ValueError, match="collapse"):
        M.ep_multiplicity_function_emu(walk, b, [10.0], collapse=-1.0)
