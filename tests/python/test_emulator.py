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


# --- the EVM size function ---

H = 0.67
SIGMA12_Z99, SIGMA12_Z0 = 0.010642019592327554, 0.8247254042480003


@pytest.fixture(scope="module")
def evm_kernel():
    return M.VarianceKernel(k=np.geomspace(3.1e-4, 100.0, 600), r_min=0.3,
                            r_max=1000.0, dln_r=0.02)


def evm_pipeline(kernel, pk, radii, delta_nl):
    """The model as the calibration scripts evaluate it, step by step from
    the public pieces and with the constants as the thesis quotes them."""
    walk = kernel(pk)
    ln_r = np.log(np.asarray(walk.radii, float))
    ln_s = np.log(np.asarray(walk.sigma, float))

    def sigma_at(r):
        return np.exp(np.interp(np.log(r), ln_r, ln_s))

    sigma12 = float(sigma_at(12.0 * H))
    q = sigma12 / SIGMA12_Z99
    depth = -M.spherical_map_linear(delta_nl)
    alpha = 1.28421 * depth ** 1.377811 * q ** (1.0 - 1.377811)
    beta, gamma = 0.015393 * q, 0.423650

    barrier = M.ep_barrier_smt(walk.sigma, alpha, beta, gamma)
    r_l = radii * (1.0 + delta_nl) ** (1.0 / 3.0)
    f = np.asarray(M.ep_multiplicity_function_emu(walk, barrier, r_l), float)
    volume_conserving = (3.0 * f / (4.0 * np.pi * r_l ** 2)
                         * (1.0 + delta_nl))

    sigma = sigma_at(radii)
    nu = alpha * (1.0 + (beta / sigma) ** gamma) / sigma
    u = np.log(nu) - 0.3483074
    w = np.log(1.0 + delta_nl)
    ell = np.log(sigma12 / SIGMA12_Z0)
    s = np.log(sigma)
    a, c1, c2, c3, c4 = 0.194614, 0.232528, 0.739356, 0.817131, 1.679090
    big_q = w - ell ** 2 * w - ell * u
    e = u + np.exp(w - ell * (ell - big_q ** 2))
    ln_t = a * (ell * (w - ell * u * np.exp(ell))
                + (ell * w + s + c3)
                * (u ** 2 - e ** 2 * (c1 * u ** 2 - c2)
                   * (u ** 2 + u + w - np.exp(-ell * w)))
                + (u - np.exp(w) + c4) ** 2)
    return volume_conserving * np.exp(ln_t)


@pytest.mark.parametrize("delta_nl", [-0.3, -0.5, -0.7])
@pytest.mark.parametrize("growth", [1.0, 0.6])
def test_evm_is_the_separate_pipeline(evm_kernel, delta_nl, growth):
    pk = cdm(evm_kernel.k) * growth ** 2
    radii = np.geomspace(6.0, 60.0, 24)
    expected = evm_pipeline(evm_kernel, pk, radii, delta_nl)
    got = M.size_function_evm_kernel(
        evm_kernel, pk, radii, M.spherical_map_linear(delta_nl), H)
    # the barrier constants above are rounded to the quoted six figures
    np.testing.assert_allclose(got.vsf, expected, rtol=2e-3)


def test_evm_wrapper_is_the_kernel_version(evm_kernel):
    pk = cdm(evm_kernel.k)
    radii = np.geomspace(6.0, 60.0, 12)
    delta_v = M.spherical_map_linear(-0.5)
    wrapped = M.size_function_evm(evm_kernel.k, pk, radii, delta_v, H)
    direct = M.size_function_evm_kernel(evm_kernel, pk, radii, delta_v, H)
    np.testing.assert_array_equal(wrapped.vsf, direct.vsf)
    np.testing.assert_array_equal(wrapped.r_centres, radii.astype(np.float32))


def test_evm_walk_is_the_kernel_version(evm_kernel):
    pk = cdm(evm_kernel.k)
    walk = evm_kernel(pk)
    radii = np.geomspace(6.0, 60.0, 12)
    for delta_nl in (-0.3, -0.5, -0.7):
        delta_v = M.spherical_map_linear(delta_nl)
        shared = M.size_function_evm_walk(walk, radii, delta_v, H, units="r")
        direct = M.size_function_evm_kernel(
            evm_kernel, pk, radii, delta_v, H, units="r")
        np.testing.assert_array_equal(shared.vsf, direct.vsf)


def test_evm_arguments_reach_the_c_side(evm_kernel):
    pk = cdm(evm_kernel.k)
    radii = np.geomspace(6.0, 60.0, 12)
    delta_v = M.spherical_map_linear(-0.5)
    per_ln_r = M.size_function_evm_kernel(evm_kernel, pk, radii, delta_v, H)
    per_r = M.size_function_evm_kernel(
        evm_kernel, pk, radii, delta_v, H, units="r")
    np.testing.assert_allclose(per_r.vsf, per_ln_r.vsf / radii, rtol=1e-5)

    exact = M.size_function_evm_kernel(
        evm_kernel, pk, radii, delta_v, H, method="eds")
    assert not np.array_equal(exact.vsf, per_ln_r.vsf)

    other_h = M.size_function_evm_kernel(evm_kernel, pk, radii, delta_v, 0.7)
    assert not np.array_equal(other_h.vsf, per_ln_r.vsf)


def test_evm_tail_is_zero_not_nan(evm_kernel):
    radii = np.geomspace(6.0, 900.0, 60)
    vsf = M.size_function_evm_kernel(
        evm_kernel, cdm(evm_kernel.k), radii,
        M.spherical_map_linear(-0.5), H).vsf
    assert np.isfinite(vsf).all()
    assert vsf[-1] == 0.0 and vsf[0] > 0.0


def test_evm_refusals(evm_kernel):
    pk = cdm(evm_kernel.k)
    radii = np.geomspace(6.0, 60.0, 6)
    with pytest.raises(ValueError, match="delta_v"):
        M.size_function_evm_kernel(evm_kernel, pk, radii, 0.5, H)
    with pytest.raises(ValueError, match="h must"):
        M.size_function_evm(evm_kernel.k, pk, radii, -0.9, 0.0)
    with pytest.raises(ValueError, match="one per node"):
        M.size_function_evm_kernel(evm_kernel, pk[:-1], radii, -0.9, H)
    with pytest.raises(ValueError):
        M.size_function_evm_kernel(evm_kernel, pk, radii * 100.0, -0.9, H)
    with pytest.raises(ValueError):
        M.size_function_evm_kernel(evm_kernel, pk, radii[::-1], -0.9, H)
    with pytest.raises(TypeError):
        M.size_function_evm_kernel("kernel", pk, radii, -0.9, H)

    gaussian = M.VarianceKernel(window="gaussian")
    with pytest.raises(ValueError):
        M.size_function_evm_kernel(
            gaussian, cdm(gaussian.k), radii, -0.9, H)
    with pytest.raises(ValueError):
        M.size_function_evm_walk(gaussian(cdm(gaussian.k)), radii, -0.9, H)
    with pytest.raises(TypeError):
        M.size_function_evm_walk(evm_kernel, radii, -0.9, H)


def test_spherical_map_methods():
    # Omega_m = 1 is Einstein-de Sitter; at 0.31858 the reference from an
    # independent integration (see tests/test_vsf_models.c).
    for delta_nl in (-0.3, -0.8):
        assert M.spherical_map_linear(delta_nl, method="lcdm", omega_m=1.0) == \
            pytest.approx(M.spherical_map_linear(delta_nl, method="eds"), rel=2e-6)
    assert M.spherical_map_linear(-0.8, method="lcdm", omega_m=0.31858) == \
        pytest.approx(-2.78584771, rel=2e-6)
    delta_l = M.spherical_map_linear(-0.5, method="lcdm", omega_m=0.3)
    assert M.spherical_map_nonlinear(delta_l, method="lcdm", omega_m=0.3) == \
        pytest.approx(-0.5, rel=1e-6)

    with pytest.raises(ValueError, match="required"):
        M.spherical_map_linear(-0.5, method="lcdm")
    with pytest.raises(ValueError, match="only method='lcdm'"):
        M.spherical_map_linear(-0.5, method="eds", omega_m=0.3)
    with pytest.raises(ValueError, match="expected 0 < omega_m <= 1"):
        M.spherical_map_nonlinear(-1.0, method="lcdm", omega_m=1.2)
    with pytest.raises(ValueError, match="expected 'b94', 'eds' or 'lcdm'"):
        M.spherical_map_linear(-0.5, method="exact")


def test_size_function_lcdm(evm_kernel):
    pk = cdm(evm_kernel.k)
    radii = np.geomspace(6.0, 60.0, 12)
    delta_v = M.spherical_map_linear(-0.5)
    eds = M.size_function_evm_kernel(evm_kernel, pk, radii, delta_v, H, method="eds")
    one = M.size_function_evm_kernel(
        evm_kernel, pk, radii, delta_v, H, method="lcdm", omega_m=1.0)
    lcdm = M.size_function_evm_kernel(
        evm_kernel, pk, radii, delta_v, H, method="lcdm", omega_m=0.31858)
    np.testing.assert_allclose(one.vsf, eds.vsf, rtol=1e-5)
    assert not np.array_equal(lcdm.vsf, eds.vsf)
