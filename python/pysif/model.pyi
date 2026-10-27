# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The types of pysif.model; see __init__.pyi. A flag that changes what a
# function returns (return_counts, return_domain) is an overload per value.

"""Theoretical predictions.

Size functions from the BBKS peak statistics, the SvdW and Vdn
excursion-set models, and the excursion-set first-crossing
problem in both its Monte Carlo and emulated forms. Also the
spectral moments and covariance of a model power spectrum, and
the spherical-evolution map between linear and non-linear
density contrast.

These return the same containers pysif.measure fills, so a
prediction and a measurement can be compared directly.
"""

from typing import Literal, TypedDict, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import SizeFunction, _RealArray
from .measure import DeltaMoments

_Window = Literal["top_hat", "gaussian"]
_G = Literal["fitted", "exact"]
_Method = Literal["b94", "exact"]
_Units = Literal["ln_r", "r"]
_Counts = NDArray[np.uint64]

class _Domain(TypedDict):
    in_domain: bool
    n_bins_outside: int
    nu_origin: float
    first_step_mass: float
    expected_error: float

@overload
def comoving_distance(z: float, omega_m: float, omega_de: float | None = None,
                      omega_r: float = 0.0, w0: float = -1.0,
                      wa: float = 0.0) -> float:
    """Line-of-sight comoving distance to redshift z, in Mpc/h.

    For a w0waCDM background: dark energy with w(a) = w0 + wa (1 - a),
    curvature 1 - omega_m - omega_de - omega_r. The distance a survey's
    tracers are placed at by Field.convert_sky_coordinates(), with the
    same arguments.

    Args:
        z: A redshift, or an array of them; at least 0.
        omega_m: Matter density today.
        omega_de: Dark-energy density today; None, the default, makes the
            model flat (1 - omega_m - omega_r).
        omega_r: Radiation density today.
        w0, wa: The dark-energy equation of state: -1 and 0 for a
            cosmological constant.

    Returns:
        float or ndarray: D_C in Mpc/h, shaped like z.

    Raises:
        ValueError: For a negative or non-finite redshift, a negative
            density, or a model with no expansion history out to z.
    """
@overload
def comoving_distance(z: ArrayLike, omega_m: float,
                      omega_de: float | None = None, omega_r: float = 0.0,
                      w0: float = -1.0,
                      wa: float = 0.0) -> NDArray[np.float64]: ...

def delta_moments_pk(k: ArrayLike, pk: ArrayLike, radii: ArrayLike,
                     order: int, window: _Window = "top_hat") -> DeltaMoments:
    """Evaluates the spectral moments sigma_0..sigma_order from a tabulated power spectrum, as the continuum integral.
    The theory counterpart of measure.delta_moments_grid: no box, no realisation, no cosmic variance.
    """

def bbks_g(gamma: float, w: float, g: _G = "fitted") -> float:
    """The BBKS G(gamma, w) function, w = gamma * nu. Fitted form by default (BBKS 1986 eq. 4.4); g='exact' quadratures the defining integral (Wu 2020 eqs. 18-19)."""

def bbks_number_density_differential(nu: ArrayLike, gamma: ArrayLike,
                                     r_star: ArrayLike,
                                     g: _G = "fitted") -> _RealArray:
    """Differential number density of maxima of a Gaussian field, per unit volume per unit nu (BBKS 1986 eq. 4.3).
    Takes parallel nu, gamma and r_star arrays; gamma and r_star come from the matching properties of a DeltaMoments.
    """

def bbks_number_density_cumulative(moments: DeltaMoments, delta: float,
                                   g: _G = "fitted") -> _RealArray:
    """Number density of Gaussian-field maxima above a density contrast, one value per smoothing radius of the given DeltaMoments (order >= 2).
    The threshold is converted per radius as nu_t = delta / sigma_0(R).
    """

def size_function_bbks(moments: DeltaMoments, delta: float,
                       units: _Units = "ln_r",
                       g: _G = "fitted") -> SizeFunction:
    """Size function of a Gaussian field above a density threshold, as a SizeFunction over the radii of the given DeltaMoments (order >= 2).
    units='ln_r' (default) puts -dC/dlnR in vsf, units='r' puts -dC/dR; options records which. Evaluated as a finite difference over the moments' radii, which must be strictly increasing.
    """

def delta_sigma_slope_pk(k: ArrayLike, pk: ArrayLike, radii: ArrayLike,
                         window: _Window = "top_hat") -> _RealArray:
    """Logarithmic slope dln(sigma)/dln(R) from a tabulated power spectrum.
    Differentiates the window under the integral, so it is exact rather than a finite difference: the radii need not be ordered or finely spaced.
    """

def spherical_map_nonlinear(delta_linear: float,
                            method: _Method = "b94") -> float:
    """Non-linear density contrast a void of a given linear contrast evolves to.
    method='b94' (default) uses the Bernardeau (1994) fit, method='exact' solves the Einstein-de Sitter expansion by root-find. The expansion factor r_NL/r_L is (1 + delta_NL)**(-1/3); delta_L = -2.7 gives ~1.69.
    """

def spherical_map_linear(delta_nonlinear: float,
                         method: _Method = "b94") -> float:
    """Linear density contrast that evolves into a given non-linear one, the inverse of delta_nonlinear.
    Wanted when a barrier is quoted as an observed underdensity rather than as a linear threshold. Same method= choices.
    """

def svdw_multiplicity_function(sigma: ArrayLike, delta_v: float,
                               delta_c: float) -> _RealArray:
    """Excursion-set void multiplicity function f_ln(sigma), Sheth & van de Weygaert (2004).
    Shared by the SvdW and Vdn size functions, which differ only in the Lagrangian-to-Eulerian mapping.
    """

def size_function_svdw(k: ArrayLike, pk: ArrayLike, radii: ArrayLike,
                       delta_v: float, delta_c: float, units: _Units = "ln_r",
                       method: _Method = "b94") -> SizeFunction:
    """Sheth & van de Weygaert void size function, as a SizeFunction over the given Eulerian radii.
    Number-conserving, so its void volume fraction exceeds one at large radii; present as the reference rather than the recommendation.
    The expansion factor is not an argument: delta_v already fixes it. Use method='b94' (default) or 'exact' to choose how.
    """

def delta_covariance_pk(
        k: ArrayLike, pk: ArrayLike, radii: ArrayLike,
        window: _Window = "top_hat"
) -> tuple[NDArray[np.float64], _RealArray, _RealArray, NDArray[np.float64]]:
    """Covariance of the smoothed field between every pair of smoothing radii, from a tabulated power spectrum.
    Returns (cov, sigma, high_k_fraction, deriv_variance). cov is the packed lower triangle in float64, S(i,j) at i*(i+1)/2 + j for j <= i; sigma is the sqrt of its diagonal, which equals sigma_0 from delta_moments_pk; deriv_variance is <(d delta / dS)^2> with S = sigma^2, differentiated under the integral rather than differenced off the matrix, so it does not depend on how finely radii was sampled. window='top_hat' (default) or 'gaussian'. Requires a P(k) sampled on at least ~1000 log-spaced points.
    """

def ep_barrier_smt(sigma: ArrayLike, alpha: float, beta: float,
                   gamma: float) -> _RealArray:
    """Sheth-Mo-Tormen moving barrier, alpha * (1 + (beta/sigma)**gamma).
    Feed it the sigma returned by delta_covariance_pk, so the barrier and the walk share one variance.
    """

@overload
def ep_first_crossing_counts(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             collapse: float | ArrayLike | None = None,
                             return_collapse_counts: Literal[False] = False
                             ) -> _Counts:
    """Raw first-crossing counts of a correlated random walk against a moving barrier, one per radius.
    radii ascending, cov the packed triangle from delta_covariance_pk, barrier one entry per radius. The result is a deterministic function of (seed, n_paths), independent of the thread count.
    collapse adds void-in-cloud: delta_c, positive, as a number or one entry per radius. A walk that reaches it before the void barrier is inside a collapsing region and is not counted as a void at any radius. The same seed gives the same walks with or without it. return_collapse_counts=True returns (counts, collapse_counts), the latter the walks discarded at each radius.
    """
@overload
def ep_first_crossing_counts(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             collapse: float | ArrayLike | None = None, *,
                             return_collapse_counts: Literal[True]
                             ) -> tuple[_Counts, _Counts]: ...
@overload
def ep_first_crossing_counts(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             collapse: float | ArrayLike | None = None,
                             return_collapse_counts: bool = False
                             ) -> _Counts | tuple[_Counts, _Counts]: ...

@overload
def ep_multiplicity_function(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             return_counts: Literal[False] = False,
                             collapse: float | ArrayLike | None = None
                             ) -> _RealArray:
    """Void multiplicity function from the first crossing of a moving barrier by a correlated random walk.
    Returns len(radii) - 1 bin-centred values: a Monte Carlo can only place a crossing between two consecutive smoothing scales. Same arguments as ep_first_crossing_counts; collapse adds void-in-cloud.
    return_counts=True returns (mult, counts) instead, with the raw len(radii) counts the multiplicity was built from. Prefer it to a second call to ep_first_crossing_counts, which would run the whole walk again.
    """
@overload
def ep_multiplicity_function(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             *, return_counts: Literal[True],
                             collapse: float | ArrayLike | None = None
                             ) -> tuple[_RealArray, _Counts]: ...
@overload
def ep_multiplicity_function(radii: ArrayLike, cov: ArrayLike,
                             barrier: ArrayLike, n_paths: int, seed: int = 0,
                             return_counts: bool = False,
                             collapse: float | ArrayLike | None = None
                             ) -> _RealArray | tuple[_RealArray, _Counts]: ...

@overload
def ep_multiplicity_function_emu(radii: ArrayLike, sigma: ArrayLike,
                                 barrier: ArrayLike, deriv_variance: ArrayLike,
                                 return_domain: Literal[False] = False,
                                 binned: bool = False,
                                 collapse: float | ArrayLike | None = None,
                                 cov: ArrayLike | None = None) -> _RealArray:
    """Emulated void multiplicity function: the same quantity as ep_multiplicity_function, without the random walks.
    The up-crossing rate of Verza et al. (2024) eq. (3.15) -- Musso-Sheth with the exact scale-dependent <(d delta / dS)^2> -- corrected by a small trained network. About 0.06% of the Monte Carlo it was fitted against (median, radii below 80 Mpc/h), for walks starting anywhere from ~2 to 250 Mpc/h, and a few million times faster.
    Takes sigma, NOT the packed covariance the Monte Carlo needs: the single-barrier emulator reads only the diagonal, so it wants n values rather than n(n+1)/2 (void-in-cloud is the exception, below). deriv_variance is required, from delta_covariance_pk.
    Returns the answer the Monte Carlo CONVERGES to, not the answer it gives on a coarse grid; the emulated result agrees to 0.02% on 65, 129 and 257 radii.
    Returns the first-crossing density per unit R at each radius (n values); binned=True returns its average over each bin instead, n - 1 values on the bin centres, which is what the Monte Carlo ep_multiplicity_function measures.
    Place the largest radius where nu = barrier / sigma is ~3.5 or more (further out costs nothing; with collapse, where delta_c / sigma is ~4 or more too) and keep nu above ~0.21 at the smallest; collapse (delta_c, a number or one per radius) adds void-in-cloud, as a factor K <= 1 on the single-barrier answer (about 0.09% where it matters), and needs cov, the packed covariance from delta_covariance_pk. return_domain=True returns (mult, domain) with in_domain, n_bins_outside, nu_origin, first_step_mass and expected_error.
    """
@overload
def ep_multiplicity_function_emu(radii: ArrayLike, sigma: ArrayLike,
                                 barrier: ArrayLike, deriv_variance: ArrayLike,
                                 *, return_domain: Literal[True],
                                 binned: bool = False,
                                 collapse: float | ArrayLike | None = None,
                                 cov: ArrayLike | None = None
                                 ) -> tuple[_RealArray, _Domain]: ...
@overload
def ep_multiplicity_function_emu(radii: ArrayLike, sigma: ArrayLike,
                                 barrier: ArrayLike, deriv_variance: ArrayLike,
                                 return_domain: bool = False,
                                 binned: bool = False,
                                 collapse: float | ArrayLike | None = None,
                                 cov: ArrayLike | None = None
                                 ) -> _RealArray | tuple[_RealArray, _Domain]:
    ...

def size_function_vdn(k: ArrayLike, pk: ArrayLike, radii: ArrayLike,
                      delta_v: float, delta_c: float, units: _Units = "ln_r",
                      method: _Method = "b94") -> SizeFunction:
    """Volume-conserving (Vdn) void size function, as a SizeFunction over the given Eulerian radii.
    Identical to SvdW but divided by the Eulerian volume, which keeps the void volume fraction below unity. Same arguments as size_function_svdw.
    """
