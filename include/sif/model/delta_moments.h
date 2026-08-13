/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file delta_moments.h
 * @brief Spectral moments and the covariance of the smoothed field, evaluated
 * from a tabulated power spectrum.
 *
 * The modelled counterpart to the measured moments in `measure/`: same
 * container, same window conventions, so a measurement and a prediction can be
 * compared without converting between them.
 *
 * Everything here quadratures the supplied table directly rather than
 * interpolating it, so accuracy follows from how finely P(k) was sampled. See
 * the note on sif_delta_covariance_pk() for what "finely" means.
 */

#ifndef SIF_MODEL_DELTA_MOMENTS_H
#define SIF_MODEL_DELTA_MOMENTS_H

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/delta_moments.h"

/* Largest number of radii a covariance is built over. */
#define SIF_COV_MAX_RADII 4096

/**
 * @brief Index of S(i, j) in a packed lower triangle, for j <= i.
 *
 * Row-major over the lower triangle, in the ascending order of the radii.
 */
#define SIF_COV_INDEX(i, j) ((size_t)(i) * ((size_t)(i) + 1) / 2 + (size_t)(j))

/**
 * @brief Number of entries in a packed lower triangle of n rows.
 */
#define SIF_COV_SIZE(n) ((size_t)(n) * ((size_t)(n) + 1) / 2)

/**
 * @brief Evaluates the spectral moments from a tabulated power spectrum.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Power spectrum sampled at k. Subtract any Poisson floor before
 * calling; this takes the table as given.
 * @param n_points Length of k and pk, at least 2
 * @param radii Smoothing radii, strictly positive
 * @param n_radii Number of radii
 * @param order Highest moment order, at most SIF_MAX_MOMENT_ORDER
 * @param opt SIF_DELTA_FILTER_*
 *
 * @return Newly allocated moment set, or NULL on invalid input or failure.
 */
SIF_NODISCARD sif_delta_moments_t* sif_delta_moments_pk(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, uint8_t order, sif_option opt);

/**
 * @brief Logarithmic slope dln(sigma)/dln(R) of the r.m.s. density contrast.
 *
 * Evaluated by differentiating the window under the integral, so it is exact
 * for the tabulated spectrum rather than a finite difference over `radii`:
 * the radii need not be ordered, spaced, or numerous.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Power spectrum sampled at k
 * @param n_points Length of k and pk, at least 2
 * @param radii Smoothing radii, strictly positive
 * @param n_radii Number of radii
 * @param opt SIF_DELTA_FILTER_*
 *
 * @return Newly allocated array of n_radii slopes, negative where sigma falls
 * with R, released with sif_free_aligned, or NULL on invalid input.
 */
SIF_NODISCARD sif_real* sif_delta_sigma_slope_pk(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_option opt);

/**
 * @brief Covariance of the smoothed density field between every pair of
 * smoothing radii.
 *
 *     S(R_i, R_j) = (1 / 2 pi^2) integral dk k^2 P(k) W(k R_i) W(k R_j)
 *
 * Its diagonal is sigma_0^2(R), and reproduces sif_delta_moments_pk to
 * rounding: both use the same window and the same trapezoid in log k.
 *
 * @note Quadratured directly, with no interpolation of P(k), so the accuracy
 * is set by how finely the table is sampled rather than by an interpolant.
 * The top-hat window oscillates as cos(kR) above kR ~ 1, and resolving that
 * needs on the order of 1000 log-spaced points across the range where the
 * integrand has support; 4000 over eleven decades leaves the residual below
 * 1e-7. A coarse table will not be detected, it will simply be integrated
 * badly.
 *
 * @note Accumulated as a Gram product, S = A A^T, so the result is positive
 * semi-definite by construction for any non-negative P(k) -- the property
 * anything factorizing this matrix depends on.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Power spectrum sampled at k, non-negative at every point. Stricter
 * than sif_delta_moments_pk, which never takes its square root.
 * @param n_points Length of k and pk, at least 2; see the sampling note above
 * @param radii Smoothing radii, strictly positive, at least 1 and at most
 * SIF_COV_MAX_RADII. Packed in the order given; consumers generally require
 * ascending.
 * @param n_radii Number of radii
 * @param sigma Optional output, n_radii entries: sqrt of the diagonal, and the
 * sigma the field described by this covariance actually has. Pass NULL to skip.
 * @param high_k_fraction Optional output, n_radii entries: the fraction of
 * each diagonal element accumulated above half of k[n_points-1], as in
 * sif_delta_moments_pk. Pass NULL to skip.
 * @param deriv_variance Optional output, n_radii entries:
 *
 *     <(d delta / dS)^2>,   S = sigma^2(R)
 *
 * the variance of the smoothed field's derivative with respect to its own
 * variance, which is the mixed second derivative of this covariance evaluated
 * on its diagonal. Pass NULL to skip.
 *
 * @note Evaluated by differentiating the window under the integral, as
 * sif_delta_sigma_slope_pk does, rather than by differencing the matrix. That
 * is not a refinement: a finite difference of the diagonal converges only at
 * second order in the radius spacing, so the value it returns depends on how
 * finely the caller sampled `radii` -- by around 3% at 100 radii and 8% at 50.
 * Any consumer that treats this as a property of the field rather than of the
 * grid needs the form computed here.
 *
 * @note Carries units of 1 / sigma^2. The dimensionless combination is
 * 1 / (4 S <(d delta / dS)^2>), the squared correlation between the walk and
 * its own derivative: zero for uncorrelated steps, one for a walk whose slope
 * its value determines.
 *
 * @param opt SIF_DELTA_FILTER_TOP_HAT (default) or SIF_DELTA_FILTER_GAUSSIAN
 *
 * @return Newly allocated packed lower triangle of SIF_COV_SIZE(n_radii)
 * doubles, S(i, j) at SIF_COV_INDEX(i, j) for j <= i. Double rather than
 * sif_real, since a single-precision factorization of a realistic radius grid
 * reaches a non-positive pivot and fails. Released with sif_free_aligned, or
 * NULL on invalid input.
 */
SIF_NODISCARD double* sif_delta_covariance_pk(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real* sigma, sif_real* high_k_fraction,
  double* deriv_variance, sif_option opt);

#endif /* SIF_MODEL_DELTA_MOMENTS_H */
