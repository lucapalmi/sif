#ifndef __SIF_MODEL_DELTAMOMENTS_H__
#define __SIF_MODEL_DELTAMOMENTS_H__

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/deltamoments.h"

/* Largest number of radii a covariance is built over. */
#define SIF_COV_MAX_RADII 4096

/*
 * @brief Index of S(i, j) in a packed lower triangle, for j <= i.
 *
 * Row-major over the lower triangle, in the ascending order of the radii.
 */
#define SIF_COV_INDEX(i, j) ((size_t)(i) * ((size_t)(i) + 1) / 2 + (size_t)(j))

/*
 * @brief Number of entries in a packed lower triangle of n rows.
 */
#define SIF_COV_SIZE(n) ((size_t)(n) * ((size_t)(n) + 1) / 2)

/*
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
NODISCARD sif_delta_moments_t* sif_delta_moments_pk(const real_t* k,
  const real_t* pk, uint32_t n_points, const real_t* radii, uint32_t n_radii,
  uint8_t order, sif_option_t opt);

/*
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
NODISCARD real_t* sif_sigma_slope_pk(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, sif_option_t opt);

/*
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
 * @param opt SIF_DELTA_FILTER_TOP_HAT (default) or SIF_DELTA_FILTER_GAUSSIAN
 *
 * @return Newly allocated packed lower triangle of SIF_COV_SIZE(n_radii)
 * doubles, S(i, j) at SIF_COV_INDEX(i, j) for j <= i. Double rather than
 * real_t, since a single-precision factorization of a realistic radius grid
 * reaches a non-positive pivot and fails. Released with sif_free_aligned, or
 * NULL on invalid input.
 */
NODISCARD double* sif_delta_covariance_pk(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, real_t* sigma,
  real_t* high_k_fraction, sif_option_t opt);

#endif /* __SIF_MODEL_DELTAMOMENTS_H__ */
