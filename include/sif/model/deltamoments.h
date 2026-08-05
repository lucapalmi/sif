#ifndef __SIF_MODEL_DELTAMOMENTS_H__
#define __SIF_MODEL_DELTAMOMENTS_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/deltamoments.h"

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

#endif /* __SIF_MODEL_DELTAMOMENTS_H__ */
