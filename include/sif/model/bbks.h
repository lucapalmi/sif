#ifndef __SIF_MODEL_BBKS_H__
#define __SIF_MODEL_BBKS_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/deltamoments.h"
#include "sif/structures/sizefunction.h"

/*
 * @brief The BBKS G function.
 *
 * @param gamma Spectral parameter, strictly inside (0, 1)
 * @param w Peak-height variable, gamma * nu
 * @param opt SIF_BBKS_G_FITTED (BBKS 1986 eq. 4.4) or SIF_BBKS_G_EXACT
 * (Wu 2020 eqs. 18-19, quadratured)
 *
 * @return G(gamma, w), or 0 for gamma outside (0, 1).
 */
real_t sif_g_bbks(real_t gamma, real_t w, sif_option_t opt);

/*
 * @brief Differential number density of maxima of a Gaussian field, per unit
 * volume per unit nu (BBKS 1986 eq. 4.3).
 *
 * The three input arrays are parallel and read element-wise. Results are
 * clamped at zero.
 *
 * @param nu Peak heights in units of sigma_0
 * @param gamma Spectral parameter per entry, each strictly inside (0, 1)
 * @param r_star Coherence scale per entry, each strictly positive
 * @param size Length of all three arrays
 * @param opt SIF_BBKS_G_FITTED or SIF_BBKS_G_EXACT
 *
 * @return Newly allocated array of `size` densities, released with
 * sif_free_aligned, or NULL on invalid input.
 */
NODISCARD real_t* sif_differential_number_density_bbks(const real_t* nu,
  const real_t* gamma, const real_t* r_star, uint32_t size, sif_option_t opt);

/*
 * @brief Number density of Gaussian-field maxima above a density threshold,
 * one value per smoothing radius.
 *
 * The threshold is converted per radius as nu_t = |delta| / sigma_0(R), then
 * integrated over nu. Results are clamped at zero.
 *
 * @note Only the magnitude of `delta` is used. BBKS counts maxima, and for a
 * Gaussian field the density of minima below -|delta| equals the density of
 * maxima above +|delta|, so the same integral serves a void threshold and a
 * peak threshold of the same depth.
 *
 * @note Carries a factor exp(-nu_t^2/2), so a single-precision build
 * underflows to zero around nu_t ~ 12.
 *
 * @param delta Density-contrast threshold
 * @param moments Moment set with order >= 2, measured or modelled
 * @param opt SIF_BBKS_G_FITTED or SIF_BBKS_G_EXACT
 *
 * @return Newly allocated array of n_radii densities, released with
 * sif_free_aligned, or NULL on invalid input.
 */
NODISCARD real_t* sif_cumulative_number_density_bbks(
  real_t delta, const sif_delta_moments_t* moments, sif_option_t opt);

/*
 * @brief Number density of Gaussian-field structures per unit radius, one
 * value per smoothing radius: the size function implied by the cumulative
 * density above a threshold.
 *
 * `vsf` holds -dC/dlnR by default, or -dC/dR with SIF_VSF_BIN_LINEAR, matching
 * the convention of the measured size function; `options` records which. The
 * sign is chosen so the result is a positive number density.
 *
 * `r_centers` is `moments->radii`. The model is evaluated pointwise rather
 * than binned, so `r_edges` is reconstructed as geometric midpoints and
 * `counts` and `err` stay zero.
 *
 * Evaluated as a finite difference over the radii, so its accuracy is set by
 * how finely those are sampled. Radii where the cumulative density underflowed,
 * and their immediate neighbours, hold zero.
 *
 * @param delta Density-contrast threshold; only its magnitude is used
 * @param moments Moment set with order >= 2 and strictly increasing radii, at
 * least two of them
 * @param opt SIF_VSF_BIN_LN (default) or SIF_VSF_BIN_LINEAR for the units,
 * combined with SIF_BBKS_G_FITTED or SIF_BBKS_G_EXACT
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
NODISCARD sif_size_function_t* sif_size_function_bbks(
  real_t delta, const sif_delta_moments_t* moments, sif_option_t opt);

#endif /* __SIF_MODEL_BBKS_H__ */
