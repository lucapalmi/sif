/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file bbks.h
 * @brief Peak statistics of a Gaussian random field, after Bardeen, Bond,
 * Kaiser & Szalay (1986).
 *
 * The number density of maxima depends on the field only through its spectral
 * moments, which is what makes it the natural prediction to hold a
 * phase-randomized surrogate against: everything BBKS knows about a field
 * survives phase randomization, so any disagreement with a directly counted
 * catalogue is phase information.
 *
 * Follows the presentation of Wu, Phys. Dark Universe 30 (2020) 100654,
 * eqs. (17)-(19).
 */

#ifndef SIF_MODEL_BBKS_H
#define SIF_MODEL_BBKS_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/delta_moments.h"
#include "sif/structures/size_function.h"

/**
 * @brief Spectral parameter gamma = sigma_1^2 / (sigma_0 sigma_2), one value
 * per smoothing radius.
 *
 * Measures how narrow the field's power is in k: it approaches 1 for a field
 * dominated by a single scale and falls towards 0 for a broad spectrum. BBKS
 * expresses the peak density through gamma and R_star rather than through the
 * moments directly.
 *
 * @param moments Moment set with order >= 2.
 * @return Newly allocated array of n_radii values, released with
 * sif_free_aligned(), or NULL if the moments do not reach order 2. Radii where
 * sigma_0 or sigma_2 vanish are left at zero and warned about.
 */
SIF_NODISCARD sif_real* sif_bbks_gamma(const sif_delta_moments_t* moments);

/**
 * @brief Coherence scale R_star = sqrt(3) sigma_1 / sigma_2, one value per
 * smoothing radius.
 *
 * The characteristic separation between peaks of the smoothed field, and the
 * length that sets the normalization of the BBKS number density.
 *
 * @param moments Moment set with order >= 2.
 * @return Newly allocated array of n_radii lengths, released with
 * sif_free_aligned(), or NULL if the moments do not reach order 2. Radii where
 * sigma_2 vanishes are left at zero and warned about.
 */
SIF_NODISCARD sif_real* sif_bbks_r_star(const sif_delta_moments_t* moments);

/**
 * @brief The BBKS G function.
 *
 * Reports through @p out rather than through the return value, unlike the
 * other scalar entry points in the library. G is legitimately zero wherever the
 * exact form underflows -- which it does at strongly negative @p w -- and the
 * fitted form dips slightly below zero in the same region, so there is no value
 * left over to mean "invalid input". Contrast sif_spherical_map_nonlinear(),
 * which keeps a plain return because its result is strictly negative and 0 is
 * therefore free to be a sentinel.
 *
 * @param gamma Spectral parameter, strictly inside (0, 1). In a normal pipeline
 * this comes from sif_bbks_gamma(), which cannot produce anything else.
 * @param w Peak-height variable, gamma * nu. Unrestricted.
 * @param opt SIF_BBKS_G_FITTED (BBKS 1986 eq. 4.4) or SIF_BBKS_G_EXACT
 * (Wu 2020 eqs. 18-19, quadratured)
 * @param out Written with G(gamma, w) on success, untouched otherwise.
 *
 * @return SIF_OK, or SIF_ERR_INVALID for a NULL @p out or a gamma outside
 * (0, 1).
 */
SIF_NODISCARD int sif_bbks_g(
  sif_real gamma, sif_real w, sif_option opt, sif_real* out);

/**
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
SIF_NODISCARD sif_real* sif_bbks_number_density_differential(const sif_real* nu,
  const sif_real* gamma, const sif_real* r_star, uint32_t size, sif_option opt);

/**
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
SIF_NODISCARD sif_real* sif_bbks_number_density_cumulative(
  sif_real delta, const sif_delta_moments_t* moments, sif_option opt);

/**
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
SIF_NODISCARD sif_size_function_t* sif_size_function_bbks(
  sif_real delta, const sif_delta_moments_t* moments, sif_option opt);

#endif /* SIF_MODEL_BBKS_H */
