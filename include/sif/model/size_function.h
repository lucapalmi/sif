/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file size_function.h
 * @brief Model void size functions, and the spherical-evolution mapping they
 * are expressed in.
 *
 * The models predict a number density per radius bin in the same container the
 * measurement fills, so a prediction and a catalogue can be plotted and
 * compared directly.
 */

#ifndef SIF_MODEL_SIZE_FUNCTION_H
#define SIF_MODEL_SIZE_FUNCTION_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/size_function.h"

/* Bernardeau (1994) fit constant linking the linear and non-linear void
 * density contrasts. */
#define SIF_SPHERICAL_EXPANSION_C 1.594

/**
 * @brief Non-linear density contrast a void of a given linear contrast
 * evolves to.
 *
 * Under SIF_SPHERICAL_B94, the Bernardeau (1994) fit quoted by Jennings, Li &
 * Hu (2013) eq. (A4): delta_NL = (1 - delta_L/c)^-c - 1. Under
 * SIF_SPHERICAL_EXACT, the Einstein-de Sitter expansion solution, by
 * root-find. The two agree to better than 0.1% on the expansion factor.
 *
 * @note Only the expanding branch is implemented; a non-negative linear
 * contrast collapses rather than expands.
 *
 * @param delta_linear Linear density contrast, strictly negative
 * @param opt SIF_SPHERICAL_B94 (default) or SIF_SPHERICAL_EXACT
 *
 * @return The non-linear contrast, strictly between -1 and 0, or 0 for input
 * outside that branch. A valid result is never 0, so the two are
 * distinguishable.
 */
sif_real sif_spherical_map_nonlinear(sif_real delta_linear, sif_option opt);

/**
 * @brief Linear density contrast that evolves into a given non-linear one,
 * the inverse of sif_spherical_map_nonlinear.
 *
 * Wanted when a barrier is quoted as an observed underdensity rather than as a
 * linear threshold.
 *
 * @param delta_nonlinear Non-linear density contrast, strictly between -1 and 0
 * @param opt SIF_SPHERICAL_B94 (default) or SIF_SPHERICAL_EXACT
 *
 * @return The linear contrast, strictly negative, or 0 for input outside that
 * range.
 */
sif_real sif_spherical_map_linear(sif_real delta_nonlinear, sif_option opt);

/**
 * @brief Sheth & van de Weygaert void size function.
 *
 * Jennings, Li & Hu (2013) eqs. (9) and (10): the number-conserving mapping,
 * in which the linear abundance is evaluated at r_L = r / F and reported
 * against r without rescaling its amplitude.
 *
 * @note Number conservation makes this model exceed a void volume fraction of
 * one at large radii, which is why Vdn exists. It is reproduced here as the
 * reference, not as a recommendation.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Linear power spectrum sampled at k
 * @param n_points Length of k and pk, at least 2
 * @param radii Eulerian (observed) void radii, strictly positive
 * @param n_radii Number of radii
 * @param delta_v Linear void barrier, strictly negative
 * @param delta_c Collapse barrier, strictly positive
 * @param opt SIF_VSF_BIN_LN (default) or SIF_VSF_BIN_LINEAR for the units,
 * combined with SIF_SPHERICAL_B94 (default) or SIF_SPHERICAL_EXACT for the
 * mapping used to expand the radii. The window is always a top-hat, which is
 * what the barriers are calibrated against.
 *
 * @note The Lagrangian-to-Eulerian expansion factor is not an argument: it is
 * (1 + delta_NL)^(-1/3), which delta_v already determines.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_svdw(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_v, sif_real delta_c, sif_option opt);

/**
 * @brief Volume-conserving (Vdn) void size function.
 *
 * Jennings, Li & Hu (2013) eq. (12). Identical to SvdW except that the
 * abundance is divided by the Eulerian volume rather than the Lagrangian one,
 * which conserves volume instead of number and keeps the void volume fraction
 * below unity.
 *
 * Parameters are as sif_size_function_svdw.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_vdn(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_v, sif_real delta_c, sif_option opt);

#endif /* SIF_MODEL_SIZE_FUNCTION_H */
