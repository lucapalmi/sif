/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file delta_moments.h
 * @brief Spectral moments of the density contrast, over a set of smoothing
 * radii.
 *
 * A moment set is one object holding sigma_0 through sigma_order at every
 * radius, rather than one object per order. That is deliberate: the orders are
 * only meaningful together. sigma_0 and sigma_2 describe the same field only
 * if they came from the same radii and the same smoothing window, and every
 * quantity built from them -- gamma, R_star, the BBKS densities -- reads
 * several orders at once. Bundling them makes a mismatched combination
 * impossible to express.
 *
 * It costs almost nothing to compute them together, either: the k-sweep
 * evaluates the window once and folds in the k^(2j) weight per order, so each
 * extra order adds a multiply and an accumulate per sample.
 *
 * **If you only need sigma_0, ask for order 0.** Both producers take the
 * highest order as a parameter, and a set of order 0 allocates exactly one
 * array. To read one order out of a larger set, use sif_delta_moments_sigma().
 */

#ifndef SIF_STRUCTURES_DELTA_MOMENTS_H
#define SIF_STRUCTURES_DELTA_MOMENTS_H

#include <stdint.h>

#include "sif/core/macros.h"

/** @brief Highest spectral moment order the estimators accept. */
#define SIF_MAX_MOMENT_ORDER 4

/**
 * @brief Spectral moments sigma_0 .. sigma_order, at several smoothing radii.
 *
 * sigma_j carries units of length^-j. Produced either by measuring a gridded
 * field (sif_delta_moments_grid()) or by integrating a model power spectrum
 * (sif_delta_moments_pk()); released with sif_delta_moments_free().
 */
typedef struct {
  uint32_t n_radii;
  /** Highest order computed. */
  uint8_t order;
  /** order + 1. */
  uint8_t n_moments;

  /** The n_radii smoothing radii. */
  sif_real* radii;

  /** n_moments * n_radii values, order-major: all radii for j = 0, then all
   * radii for j = 1, and so on. Read it through sif_delta_moments_sigma()
   * rather than indexing by hand.
   */
  sif_real* sigma;

  /** Same layout as #sigma. Fraction of each sum or integral coming from the
   * top half of the available k range: above half Nyquist for the grid
   * estimator, above half of k_max for the P(k) one. A large value means the
   * moment is dominated by the smallest scales the input resolves, and so is
   * set by the resolution rather than by the field.
   */
  sif_real* high_k_fraction;

  /** n_moments + 1 entries. offsets[j] is where order j's block of n_radii
   * values starts in #sigma and #high_k_fraction.
   */
  uint32_t* offsets;
} sif_delta_moments_t;

/**
 * @brief Release a moment set.
 * @param moments Set to free. NULL is accepted and ignored.
 */
void sif_delta_moments_free(sif_delta_moments_t* moments);

/**
 * @brief The n_radii values of sigma_order.
 *
 * @param moments The moment set.
 * @param order Order to read.
 * @return Borrowed pointer into @p moments, valid as long as it is and not to
 * be freed. NULL if that order was not computed.
 */
const sif_real* sif_delta_moments_sigma(
  const sif_delta_moments_t* moments, uint8_t order);

#endif /* SIF_STRUCTURES_DELTA_MOMENTS_H */
