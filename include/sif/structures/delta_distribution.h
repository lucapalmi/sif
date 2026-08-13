/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file delta_distribution.h
 * @brief PDF of the smoothed density contrast, one histogram per smoothing
 * radius.
 *
 * Where the spectral moments summarise the field by its variance and its
 * derivatives, this keeps the whole one-point distribution -- which is where
 * the non-Gaussian information lives, and so what a phase-randomized surrogate
 * is compared against.
 */

#ifndef SIF_STRUCTURES_DELTA_DISTRIBUTION_H
#define SIF_STRUCTURES_DELTA_DISTRIBUTION_H

#include <stdint.h>

#include "sif/core/macros.h"

/**
 * @brief PDF of the smoothed density contrast, one row per smoothing radius.
 *
 * Produced by sif_delta_distribution_grid(), released with
 * sif_delta_distribution_free(). Every row shares one set of bin edges, so
 * rows can be compared across radii directly.
 */
typedef struct {
  uint32_t n_radii;
  uint32_t n_bins;

  /** Grid cells behind each row. Samples outside the histogram's range are
   *  counted here but not binned, so a row integrates to the fraction of
   *  samples that fell in range rather than to 1. Compare the two to find out
   *  how much of the distribution the bounds are cutting off. */
  uint64_t n_samples;

  sif_real* radii;         /**< n_radii smoothing radii. */
  sif_real* delta_edges;   /**< n_bins + 1 bin edges, shared by every row. */
  sif_real* distributions; /**< n_radii * n_bins values, row-major. */
} sif_delta_distribution_t;

/**
 * @brief Release a distribution set.
 * @param dist Set to free. NULL is accepted and ignored.
 */
void sif_delta_distribution_free(sif_delta_distribution_t* dist);

#endif /* SIF_STRUCTURES_DELTA_DISTRIBUTION_H */
