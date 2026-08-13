/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file delta_distribution.h
 * @brief Measuring the one-point PDF of the smoothed density contrast.
 *
 * Smooths the gridded field at each radius in Fourier space and histograms the
 * cell values. Pair it with a phase-randomized surrogate -- same seed, same
 * spectrum, randomized phases -- and the difference between the two PDFs is
 * the non-Gaussian information the field carries.
 */

#ifndef SIF_MEASURE_DELTA_DISTRIBUTION_H
#define SIF_MEASURE_DELTA_DISTRIBUTION_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/delta_distribution.h"
#include "sif/structures/grid.h"

/**
 * @brief Measure the PDF of the smoothed density contrast on a grid.
 *
 * @param grid Density contrast field, from sif_grid_to_density_contrast().
 * Read only.
 * @param radii Smoothing radii, each at least two cell lengths and at most
 * half the box. Radii below that resolve grid artefacts rather than field, so
 * they are rejected instead of returning a plausible-looking answer.
 * @param n_radii Number of radii.
 * @param n_bins Histogram bins per radius.
 * @param delta_bounds Histogram range, {min, max}, strictly increasing.
 * Samples outside it are counted in the total but not binned, so a row
 * integrates to the fraction that fell in range.
 * @param seed Seed for the phase shuffle; unused when not shuffling. The same
 * seed and grid size give the same surrogate here and in
 * sif_delta_moments_grid(), which is what lets the two describe one field.
 * @param opt SIF_DELTA_SHUFFLE_*, SIF_DELTA_FILTER_*,
 * SIF_DELTA_KEEP_CIC_WINDOW.
 * @return Newly allocated distribution, released with
 * sif_delta_distribution_free(), or NULL on invalid input or failure.
 */
SIF_NODISCARD sif_delta_distribution_t* sif_delta_distribution_grid(
  const sif_grid_t* grid, const sif_real* radii, uint32_t n_radii,
  uint32_t n_bins, const sif_real delta_bounds[2], uint64_t seed,
  sif_option opt);

#endif /* SIF_MEASURE_DELTA_DISTRIBUTION_H */
