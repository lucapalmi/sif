/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file delta_moments.h
 * @brief Measuring the spectral moments of a gridded density field.
 *
 * The counterpart to sif_delta_moments_pk(), which evaluates the same moments
 * from a model spectrum: measuring one field and modelling another is how a
 * measurement is checked against theory, so the two deliberately produce the
 * same container.
 */

#ifndef SIF_MEASURE_DELTA_MOMENTS_H
#define SIF_MEASURE_DELTA_MOMENTS_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/delta_moments.h"
#include "sif/structures/grid.h"

/**
 * @brief Measure the spectral moments of a gridded density field.
 *
 * @param grid Density contrast field, from sif_grid_to_density_contrast().
 * Read only.
 * @param radii Smoothing radii, each at least two cell lengths and at most
 * half the box; see sif_delta_distribution_grid() on why smaller is rejected.
 * @param n_radii Number of radii.
 * @param order Highest moment order, at most #SIF_MAX_MOMENT_ORDER. Each extra
 * order costs a multiply and an accumulate per k sample, so ask for what you
 * need and no less: pass 0 if only sigma_0 is wanted.
 * @param n_tracers Tracer count behind the grid, used to subtract the shot
 * noise. Pass 0 to leave it in. Assumes unweighted tracers.
 * @param seed Seed for the phase shuffle; unused when not shuffling. Shared
 * with sif_delta_distribution_grid() -- see there.
 * @param opt SIF_DELTA_SHUFFLE_*, SIF_DELTA_FILTER_*,
 * SIF_DELTA_KEEP_CIC_WINDOW.
 * @return Newly allocated moment set, released with
 * sif_delta_moments_free(), or NULL on invalid input or failure.
 */
SIF_NODISCARD sif_delta_moments_t* sif_delta_moments_grid(
  const sif_grid_t* grid, const sif_real* radii, uint32_t n_radii,
  uint8_t order, uint64_t n_tracers, uint64_t seed, sif_option opt);

#endif /* SIF_MEASURE_DELTA_MOMENTS_H */
