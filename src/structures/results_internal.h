/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF__STRUCTURES_RESULTS_INTERNAL_H
#define SIF__STRUCTURES_RESULTS_INTERNAL_H

/**
 * @file results_internal.h
 * @brief Allocators for the result containers in structures/.
 *
 * Not a public header. The containers are only meaningful once an estimator
 * has filled them in, so the public API exposes the struct and its free and
 * nothing else. But several of them have more than one producer -- a moment
 * set can come from a grid or from a model spectrum -- sitting in different
 * modules, and duplicating the allocator in each is how the two drift apart
 * when a field is added. They live here instead, next to the free that has to
 * match them.
 */

#include "sif/structures/delta_distribution.h"
#include "sif/structures/delta_moments.h"
#include "sif/structures/size_function.h"

SIF_NODISCARD sif_delta_moments_t* sif__delta_moments_alloc(
  uint32_t n_radii, uint8_t order);

SIF_NODISCARD sif_delta_distribution_t* sif__delta_distribution_alloc(
  uint32_t n_radii, uint32_t n_bins);

SIF_NODISCARD sif_size_function_t* sif__size_function_alloc(uint32_t n_bins);

/**
 * @brief Reconstructs bin edges from the radii a model was evaluated at.
 *
 * A measured size function is a histogram and its edges are the grid it was
 * binned on. A model is evaluated pointwise, so there is no such grid: the
 * radii are centres and the edges are geometric midpoints, present so the
 * container is complete and plottable rather than because the model
 * integrated over them.
 *
 * @param centers n entries, strictly positive
 * @param n Number of centres, at least 2
 * @param edges Output, n + 1 entries
 */
void sif__edges_from_centers(
  const sif_real* centers, uint32_t n, sif_real* edges);

#endif /* SIF__STRUCTURES_RESULTS_INTERNAL_H */
