/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/delta_distribution.h"

#include "results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <stdlib.h>

#define TAG "delta"

/*
 * One PDF per radius, laid out as n_radii rows of n_bins in a single block.
 *
 * The edges are shared by every row rather than stored per radius: the
 * histogram range is a property of the measurement, not of the scale, and
 * comparing a PDF at 5 Mpc against one at 20 requires that they were binned
 * identically. Keeping one edge array makes that structural instead of a
 * convention the estimators have to remember.
 */
sif_delta_distribution_t* sif__delta_distribution_alloc(
  uint32_t n_radii, uint32_t n_bins) {

  /* calloc for the struct, so the free below is safe from the first failure
   * onwards. */
  sif_delta_distribution_t* dist = calloc(1, sizeof(sif_delta_distribution_t));
  if (!dist) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta distribution struct");
    return NULL;
  }

  dist->n_radii = n_radii;
  dist->n_bins = n_bins;

  /* n_samples is filled by the estimator with the cell count it histogrammed,
   * not with the number that landed in range: a row integrates to the fraction
   * that fell inside delta_bounds, and recovering the outliers means knowing
   * the denominator. */
  dist->n_samples = 0;

  dist->radii = sif_malloc_aligned((size_t)n_radii * sizeof(sif_real));
  dist->delta_edges =
    sif_malloc_aligned(((size_t)n_bins + 1) * sizeof(sif_real));
  dist->distributions =
    sif_calloc_aligned((size_t)n_radii * n_bins, sizeof(sif_real));

  if (!dist->radii || !dist->delta_edges || !dist->distributions) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta distribution arrays");
    sif_delta_distribution_free(dist);
    return NULL;
  }

  return dist;
}

/* Accepts a partially built object, which is what lets the allocator above
 * clean up after itself on any failure path rather than unwinding by hand. */
void sif_delta_distribution_free(sif_delta_distribution_t* dist) {
  if (!dist)
    return;
  sif_free_aligned(dist->radii);
  sif_free_aligned(dist->delta_edges);
  sif_free_aligned(dist->distributions);
  free(dist);
}

#undef TAG
