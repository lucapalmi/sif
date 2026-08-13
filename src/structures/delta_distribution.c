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

sif_delta_distribution_t* sif__delta_distribution_alloc(
  uint32_t n_radii, uint32_t n_bins) {

  sif_delta_distribution_t* dist = calloc(1, sizeof(sif_delta_distribution_t));
  if (!dist) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta distribution struct");
    return NULL;
  }

  dist->n_radii = n_radii;
  dist->n_bins = n_bins;
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

void sif_delta_distribution_free(sif_delta_distribution_t* dist) {
  if (!dist)
    return;
  sif_free_aligned(dist->radii);
  sif_free_aligned(dist->delta_edges);
  sif_free_aligned(dist->distributions);
  free(dist);
}
