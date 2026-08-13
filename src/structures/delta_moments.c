/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/delta_moments.h"

#include "results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

#define TAG "delta"

sif_delta_moments_t* sif__delta_moments_alloc(uint32_t n_radii, uint8_t order) {

  sif_delta_moments_t* m = calloc(1, sizeof(sif_delta_moments_t));
  if (!m) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta moments struct");
    return NULL;
  }

  m->n_radii = n_radii;
  m->order = order;
  m->n_moments = (uint8_t)(order + 1);

  const size_t total = (size_t)m->n_moments * n_radii;

  m->radii = sif_malloc_aligned((size_t)n_radii * sizeof(sif_real));
  m->sigma = sif_calloc_aligned(total, sizeof(sif_real));
  m->high_k_fraction = sif_calloc_aligned(total, sizeof(sif_real));
  m->offsets =
    sif_malloc_aligned(((size_t)m->n_moments + 1) * sizeof(uint32_t));

  if (!m->radii || !m->sigma || !m->high_k_fraction || !m->offsets) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta moments arrays");
    sif_delta_moments_free(m);
    return NULL;
  }

  for (uint8_t j = 0; j <= m->n_moments; j++) {
    m->offsets[j] = (uint32_t)j * n_radii;
  }

  return m;
}

void sif_delta_moments_free(sif_delta_moments_t* moments) {
  if (!moments)
    return;
  sif_free_aligned(moments->radii);
  sif_free_aligned(moments->sigma);
  sif_free_aligned(moments->high_k_fraction);
  sif_free_aligned(moments->offsets);
  free(moments);
}

const sif_real* sif_delta_moments_sigma(
  const sif_delta_moments_t* moments, uint8_t order) {

  if (!moments || !moments->sigma || order > moments->order)
    return NULL;
  return moments->sigma + moments->offsets[order];
}
