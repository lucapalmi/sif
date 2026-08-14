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

/*
 * The moment set is one array per quantity, indexed [order][radius] through an
 * offset table rather than as an array of per-order arrays.
 *
 * Two reasons. The estimators fill it one radius at a time across every order
 * -- a single pass over the spectrum yields all of them -- so the orders want
 * to be reachable by arithmetic rather than by chasing a pointer per order.
 * And the whole set travels to the Python bindings as one buffer per quantity,
 * which a jagged layout would have to flatten anyway.
 */
sif_delta_moments_t* sif__delta_moments_alloc(uint32_t n_radii, uint8_t order) {

  /* calloc for the struct, so every pointer is NULL before the first failure
   * path can reach the free below. */
  sif_delta_moments_t* m = calloc(1, sizeof(sif_delta_moments_t));
  if (!m) {
    SIF_LOG_ERROR(TAG, "failed to allocate the delta moments struct");
    return NULL;
  }

  m->n_radii = n_radii;
  m->order = order;
  m->n_moments = (uint8_t)(order + 1);

  const size_t total = (size_t)m->n_moments * n_radii;

  /* sigma and high_k_fraction are zeroed rather than merely allocated: an
   * estimator that fails partway through leaves the untouched orders reading
   * as zero, which every consumer already treats as "no measurement", instead
   * of as whatever the allocator returned. */
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

  /* One entry past the last order, so a consumer can bracket order j as
   * [offsets[j], offsets[j + 1]) without special-casing the end. */
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

  /* The order check is what keeps the offset lookup in range, so it has to
   * come first and cannot be an assert: callers ask for sigma_2 from a set
   * that may only carry sigma_0, and getting NULL back is how they find out. */
  if (!moments || !moments->sigma || order > moments->order)
    return NULL;

  return moments->sigma + moments->offsets[order];
}

#undef TAG
