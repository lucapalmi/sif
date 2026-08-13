/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/array.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>

/* --- Allocation & Initialization --- */

sif_real* sif_array_zeros(uint64_t size) {
  if (size == 0)
    return NULL;
  /* calloc_aligned already memsets to 0 extremely fast */
  return sif_calloc_aligned(size, sizeof(sif_real));
}

sif_real* sif_array_full(uint64_t size, sif_real fill_value) {
  if (size == 0)
    return NULL;

  sif_real* arr = sif_malloc_aligned(size * sizeof(sif_real));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate array of size %llu",
      (unsigned long long)size);
    return NULL;
  }

/* SIMD vectorized fill */
#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = fill_value;
  }
  return arr;
}

sif_real* sif_array_ones(uint64_t size) { return sif_array_full(size, 1.0f); }

/* --- Ranges & Sequences --- */

sif_real* sif_array_linspace(sif_real start, sif_real stop, uint64_t num) {
  if (num == 0)
    return NULL;

  sif_real* arr = sif_malloc_aligned(num * sizeof(sif_real));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate linspace array");
    return NULL;
  }

  if (num == 1) {
    arr[0] = start;
    return arr;
  }

  sif_real step = (stop - start) / (sif_real)(num - 1);

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    arr[i] = start + ((sif_real)i * step);
  }

  /* Force exact boundary on the last element to prevent floating point drift */
  arr[num - 1] = stop;

  return arr;
}

sif_real* sif_array_arange(
  sif_real start, sif_real stop, sif_real step, uint64_t* out_size) {
  if (out_size)
    *out_size = 0;

  if (step == 0.0f) {
    SIF_LOG_ERROR("array", "arange step size cannot be zero");
    return NULL;
  }
  if ((step > 0.0f && start >= stop) || (step < 0.0f && start <= stop)) {
    return NULL; /* Empty range */
  }

  /* Calculate required size */
  uint64_t size = (uint64_t)SIF_REAL_CEIL((stop - start) / step);
  if (out_size)
    *out_size = size;

  sif_real* arr = sif_malloc_aligned(size * sizeof(sif_real));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate arange array");
    return NULL;
  }

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = start + ((sif_real)i * step);
  }

  return arr;
}

sif_real* sif_array_logspace(
  sif_real start, sif_real stop, uint64_t num, sif_real base) {
  if (num == 0)
    return NULL;

  sif_real* arr = sif_malloc_aligned(num * sizeof(sif_real));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate logspace array");
    return NULL;
  }

  if (num == 1) {
    arr[0] = SIF_REAL_POW(base, start);
    return arr;
  }

  sif_real step = (stop - start) / (sif_real)(num - 1);

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    /* Calculate the linear exponent, then raise to the base */
    sif_real exponent = start + ((sif_real)i * step);
    arr[i] = SIF_REAL_POW(base, exponent);
  }

  /* Force exact boundary on the last element to prevent floating point drift */
  arr[num - 1] = SIF_REAL_POW(base, stop);

  return arr;
}

/* --- Basic Reductions --- */

sif_real sif_array_sum(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  sif_real total = 0.0f;
#pragma omp parallel for simd reduction(+ : total)
  for (uint64_t i = 0; i < size; i++) {
    total += arr[i];
  }
  return total;
}

sif_real sif_array_min(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  sif_real min_val = arr[0];
#pragma omp parallel for simd reduction(min : min_val)
  for (uint64_t i = 1; i < size; i++) {
    if (arr[i] < min_val)
      min_val = arr[i];
  }
  return min_val;
}

sif_real sif_array_max(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  sif_real max_val = arr[0];
#pragma omp parallel for simd reduction(max : max_val)
  for (uint64_t i = 1; i < size; i++) {
    if (arr[i] > max_val)
      max_val = arr[i];
  }
  return max_val;
}