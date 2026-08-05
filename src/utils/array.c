#include "sif/utils/array.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>

/* --- Allocation & Initialization --- */

real_t* sif_array_zeros(uint64_t size) {
  if (size == 0)
    return NULL;
  /* calloc_aligned already memsets to 0 extremely fast */
  return sif_calloc_aligned(size, sizeof(real_t));
}

real_t* sif_array_full(uint64_t size, real_t fill_value) {
  if (size == 0)
    return NULL;

  real_t* arr = sif_malloc_aligned(size * sizeof(real_t));
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

real_t* sif_array_ones(uint64_t size) { return sif_array_full(size, 1.0f); }

/* --- Ranges & Sequences --- */

real_t* sif_array_linspace(real_t start, real_t stop, uint64_t num) {
  if (num == 0)
    return NULL;

  real_t* arr = sif_malloc_aligned(num * sizeof(real_t));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate linspace array");
    return NULL;
  }

  if (num == 1) {
    arr[0] = start;
    return arr;
  }

  real_t step = (stop - start) / (real_t)(num - 1);

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    arr[i] = start + ((real_t)i * step);
  }

  /* Force exact boundary on the last element to prevent floating point drift */
  arr[num - 1] = stop;

  return arr;
}

real_t* sif_array_arange(
  real_t start, real_t stop, real_t step, uint64_t* out_size) {
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
  uint64_t size = (uint64_t)REAL_CEIL((stop - start) / step);
  if (out_size)
    *out_size = size;

  real_t* arr = sif_malloc_aligned(size * sizeof(real_t));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate arange array");
    return NULL;
  }

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = start + ((real_t)i * step);
  }

  return arr;
}

real_t* sif_array_logspace(
  real_t start, real_t stop, uint64_t num, real_t base) {
  if (num == 0)
    return NULL;

  real_t* arr = sif_malloc_aligned(num * sizeof(real_t));
  if (!arr) {
    SIF_LOG_ERROR("array", "failed to allocate logspace array");
    return NULL;
  }

  if (num == 1) {
    arr[0] = REAL_POW(base, start);
    return arr;
  }

  real_t step = (stop - start) / (real_t)(num - 1);

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    /* Calculate the linear exponent, then raise to the base */
    real_t exponent = start + ((real_t)i * step);
    arr[i] = REAL_POW(base, exponent);
  }

  /* Force exact boundary on the last element to prevent floating point drift */
  arr[num - 1] = REAL_POW(base, stop);

  return arr;
}

/* --- Basic Reductions --- */

real_t sif_array_sum(const real_t* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  real_t total = 0.0f;
#pragma omp parallel for simd reduction(+ : total)
  for (uint64_t i = 0; i < size; i++) {
    total += arr[i];
  }
  return total;
}

real_t sif_array_min(const real_t* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  real_t min_val = arr[0];
#pragma omp parallel for simd reduction(min : min_val)
  for (uint64_t i = 1; i < size; i++) {
    if (arr[i] < min_val)
      min_val = arr[i];
  }
  return min_val;
}

real_t sif_array_max(const real_t* arr, uint64_t size) {
  if (!arr || size == 0)
    return 0.0f;

  real_t max_val = arr[0];
#pragma omp parallel for simd reduction(max : max_val)
  for (uint64_t i = 1; i < size; i++) {
    if (arr[i] > max_val)
      max_val = arr[i];
  }
  return max_val;
}