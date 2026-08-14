/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/array.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdint.h>

/* Blocks every reduction below splits its input into. Fixed, and deliberately
 * not derived from the thread count: the partition of the array -- and so the
 * order the partial results are combined in -- has to be a property of the
 * array alone, or the same call answers differently on a machine with a
 * different core count. */
#define REDUCE_BLOCKS 64

/* --- allocation --- */

/*
 * Allocate @p size elements, or NULL with a diagnostic.
 *
 * The element count is a uint64_t across this API, which on any target can
 * name more elements than the address space holds, so the byte count has to be
 * checked before it is formed rather than after it has already wrapped.
 */
static sif_real* alloc_elements(uint64_t size, const char* what) {
  if (size > SIZE_MAX / sizeof(sif_real)) {
    SIF_LOG_ERROR("array",
      "%s of %" PRIu64 " elements exceeds addressable "
      "memory",
      what, size);
    return NULL;
  }

  sif_real* arr = sif_malloc_aligned((size_t)size * sizeof(sif_real));
  if (!arr) {
    SIF_LOG_ERROR(
      "array", "failed to allocate %s of %" PRIu64 " elements", what, size);
  }

  return arr;
}

/* --- construction --- */

sif_real* sif_array_zeros(uint64_t size) {
  if (size == 0)
    return NULL;

  /* calloc_aligned rather than a fill loop: it is a memset over a
   * cache-aligned block, which the C library already does at the memory
   * bandwidth, and it checks the multiplication for overflow on the way. */
  if (size > SIZE_MAX / sizeof(sif_real)) {
    SIF_LOG_ERROR(
      "array", "%" PRIu64 " elements exceeds addressable memory", size);
    return NULL;
  }

  return sif_calloc_aligned((size_t)size, sizeof(sif_real));
}

sif_real* sif_array_full(uint64_t size, sif_real fill_value) {
  if (size == 0)
    return NULL;

  sif_real* arr = alloc_elements(size, "array");
  if (!arr)
    return NULL;

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = fill_value;
  }

  return arr;
}

sif_real* sif_array_ones(uint64_t size) {
  return sif_array_full(size, (sif_real)1.0);
}

/* --- ranges --- */

sif_real* sif_array_linspace(sif_real start, sif_real stop, uint64_t num) {
  if (num == 0)
    return NULL;

  sif_real* arr = alloc_elements(num, "linspace array");
  if (!arr)
    return NULL;

  if (num == 1) {
    arr[0] = start;
    return arr;
  }

  const sif_real step = (stop - start) / (sif_real)(num - 1);

  /* start + i * step, not a running total: the multiplication carries one
   * rounding error whatever i is, while accumulation carries i of them. */
#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    arr[i] = start + ((sif_real)i * step);
  }

  /* The endpoint is assigned rather than computed. Callers use these arrays as
   * bin edges and compare against the last one, so it has to be exactly the
   * value asked for and not one ulp below it. */
  arr[num - 1] = stop;

  return arr;
}

sif_real* sif_array_arange(
  sif_real start, sif_real stop, sif_real step, uint64_t* out_size) {
  if (out_size)
    *out_size = 0;

  if (step == (sif_real)0.0) {
    SIF_LOG_ERROR("array", "arange step size cannot be zero");
    return NULL;
  }

  if ((step > (sif_real)0.0 && start >= stop) ||
      (step < (sif_real)0.0 && start <= stop)) {
    return NULL; /* empty range, and not an error: numpy returns [] here */
  }

  /* Worked out in double whatever sif_real is, and bounded before the cast:
   * a small step makes this quotient enormous, and converting a floating value
   * that does not fit into a uint64_t is undefined rather than merely wrong.
   * 2^53 is where a double stops representing consecutive integers at all,
   * which is orders of magnitude past anything allocatable. */
  const double count = ceil(((double)stop - (double)start) / (double)step);

  if (!(count >= 1.0) || count > 9007199254740992.0) {
    SIF_LOG_ERROR("array",
      "arange over [%g, %g) by %g yields no usable "
      "element count",
      (double)start, (double)stop, (double)step);
    return NULL;
  }

  const uint64_t size = (uint64_t)count;

  sif_real* arr = alloc_elements(size, "arange array");
  if (!arr)
    return NULL;

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    arr[i] = start + ((sif_real)i * step);
  }

  /* Reported only now that there is something to report. Set earlier, a
   * failed allocation would leave the caller holding NULL and a count it has
   * every reason to trust. */
  if (out_size)
    *out_size = size;

  return arr;
}

sif_real* sif_array_logspace(
  sif_real start, sif_real stop, uint64_t num, sif_real base) {
  if (num == 0)
    return NULL;

  sif_real* arr = alloc_elements(num, "logspace array");
  if (!arr)
    return NULL;

  if (num == 1) {
    arr[0] = SIF_REAL_POW(base, start);
    return arr;
  }

  const sif_real step = (stop - start) / (sif_real)(num - 1);

  /* The exponents are what is evenly spaced; spacing the values themselves
   * and taking logs afterwards would put the samples in the wrong places. */
#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < num; i++) {
    const sif_real exponent = start + ((sif_real)i * step);
    arr[i] = SIF_REAL_POW(base, exponent);
  }

  arr[num - 1] = SIF_REAL_POW(base, stop);

  return arr;
}

/* --- reductions --- */

/*
 * Half-open bounds of block @p b of #REDUCE_BLOCKS over [0, @p size).
 *
 * The remainder is spread one element per block instead of being dumped on the
 * last one, so no block runs more than a single element longer than another
 * and the work stays balanced for any size.
 */
static void block_bounds(uint64_t size, int b, uint64_t* lo, uint64_t* hi) {
  const uint64_t chunk = size / REDUCE_BLOCKS;
  const uint64_t rem = size % REDUCE_BLOCKS;
  const uint64_t ub = (uint64_t)b;

  *lo = ub * chunk + (ub < rem ? ub : rem);
  *hi = *lo + chunk + (ub < rem ? 1 : 0);
}

sif_real sif_array_sum(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return (sif_real)0.0;

  double partials[REDUCE_BLOCKS];

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(size, b, &lo, &hi);

    /* Accumulated in double however sif_real is configured. A float running
     * total stops growing once it exceeds the next term by 2^24 -- around ten
     * million elements of similar magnitude -- and silently returns the
     * partial sum it reached. */
    double acc = 0.0;
    for (uint64_t i = lo; i < hi; i++)
      acc += (double)arr[i];

    partials[b] = acc;
  }

  /* Combined serially in block order, so the answer depends only on the input.
   * (The release build's -ffast-math may still reassociate the block loops
   * above, but it does so identically in every run of the same binary.) */
  double total = 0.0;
  for (int b = 0; b < REDUCE_BLOCKS; b++)
    total += partials[b];

  return (sif_real)total;
}

sif_real sif_array_min(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return (sif_real)0.0;

  sif_real partials[REDUCE_BLOCKS];

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(size, b, &lo, &hi);

    /* Seeded with the identity, not with arr[lo]. Every comparison against a
     * NaN is false, so a block seeded with one would keep it and swallow every
     * real value behind it. This is damage control rather than a guarantee --
     * the release build's -ffast-math may assume NaN never occurs at all --
     * but it costs nothing and it bounds the harm to nothing at all in the
     * builds where the comparison does behave.
     *
     * SIF_REAL_MAX_VAL rather than an infinity, for the same reason: the
     * optimizer is entitled to assume infinities are absent too. */
    sif_real best = SIF_REAL_MAX_VAL;
    for (uint64_t i = lo; i < hi; i++)
      if (arr[i] < best)
        best = arr[i];

    partials[b] = best;
  }

  sif_real result = SIF_REAL_MAX_VAL;
  for (int b = 0; b < REDUCE_BLOCKS; b++)
    if (partials[b] < result)
      result = partials[b];

  return result;
}

sif_real sif_array_max(const sif_real* arr, uint64_t size) {
  if (!arr || size == 0)
    return (sif_real)0.0;

  sif_real partials[REDUCE_BLOCKS];

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(size, b, &lo, &hi);

    sif_real best = -SIF_REAL_MAX_VAL;
    for (uint64_t i = lo; i < hi; i++)
      if (arr[i] > best)
        best = arr[i];

    partials[b] = best;
  }

  sif_real result = -SIF_REAL_MAX_VAL;
  for (int b = 0; b < REDUCE_BLOCKS; b++)
    if (partials[b] > result)
      result = partials[b];

  return result;
}

#undef REDUCE_BLOCKS
