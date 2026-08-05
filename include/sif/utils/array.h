#ifndef __SIF_ARRAY_H__
#define __SIF_ARRAY_H__

#include "sif/core/macros.h"
#include <stdint.h>

/* --- Allocation & Initialization --- */

/* Equivalent to np.zeros() */
NODISCARD real_t* sif_array_zeros(uint64_t size);

/* Equivalent to np.ones() */
NODISCARD real_t* sif_array_ones(uint64_t size);

/* Equivalent to np.full() */
NODISCARD real_t* sif_array_full(uint64_t size, real_t fill_value);

/* --- Ranges & Sequences --- */

/* * Equivalent to np.linspace(start, stop, num).
 * Generates `num` evenly spaced samples, calculated over the interval [start,
 * stop].
 */
NODISCARD real_t* sif_array_linspace(real_t start, real_t stop, uint64_t num);

/* * Equivalent to np.arange(start, stop, step).
 * Return evenly spaced values within a given half-open interval [start, stop).
 * The number of generated elements is written to `out_size`.
 */
NODISCARD real_t* sif_array_arange(
  real_t start, real_t stop, real_t step, uint64_t* out_size);

/* * Equivalent to np.logspace(start, stop, num).
 * Generates `num` numbers spaced evenly on a log scale.
 * The sequence starts at base^start and ends with base^stop.
 */
NODISCARD real_t* sif_array_logspace(
  real_t start, real_t stop, uint64_t num, real_t base);

/* --- Basic Reductions --- */

/* Returns the sum of all elements in the array */
real_t sif_array_sum(const real_t* arr, uint64_t size);

/* Returns the minimum value in the array */
real_t sif_array_min(const real_t* arr, uint64_t size);

/* Returns the maximum value in the array */
real_t sif_array_max(const real_t* arr, uint64_t size);

#endif /* __SIF_ARRAY_H__ */