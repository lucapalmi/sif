/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file array.h
 * @brief Constructors and reductions for flat sif_real arrays.
 *
 * The constructors mirror the numpy functions of the same name, so a pipeline
 * prototyped in Python translates without re-deriving the bin edges.
 *
 * @warning Every array returned here is cache-line aligned memory from
 * sif_malloc_aligned(). Release it with sif_free_aligned(), never with free().
 */

#ifndef SIF_UTILS_ARRAY_H
#define SIF_UTILS_ARRAY_H

#include "sif/core/macros.h"
#include <stdint.h>

/* --- construction --- */

/**
 * @brief Array of @p size zeros. Equivalent to `np.zeros`.
 * @return Owned array, or NULL on failure or if @p size is 0.
 */
SIF_NODISCARD sif_real* sif_array_zeros(uint64_t size);

/**
 * @brief Array of @p size ones. Equivalent to `np.ones`.
 * @return Owned array, or NULL on failure or if @p size is 0.
 */
SIF_NODISCARD sif_real* sif_array_ones(uint64_t size);

/**
 * @brief Array of @p size copies of @p fill_value. Equivalent to `np.full`.
 * @return Owned array, or NULL on failure or if @p size is 0.
 */
SIF_NODISCARD sif_real* sif_array_full(uint64_t size, sif_real fill_value);

/* --- ranges --- */

/**
 * @brief @p num evenly spaced samples over the closed interval
 * [@p start, @p stop]. Equivalent to `np.linspace`.
 *
 * Both endpoints are included, so consecutive samples are separated by
 * (stop - start) / (num - 1), and the last element is assigned @p stop
 * exactly rather than accumulated, which would drift. @p num of 1 yields
 * `{start}`.
 *
 * @return Owned array of @p num elements, or NULL on failure.
 */
SIF_NODISCARD sif_real* sif_array_linspace(
  sif_real start, sif_real stop, uint64_t num);

/**
 * @brief Evenly spaced values over the half-open interval
 * [@p start, @p stop). Equivalent to `np.arange`.
 *
 * The element count follows from the step rather than being given, so it is
 * reported through @p out_size.
 *
 * @param start First value.
 * @param stop Exclusive upper bound. May be below @p start if @p step is
 * negative.
 * @param step Spacing. Must not be zero.
 * @param out_size Written with the number of elements produced, including 0
 * for an empty or rejected range. May not be NULL if the result is to be used.
 * @return Owned array, or NULL for an empty range, a zero step, or on failure.
 */
SIF_NODISCARD sif_real* sif_array_arange(
  sif_real start, sif_real stop, sif_real step, uint64_t* out_size);

/**
 * @brief @p num values spaced evenly on a log scale, from `base^start` to
 * `base^stop`. Equivalent to `np.logspace`.
 *
 * Note that the *exponents* are evenly spaced, not the values: the endpoints
 * are powers of @p base, not @p start and @p stop themselves. As with
 * sif_array_linspace() the final element is computed directly from @p stop.
 *
 * @return Owned array of @p num elements, or NULL on failure.
 */
SIF_NODISCARD sif_real* sif_array_logspace(
  sif_real start, sif_real stop, uint64_t num, sif_real base);

/* --- reductions --- */

/**
 * @brief Sum of every element.
 * @return The sum; 0 for an empty or NULL array.
 *
 * @note The array is split into a fixed number of blocks, each accumulated in
 * `double` and combined in index order, so the result depends on the input
 * alone and not on how many threads happened to run. It is also markedly more
 * accurate than a running `sif_real` total in a single-precision build, where
 * a few million similar terms are enough to stall the accumulator.
 */
SIF_NODISCARD sif_real sif_array_sum(const sif_real* arr, uint64_t size);

/**
 * @brief Smallest element.
 * @return The minimum, or 0 for an empty or NULL array -- which is
 * indistinguishable from a genuine minimum of 0. Check @p size first if the
 * difference matters.
 *
 * @warning The release build compiles with `-ffast-math`, which lets the
 * compiler assume no NaN ever occurs, so an array containing one has no
 * defined result. The reduction is seeded with #SIF_REAL_MAX_VAL rather than
 * with the first element, which keeps a NaN from capturing everything behind
 * it wherever the comparison does behave -- but that is damage control, not a
 * guarantee. Screen for NaN before reducing if the data can carry it.
 */
SIF_NODISCARD sif_real sif_array_min(const sif_real* arr, uint64_t size);

/**
 * @brief Largest element.
 * @return The maximum, or 0 for an empty or NULL array -- which is
 * indistinguishable from a genuine maximum of 0. Check @p size first if the
 * difference matters.
 *
 * @warning NaN carries the same caveat as sif_array_min().
 */
SIF_NODISCARD sif_real sif_array_max(const sif_real* arr, uint64_t size);

#endif /* SIF_UTILS_ARRAY_H */
