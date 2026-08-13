/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file align.h
 * @brief Cache-line aligned allocation.
 *
 * Every bulk buffer in sif comes from here rather than from malloc. Two
 * reasons: the vectorized loops want their operands aligned, and blocks handed
 * to different OpenMP threads must not share a cache line, or the threads
 * serialize on the coherence protocol while appearing to touch separate data.
 *
 * Allocations are therefore both aligned to #SIF_CACHE_LINE and rounded *up*
 * to a whole number of cache lines, so two adjacent allocations can never
 * share one.
 *
 * @warning Memory from these functions must be released with
 * sif_free_aligned(), never with free().
 */

#ifndef SIF_UTILS_ALIGN_H
#define SIF_UTILS_ALIGN_H

#include "sif/core/macros.h"
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
/** @brief Tell the compiler a pointer is cache-line aligned. */
#  define SIF_ASSUME_ALIGNED(ptr)                                              \
    __builtin_assume_aligned((ptr), SIF_CACHE_LINE)
/** @brief Mark a function as returning fresh, cache-line aligned memory. */
#  define SIF_ALIGNED_FN __attribute__((malloc, assume_aligned(SIF_CACHE_LINE)))
#else
#  define SIF_ASSUME_ALIGNED(ptr) (ptr)
#  define SIF_ALIGNED_FN
#endif

/**
 * @brief Allocate aligned, uninitialized memory.
 *
 * @param size Number of bytes required.
 * @return Pointer to the block, owned by the caller and released with
 * sif_free_aligned(). NULL on failure, and also when @p size is 0.
 */
SIF_NODISCARD SIF_ALIGNED_FN void* sif_malloc_aligned(size_t size);

/**
 * @brief Allocate aligned, zero-initialized memory.
 *
 * @param count Number of elements.
 * @param size Size of each element, in bytes.
 * @return Pointer to the zeroed block, owned by the caller and released with
 * sif_free_aligned(). NULL on failure.
 */
SIF_NODISCARD SIF_ALIGNED_FN void* sif_calloc_aligned(
  size_t count, size_t size);

/**
 * @brief Resize an aligned block, preserving its contents.
 *
 * Unlike realloc() this always allocates a new block and copies -- alignment
 * cannot be preserved by growing in place -- so it costs a full copy every
 * time. Prefer sizing the buffer correctly over growing it in a loop.
 *
 * Passing NULL for @p ptr allocates; passing 0 for @p new_size frees and
 * returns NULL. On failure the original block is left untouched, so assigning
 * the result straight back to @p ptr leaks it.
 *
 * @param ptr Block to resize, or NULL.
 * @param old_size Current size of @p ptr in bytes.
 * @param new_size Requested size in bytes. Content beyond it is discarded.
 * @return The new block, or NULL on failure.
 *
 * @warning @p old_size is trusted, not checked. Passing a value larger than
 * the real allocation reads past its end.
 */
SIF_NODISCARD void* sif_realloc_aligned(
  void* ptr, size_t old_size, size_t new_size);

/**
 * @brief Release a block from one of the allocators above.
 * @param ptr Block to free. NULL is accepted and ignored.
 */
void sif_free_aligned(void* ptr);

#endif /* SIF_UTILS_ALIGN_H */
