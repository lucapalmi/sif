/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/align.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Round a byte count up to a whole number of cache lines.
 *
 * Rounding the size matters as much as aligning the address: an aligned start
 * alone still lets the tail of one block share a line with the head of the
 * next, and two OpenMP threads writing into those two blocks would then
 * serialize on the coherence protocol while appearing to touch separate data.
 *
 * Returns 0 if the rounding would wrap. Callers treat that as failure -- a
 * wrapped count asks the allocator for a handful of bytes and gets a block the
 * caller believes is enormous.
 */
static size_t round_to_cache_line(size_t size) {
  if (size > SIZE_MAX - (SIF_CACHE_LINE - 1))
    return 0;

  return (size + SIF_CACHE_LINE - 1) & ~(size_t)(SIF_CACHE_LINE - 1);
}

void* sif_malloc_aligned(size_t size) {
  if (size == 0)
    return NULL;

  const size_t padded = round_to_cache_line(size);
  if (padded == 0)
    return NULL;

  void* ptr = NULL;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L &&                \
  !defined(__APPLE__)
  ptr = aligned_alloc(SIF_CACHE_LINE, padded);
#else
  /* Apple is excluded from the C11 path deliberately: aligned_alloc arrived
   * only in macOS 10.15, and a build targeting anything older links against a
   * libc that does not have it. posix_memalign is there in every version. */
  if (posix_memalign(&ptr, SIF_CACHE_LINE, padded) != 0)
    return NULL;
#endif

  return ptr;
}

void* sif_calloc_aligned(size_t count, size_t size) {
  if (count == 0 || size == 0)
    return NULL;

  /* This check is the reason to call the calloc form rather than malloc and
   * memset by hand: a product that wraps allocates a small block, clears that
   * small block, and leaves every subsequent write running off its end. */
  if (count > SIZE_MAX / size)
    return NULL;

  const size_t total = count * size;
  void* ptr = sif_malloc_aligned(total);
  if (!ptr)
    return NULL;

  /* The padding is cleared along with the request. It costs nothing -- the
   * tail line is in cache either way -- and it means the block holds no
   * leftover bytes anywhere, which is what a caller reading a rounded-up
   * length (the field and grid blocks do) is entitled to assume. */
  memset(ptr, 0, round_to_cache_line(total));

  return ptr;
}

void* sif_realloc_aligned(void* ptr, size_t old_size, size_t new_size) {
  /* realloc()'s two degenerate cases, kept so this is a drop-in replacement. */
  if (new_size == 0) {
    sif_free_aligned(ptr);
    return NULL;
  }
  if (!ptr)
    return sif_malloc_aligned(new_size);

  /* Always a fresh block and a copy. realloc() may extend in place, but the
   * address it extends is only guaranteed to keep malloc's alignment, not
   * ours, so there is nothing to salvage from trying. */
  void* new_ptr = sif_malloc_aligned(new_size);
  if (!new_ptr)
    return NULL; /* the original is left intact for the caller to keep */

  const size_t copy_size = (old_size < new_size) ? old_size : new_size;
  memcpy(new_ptr, ptr, copy_size);
  sif_free_aligned(ptr);

  return new_ptr;
}

void sif_free_aligned(void* ptr) {
  /* Both allocation paths above hand back memory that plain free() accepts:
   * aligned_alloc and posix_memalign are specified that way. The wrapper
   * exists so the header can promise a matching _free for every allocator,
   * and so a future allocator that does need special handling has somewhere
   * to put it. */
  free(ptr);
}
