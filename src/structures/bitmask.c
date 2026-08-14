/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/bitmask.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <stdlib.h>
#include <string.h>

sif_bitmask_t* sif_bitmask_alloc(uint64_t n_bits) {
  if (n_bits == 0) {
    SIF_LOG_ERROR("bitmask", "cannot allocate a zero-length bitmask");
    return NULL;
  }

  sif_bitmask_t* mask = malloc(sizeof(sif_bitmask_t));
  if (!mask) {
    SIF_LOG_ERROR("bitmask", "failed to allocate bitmask (%zu bytes)",
      sizeof(sif_bitmask_t));
    return NULL;
  }

  mask->n_bits = n_bits;
  mask->n_words = (n_bits + 63) >> 6;

  /* Cleared, and load-bearing rather than tidy: the trailing bits of the last
   * word have no index that can reach them, so nothing will ever write them,
   * and sif_bitmask_count_set() popcounts whole words on the strength of their
   * being zero from here on. */
  mask->words = sif_calloc_aligned(mask->n_words, sizeof(uint64_t));

  if (!mask->words) {
    SIF_LOG_ERROR(
      "bitmask", "failed to allocate %" PRIu64 " mask words", mask->n_words);
    free(mask);
    return NULL;
  }

  return mask;
}

void sif_bitmask_free(sif_bitmask_t* mask) {
  if (!mask)
    return;

  sif_free_aligned(mask->words);
  free(mask);
}

void sif_bitmask_clear_all(sif_bitmask_t* mask) {
  if (!mask || !mask->words) {
    SIF_LOG_ERROR("bitmask", "invalid bitmask");
    return;
  }

  memset(mask->words, 0, mask->n_words * sizeof(uint64_t));
}

uint64_t sif_bitmask_count_set(const sif_bitmask_t* mask) {
  if (!mask || !mask->words) {
    SIF_LOG_ERROR("bitmask", "invalid bitmask");
    return 0;
  }

  uint64_t count = 0;
  const uint64_t n_words = mask->n_words;
  const uint64_t* words = mask->words;

  /*
   * Whole words, including the last one, with no mask over the tail.
   *
   * That is only correct because nothing can set a bit past n_bits: the
   * mutators take a bit index, and a caller that respects the documented
   * range never reaches the padding. The allocation cleared it, so the
   * popcount of the final word counts real bits only.
   *
   * Note the precondition is the caller's: SIF_ASSERT compiles away in a
   * release build, so an out-of-range index there corrupts this count rather
   * than tripping anything.
   *
   * The reduction is over integers, so it is exact and the answer does not
   * depend on the thread count.
   */
#pragma omp parallel for schedule(static) reduction(+ : count)
  for (uint64_t i = 0; i < n_words; i++) {
    count += SIF_POPCOUNT_U64(words[i]);
  }

  return count;
}
