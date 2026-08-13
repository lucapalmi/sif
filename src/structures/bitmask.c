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

  /* calloc zeros the memory automatically */
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

  /* The trailing bits of the last word past n_bits are guaranteed zero:
   * the buffer is calloc'd and nothing can set them (every mutator masks the
   * index to 6 bits and every index is asserted < n_bits). */
#pragma omp parallel for schedule(static) reduction(+ : count)
  for (uint64_t i = 0; i < n_words; i++) {
    count += (uint64_t)__builtin_popcountll(words[i]);
  }

  return count;
}
