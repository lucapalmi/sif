/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file bitmask.h
 * @brief Dense array of bits, packed 64 to a word.
 *
 * Used where one boolean per grid cell would otherwise cost a byte or more:
 * the finders mark every cell swallowed by an accepted void, and at 1024^3
 * cells the difference between a bit and a byte is 128 MiB against 1 GiB.
 *
 * The accessors are `static inline` because they sit in the innermost loop of
 * the finders, where a call would cost more than the shift and mask it wraps.
 */

#ifndef SIF_STRUCTURES_BITMASK_H
#define SIF_STRUCTURES_BITMASK_H

#include "sif/core/macros.h"
#include <stdint.h>

/**
 * @brief A bitmask for tracking boolean states densely.
 *
 * @note The plain set/unset accessors are NOT thread-safe: distinct bits share
 * a 64-bit word, so concurrent read-modify-write silently loses updates. Code
 * that mutates the mask from inside a parallel region must use the _atomic
 * variants.
 */
typedef struct {
  /** Packed storage, 64 bits per word. Bits past #n_bits are always zero. */
  uint64_t* words;
  /** Bits the caller asked for. */
  uint64_t n_bits;
  /** Words actually allocated: ceil(n_bits / 64). */
  uint64_t n_words;
} sif_bitmask_t;

/**
 * @brief Allocate a bitmask with every bit clear.
 *
 * @param n_bits Number of bits. Must be non-zero.
 * @return The mask, owned by the caller and released with sif_bitmask_free().
 * NULL if @p n_bits is 0 or on allocation failure.
 */
SIF_NODISCARD sif_bitmask_t* sif_bitmask_alloc(uint64_t n_bits);

/**
 * @brief Release a bitmask and its storage.
 * @param mask Mask to free. NULL is accepted and ignored.
 */
void sif_bitmask_free(sif_bitmask_t* mask);

/**
 * @brief Clear every bit.
 * @param mask Mask to clear.
 */
void sif_bitmask_clear_all(sif_bitmask_t* mask);

/**
 * @brief Count the set bits.
 *
 * Parallelized across words with a population count per word, so it costs a
 * pass over the storage rather than over the bits.
 *
 * @param mask Mask to count.
 * @return The number of set bits, or 0 for a NULL or empty mask.
 */
SIF_NODISCARD uint64_t sif_bitmask_count_set(const sif_bitmask_t* mask);

/**
 * @brief Test one bit.
 * @param mask The mask.
 * @param ind Bit index; must be below sif_bitmask_t::n_bits.
 * @return 1 if the bit is set, 0 otherwise.
 */
static inline uint8_t sif_bitmask_get(const sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  return (mask->words[ind >> 6] & (1ULL << (ind & 63))) != 0;
}

/**
 * @brief Set one bit. Fast, and NOT thread-safe.
 * @param mask The mask.
 * @param ind Bit index; must be below sif_bitmask_t::n_bits.
 */
static inline void sif_bitmask_set(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  mask->words[ind >> 6] |= (1ULL << (ind & 63));
}

/**
 * @brief Clear one bit. Fast, and NOT thread-safe.
 * @param mask The mask.
 * @param ind Bit index; must be below sif_bitmask_t::n_bits.
 */
static inline void sif_bitmask_unset(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  mask->words[ind >> 6] &= ~(1ULL << (ind & 63));
}

/**
 * @brief Set one bit. Safe to call concurrently.
 *
 * @param mask The mask.
 * @param ind Bit index; must be below sif_bitmask_t::n_bits.
 *
 * @note The test-before-write is an atomic relaxed load, not a plain read: it
 * skips the expensive RMW when the bit is already set without introducing a
 * data race. Two threads racing to set the same bit is harmless, OR is
 * idempotent.
 */
static inline void sif_bitmask_set_atomic(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  uint64_t word_idx = ind >> 6;
  uint64_t bit = 1ULL << (ind & 63);

#if SIF_HAS_ATOMIC_BUILTINS
  if (!(SIF_ATOMIC_LOAD_U64(&mask->words[word_idx]) & bit))
    SIF_ATOMIC_OR_U64(&mask->words[word_idx], bit);
#else
#  pragma omp atomic
  mask->words[word_idx] |= bit;
#endif
}

/**
 * @brief Clear one bit. Safe to call concurrently.
 *
 * @param mask The mask.
 * @param ind Bit index; must be below sif_bitmask_t::n_bits.
 *
 * @note Same relaxed test-before-write as sif_bitmask_set_atomic().
 */
static inline void sif_bitmask_unset_atomic(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  uint64_t word_idx = ind >> 6;
  uint64_t bit = 1ULL << (ind & 63);

#if SIF_HAS_ATOMIC_BUILTINS
  if (SIF_ATOMIC_LOAD_U64(&mask->words[word_idx]) & bit)
    SIF_ATOMIC_AND_U64(&mask->words[word_idx], ~bit);
#else
#  pragma omp atomic
  mask->words[word_idx] &= ~bit;
#endif
}

#endif /* SIF_STRUCTURES_BITMASK_H */
