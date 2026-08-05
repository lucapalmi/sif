#ifndef __SIF_BITMASK_H__
#define __SIF_BITMASK_H__

#include "sif/core/macros.h"
#include <stdint.h>

/*
 * @brief A bitmask for tracking boolean states densely
 *
 * @note The plain set/unset accessors are NOT thread-safe: distinct bits share
 * a 64-bit word, so concurrent read-modify-write silently loses updates. Code
 * that mutates the mask from inside a parallel region must use the _atomic
 * variants.
 */
typedef struct {
  uint64_t* words;
  uint64_t n_bits;
  uint64_t n_words;
} sif_bitmask_t;

NODISCARD sif_bitmask_t* sif_bitmask_alloc(uint64_t n_bits);
void sif_bitmask_free(sif_bitmask_t* mask);

void sif_bitmask_clear_all(sif_bitmask_t* mask);
uint64_t sif_bitmask_count_set(const sif_bitmask_t* mask);

/*
 * @brief Checks if a specific bit is set (1) or unset (0).
 */
static inline uint8_t sif_bitmask_get(const sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  return (mask->words[ind >> 6] & (1ULL << (ind & 63))) != 0;
}

/*
 * @brief Sets a specific bit to 1 (fast, NOT thread-safe)
 */
static inline void sif_bitmask_set(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  mask->words[ind >> 6] |= (1ULL << (ind & 63));
}

/*
 * @brief Unsets a specific bit to 0 (fast, NOT thread-safe)
 */
static inline void sif_bitmask_unset(sif_bitmask_t* mask, uint64_t ind) {
  SIF_ASSERT(ind < mask->n_bits);
  mask->words[ind >> 6] &= ~(1ULL << (ind & 63));
}

/*
 * @brief Sets a specific bit to 1. Safe to call concurrently.
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

/*
 * @brief Unsets a specific bit to 0. Safe to call concurrently.
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

#endif /* __SIF_BITMASK_H__ */
