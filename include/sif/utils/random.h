#ifndef __SIF_RANDOM_H__
#define __SIF_RANDOM_H__

#include "sif/core/macros.h"
#include <stdint.h>

/* State structure for xoshiro256+ */
typedef struct {
  uint64_t s[4];
} sif_prng_state_t;

/* --- Internal Helpers --- */

/* SplitMix64 is used to strictly un-correlate the initial seed */
static inline uint64_t __sif_splitmix64(uint64_t* state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline uint64_t __sif_rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* --- Public API --- */

/*
 * @brief Initialize the PRNG state using a 64-bit seed.
 * Safely generates the 256 bits of state required by xoshiro.
 */
static inline void sif_prng_init(sif_prng_state_t* state, uint64_t seed) {
  state->s[0] = __sif_splitmix64(&seed);
  state->s[1] = __sif_splitmix64(&seed);
  state->s[2] = __sif_splitmix64(&seed);
  state->s[3] = __sif_splitmix64(&seed);
}

/*
 * @brief Generate the next random 64-bit unsigned integer.
 */
static inline uint64_t sif_prng_next_u64(sif_prng_state_t* state) {
  uint64_t const result = state->s[0] + state->s[3];
  uint64_t const t = state->s[1] << 17;

  state->s[2] ^= state->s[0];
  state->s[3] ^= state->s[1];
  state->s[1] ^= state->s[2];
  state->s[0] ^= state->s[3];
  state->s[2] ^= t;
  state->s[3] = __sif_rotl(state->s[3], 45);

  return result;
}

/*
 * @brief Generate a random real_t in the interval [0, 1).
 * Automatically adapts to the precision of real_t (float vs double).
 */
static inline real_t sif_prng_next_real(sif_prng_state_t* state) {
#ifdef __SIF_USE_DOUBLE
  /* 53 bits of precision for double */
  return (real_t)(sif_prng_next_u64(state) >> 11) * (1.0 / (1ULL << 53));
#else
  /* 24 bits of precision for float */
  return (real_t)(sif_prng_next_u64(state) >> 40) * (1.0f / (1ULL << 24));
#endif
}

#endif /* __SIF_RANDOM_H__ */