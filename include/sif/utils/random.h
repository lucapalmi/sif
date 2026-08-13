/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file random.h
 * @brief xoshiro256+ pseudo-random generator.
 *
 * Header-only and state-passing: there is no global generator, so each thread
 * or task holds its own sif_prng_state_t and draws from it without
 * synchronization. Seed the states from distinct seeds -- the SplitMix64
 * expansion below decorrelates even adjacent ones.
 *
 * @note xoshiro256+ is fast and statistically sound for simulation, but it is
 * not cryptographically secure and its lowest bits are the weakest; the
 * conversions here take from the top.
 */

#ifndef SIF_UTILS_RANDOM_H
#define SIF_UTILS_RANDOM_H

#include "sif/core/macros.h"

#include <math.h>
#include <stdint.h>

/** @brief Generator state: the four 64-bit words of xoshiro256+. */
typedef struct {
  uint64_t s[4];
} sif_prng_state_t;

/* --- internal --- */

/**
 * @brief SplitMix64 step. Not part of the API.
 *
 * Used to expand a single seed into the four state words. Seeding xoshiro
 * directly from a small integer leaves it correlated for the first outputs;
 * SplitMix64 has a different structure and breaks that.
 */
static inline uint64_t sif__splitmix64(uint64_t* state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/** @brief 64-bit rotate left. Not part of the API. */
static inline uint64_t sif__rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* --- public --- */

/**
 * @brief Seed a generator state.
 *
 * @param state State to initialize.
 * @param seed Any 64-bit value, including 0; it is expanded through
 * SplitMix64 into the 256 bits xoshiro needs.
 */
static inline void sif_prng_init(sif_prng_state_t* state, uint64_t seed) {
  state->s[0] = sif__splitmix64(&seed);
  state->s[1] = sif__splitmix64(&seed);
  state->s[2] = sif__splitmix64(&seed);
  state->s[3] = sif__splitmix64(&seed);
}

/**
 * @brief Next raw 64-bit draw, advancing the state.
 */
static inline uint64_t sif_prng_next_u64(sif_prng_state_t* state) {
  uint64_t const result = state->s[0] + state->s[3];
  uint64_t const t = state->s[1] << 17;

  state->s[2] ^= state->s[0];
  state->s[3] ^= state->s[1];
  state->s[1] ^= state->s[2];
  state->s[0] ^= state->s[3];
  state->s[2] ^= t;
  state->s[3] = sif__rotl(state->s[3], 45);

  return result;
}

/**
 * @brief Next draw as a sif_real in [0, 1).
 *
 * Takes 53 bits in a double build and 24 in a float build, matching what the
 * type can represent.
 */
static inline sif_real sif_prng_next_real(sif_prng_state_t* state) {
#ifdef SIF_USE_DOUBLE
  /* 53 bits of precision for double */
  return (sif_real)(sif_prng_next_u64(state) >> 11) * (1.0 / (1ULL << 53));
#else
  /* 24 bits of precision for float */
  return (sif_real)(sif_prng_next_u64(state) >> 40) * (1.0f / (1ULL << 24));
#endif
}

/**
 * @brief Next draw as a double in [0, 1), always with the full 53 bits.
 *
 * Deliberately does not follow sif_real. At 24 bits the smallest non-zero draw
 * is 6e-8, which puts a hard floor of about 5.7 sigma on the tail any Gaussian
 * built on it can reach.
 */
static inline double sif_prng_next_double(sif_prng_state_t* state) {
  return (double)(sif_prng_next_u64(state) >> 11) * (1.0 / (1ULL << 53));
}

/**
 * @brief Two independent standard normal deviates, by the Box-Muller
 * transform.
 *
 * Returned as a pair because one logarithm, one square root and one
 * sine-cosine serve both, so a caller that keeps only the first pays twice per
 * normal.
 *
 * @param state PRNG state, advanced by two draws.
 * @param z0 First deviate, written.
 * @param z1 Second deviate, written.
 */
static inline void sif_prng_next_gaussian_pair(
  sif_prng_state_t* state, double* z0, double* z1) {

  /* 1 - u, since next_double can return exactly zero and log(0) would poison
   * both deviates. */
  const double u1 = 1.0 - sif_prng_next_double(state);
  const double u2 = sif_prng_next_double(state);

  const double r = sqrt(-2.0 * log(u1));
  /* 2*pi spelled out rather than taken from SIF_PI, which is sif_real and so
   * would round the angle to float precision in a single-precision build --
   * defeating the point of computing the pair in double. */
  const double theta = 6.283185307179586476925286766559 * u2;

  *z0 = r * cos(theta);
  *z1 = r * sin(theta);
}

#endif /* SIF_UTILS_RANDOM_H */
