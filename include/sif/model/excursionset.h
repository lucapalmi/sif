#ifndef __SIF_MODEL_EXCURSIONSET_H__
#define __SIF_MODEL_EXCURSIONSET_H__

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/model/deltamoments.h"

/*
 * @brief Excursion-set void multiplicity function f_ln(sigma), for a constant
 * barrier and uncorrelated steps.
 *
 * Sheth & van de Weygaert (2004) as given by Jennings, Li & Hu (2013) eq. (8).
 *
 * @note The mode series is summed to convergence rather than truncated at the
 * four terms of the reference, which keeps it valid as
 * D = |delta_v| / (delta_c + |delta_v|) approaches 1.
 *
 * @param sigma R.m.s. density contrast per entry, strictly positive
 * @param n Length of sigma
 * @param delta_v Linear void barrier, strictly negative
 * @param delta_c Collapse barrier, strictly positive
 *
 * @return Newly allocated array of n values, released with sif_free_aligned,
 * or NULL on invalid input.
 */
NODISCARD real_t* sif_multiplicity_function_svdw(
  const real_t* sigma, uint32_t n, real_t delta_v, real_t delta_c);

/*
 * @brief The Sheth-Mo-Tormen moving barrier,
 * B(sigma) = alpha [1 + (beta / sigma)^gamma].
 *
 * gamma = 0 gives the constant 2 alpha; large sigma tends to alpha.
 *
 * @note Positive, and compared as an upcrossing by the first-crossing routines
 * below. For a void that means alpha carries the magnitude of the
 * underdensity, not its sign, and the walk is that of -delta.
 *
 * @param sigma R.m.s. density contrast per smoothing scale, strictly positive.
 * Take it from sif_delta_covariance_pk, so the barrier and the walk share one
 * variance.
 * @param n Length of sigma
 * @param alpha Barrier amplitude, strictly positive
 * @param beta Barrier scale in units of sigma, strictly positive
 * @param gamma Barrier slope
 *
 * @return Newly allocated array of n barrier heights, released with
 * sif_free_aligned, or NULL on invalid input.
 */
NODISCARD real_t* sif_barrier_smt(
  const real_t* sigma, uint32_t n, real_t alpha, real_t beta, real_t gamma);

/*
 * @brief Raw first-crossing counts of a correlated random walk against a
 * moving barrier, one count per smoothing radius.
 *
 * Each path draws n_radii standard normals xi and forms the walk
 * delta(R_j) = sum_{m <= j} L_jm xi_m, with L L^T = cov, stepping from the
 * largest radius down and recording the first radius at which delta reaches
 * the barrier. Paths that never reach it are not recorded, so the counts sum
 * to at most n_paths.
 *
 * @note The barrier is an upcrossing condition, delta >= B, which makes the
 * sign convention of the barrier the caller's responsibility. Reversing it
 * produces a plausible-looking curve rather than an obvious failure.
 *
 * @note The result is a deterministic function of (seed, n_paths) alone: the
 * counts do not depend on the thread count or on how the loop was scheduled.
 *
 * @param radii Smoothing radii, strictly positive and strictly increasing, at
 * least 2 and at most SIF_COV_MAX_RADII
 * @param n_radii Number of radii
 * @param cov Covariance of the smoothed field between every pair of radii,
 * packed lower triangle in the ascending order of radii: S(i, j) at
 * SIF_COV_INDEX(i, j) for j <= i. Must be symmetric positive semi-definite
 * with a strictly positive diagonal. Double rather than real_t: a
 * single-precision factorization of a realistic radius grid reaches a
 * non-positive pivot and fails.
 * @param barrier n_radii barrier heights, in the ascending order of radii
 * @param n_paths Number of walks, strictly positive
 * @param seed Seed for the walk ensemble
 * @param opt Reserved; pass SIF_DEFAULT
 *
 * @return Newly allocated array of n_radii counts in the ascending order of
 * radii, released with sif_free_aligned, or NULL on invalid input or a failed
 * factorization.
 */
NODISCARD uint64_t* sif_first_crossing_counts_ep(const real_t* radii,
  uint32_t n_radii, const double* cov, const real_t* barrier, uint64_t n_paths,
  uint64_t seed, sif_option_t opt);

/*
 * @brief Lagrangian void multiplicity function from the first crossing of a
 * moving barrier by a correlated random walk.
 *
 *     f[i] = (walks first crossing at radii[i])
 *            / (n_paths * (radii[i+1] - radii[i]))
 *
 * @note The result has n_radii - 1 entries, on the bin centres
 * 0.5 (radii[i] + radii[i+1]), not n_radii point values on the radii
 * themselves; the largest radius is the walk's first step and has no bin above
 * it. This is a different shape from sif_multiplicity_function_svdw, which is
 * point-evaluated.
 *
 * @note The Monte Carlo error on a bin is sqrt(n_i) / (n_paths dr_i), so a bin
 * holding a fraction p of the walks is known to a relative 1 / sqrt(p n_paths).
 * Check the counts from sif_first_crossing_counts_ep before trusting the tails.
 *
 * Remaining parameters, and the barrier and reproducibility conventions, are as
 * sif_first_crossing_counts_ep.
 *
 * @return Newly allocated array of n_radii - 1 values, released with
 * sif_free_aligned, or NULL on invalid input or a failed factorization.
 */
NODISCARD real_t* sif_multiplicity_function_ep(const real_t* radii,
  uint32_t n_radii, const double* cov, const real_t* barrier, uint64_t n_paths,
  uint64_t seed, sif_option_t opt);

#endif /* __SIF_MODEL_EXCURSIONSET_H__ */
