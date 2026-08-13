/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file excursion_set.h
 * @brief The excursion-set first-crossing problem: how often a random walk in
 * density first drops below a barrier.
 *
 * Two paths to the same multiplicity function. sif_ep_multiplicity_function()
 * runs correlated walks by Monte Carlo, which needs the full packed covariance
 * and costs quadratic work in the radius count.
 * sif_ep_multiplicity_function_emu() evaluates a trained network over the same
 * features, needs only the diagonal, and is linear -- about a million times
 * faster, at a fitted accuracy quoted with the emulator.
 *
 * The upcrossing form is the analytic limit both are measured against.
 */

#ifndef SIF_MODEL_EXCURSION_SET_H
#define SIF_MODEL_EXCURSION_SET_H

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/model/delta_moments.h"

/**
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
SIF_NODISCARD sif_real* sif_svdw_multiplicity_function(
  const sif_real* sigma, uint32_t n, sif_real delta_v, sif_real delta_c);

/**
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
SIF_NODISCARD sif_real* sif_ep_barrier_smt(const sif_real* sigma, uint32_t n,
  sif_real alpha, sif_real beta, sif_real gamma);

/**
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
 * with a strictly positive diagonal. Double rather than sif_real: a
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
SIF_NODISCARD uint64_t* sif_ep_first_crossing_counts(const sif_real* radii,
  uint32_t n_radii, const double* cov, const sif_real* barrier,
  uint64_t n_paths, uint64_t seed, sif_option opt);

/**
 * @brief Lagrangian void multiplicity function from the first crossing of a
 * moving barrier by a correlated random walk.
 *
 *     f[i] = (walks first crossing at radii[i])
 *            / (n_paths * (radii[i+1] - radii[i]))
 *
 * @note The result has n_radii - 1 entries, on the bin centres
 * 0.5 (radii[i] + radii[i+1]), not n_radii point values on the radii
 * themselves; the largest radius is the walk's first step and has no bin above
 * it. This is a different shape from sif_svdw_multiplicity_function, which is
 * point-evaluated.
 *
 * @note The Monte Carlo error on a bin is sqrt(n_i) / (n_paths dr_i), so a bin
 * holding a fraction p of the walks is known to a relative 1 / sqrt(p n_paths).
 * Take `counts` and check it before trusting the tails; re-running the walk
 * through sif_ep_first_crossing_counts to get the same numbers doubles the
 * cost of the most expensive call in the library.
 *
 * @param counts Optional output, n_radii entries: the raw first-crossing
 * counts this multiplicity was built from, exactly as
 * sif_ep_first_crossing_counts would have returned them. Pass NULL to skip.
 *
 * Remaining parameters, and the barrier and reproducibility conventions, are as
 * sif_ep_first_crossing_counts.
 *
 * @return Newly allocated array of n_radii - 1 values, released with
 * sif_free_aligned, or NULL on invalid input or a failed factorization.
 */
SIF_NODISCARD sif_real* sif_ep_multiplicity_function(const sif_real* radii,
  uint32_t n_radii, const double* cov, const sif_real* barrier,
  uint64_t n_paths, uint64_t seed, uint64_t* counts, sif_option opt);

/**
 * @brief Report on where an emulated call sat relative to what the emulator
 * was trained over. Optional; pass NULL to sif_ep_multiplicity_function_emu if
 * the answer is all that is wanted.
 *
 * Leaving the trained region is not an error and does not fail the call. It
 * costs accuracy -- around 0.33% against the usual 0.13% at the edge that was
 * measured -- and a sampler should be able to notice that without parsing the
 * log or aborting a chain over one proposal.
 */
typedef struct {
  /* Zero if either check below fired. */
  int in_domain;

  /* Bins with at least one input outside the range the fit covered. */
  uint32_t n_bins_outside;

  /* B / sigma at the LARGEST radius, and the fraction of walks that therefore
   * begin already above the barrier. The walk starts there, so that fraction
   * never enters any bin; above a per cent the emulator was measured to be
   * unreliable. This one is the caller's to fix, by extending the radius grid
   * outward -- no amount of training would help. */
  sif_real nu_origin;
  sif_real first_step_mass;

  /* Expected relative error, from the trainer's held-out validation: the
   * in-domain figure when in_domain is set, and the measured out-of-domain one
   * otherwise. */
  sif_real expected_error;
} sif_emu_domain_t;

/**
 * @brief The same multiplicity function as sif_ep_multiplicity_function,
 * emulated: no random walks, no paths, well under a millisecond.
 *
 * A semi-analytic up-crossing rate corrected by a small trained network. The
 * correction multiplies a hazard rather than the multiplicity itself, and the
 * result is rebuilt through the survival recursion, so it is non-negative and
 * integrates to at most one whatever the network predicts.
 *
 * @note Takes `sigma`, not the packed covariance that
 * sif_ep_multiplicity_function needs. That is deliberate rather than an
 * oversight: the emulator reads only the diagonal, so it needs n numbers where
 * the Monte Carlo needs n(n+1)/2, and its cost is linear rather than quadratic
 * in the radius count.
 *
 * @note Returns the answer the Monte Carlo CONVERGES to, not the answer it
 * gives on the caller's grid. Those differ: a first-crossing walk sampled at 50
 * radii sits about 1% from its own continuum limit. Comparing the two on a
 * coarse grid shows a disagreement larger than either method's error, and the
 * emulator is the one to trust. The emulated result moves by under 0.25%
 * between 64 and 256 radii.
 *
 * @note Trained over CDM-like spectra and Sheth-Mo-Tormen barriers. Other
 * spectral families are outside the contract, and the `domain` report does not
 * reliably catch them: a per-feature range check passes power-law spectra that
 * are several per cent wrong, because what distinguishes them is the shape of
 * the whole trajectory rather than any pointwise value.
 *
 * @param radii Smoothing radii, strictly positive and strictly increasing, at
 * least 3. Extend the grid outward far enough that few walks start above the
 * barrier; `domain` reports whether that was achieved.
 * @param n_radii Number of radii
 * @param sigma R.m.s. density contrast per radius, strictly positive. Take it
 * from sif_delta_covariance_pk so the barrier and the walk share one variance.
 * @param barrier n_radii barrier heights, in the ascending order of radii,
 * finite and compared as an upcrossing exactly as the Monte Carlo does
 * @param deriv_variance n_radii entries: <(d delta / dS)^2> from
 * sif_delta_covariance_pk. REQUIRED, unlike the Monte Carlo path, which has no
 * use for it: differencing it off a covariance converges only at first order,
 * which would make the answer depend on how finely `radii` was sampled -- the
 * one property this entry point exists to avoid.
 * @param domain Optional report; pass NULL to skip
 * @param opt Reserved; pass SIF_DEFAULT
 *
 * @return Newly allocated array of n_radii - 1 values on the same bin centres
 * as sif_ep_multiplicity_function, released with sif_free_aligned, or NULL on
 * invalid input. An input outside the trained region is NOT invalid: the answer
 * is returned, `domain` records it, and the log names the quantity responsible.
 */
SIF_NODISCARD sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, sif_emu_domain_t* domain, sif_option opt);

#endif /* SIF_MODEL_EXCURSION_SET_H */
