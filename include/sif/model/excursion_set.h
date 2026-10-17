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
 * The upcrossing form -- Verza et al. (2024) eq. (3.15), the Musso-Sheth rate
 * with the exact scale-dependent <(d delta / dS)^2> -- is the analytic limit
 * both are measured against.
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
 * `D = |delta_v| / (delta_c + |delta_v|)` approaches 1.
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
 * With a `collapse` barrier the walk is stopped at whichever it reaches first,
 * and only a void crossing is counted: a walk that first reaches collapse at
 * some radius sits inside a collapsing region there, and is not a void at that
 * radius or any smaller one -- the void-in-cloud of Sheth & van de Weygaert
 * (2004). The same seed gives the same walks with or without it, so the two
 * runs differ by exactly the walks a cloud crushed.
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
 * single-precision factorisation of a realistic radius grid reaches a
 * non-positive pivot and fails.
 * @param barrier n_radii barrier heights, in the ascending order of radii
 * @param collapse Optional, n_radii collapse thresholds in the ascending order
 * of radii, strictly positive: delta_c (constant, or moving), the magnitude of
 * the overdensity at which a region collapses, in the same linear units as
 * the void barrier. Reached when -delta_walk <= -collapse, i.e. delta >=
 * delta_c. NULL for the single void barrier.
 * @param n_paths Number of walks, strictly positive
 * @param seed Seed for the walk ensemble
 * @param collapse_counts Optional output, n_radii entries in the ascending
 * order of radii: the walks that reached `collapse` first, at each radius, and
 * were discarded there. All zero without a collapse barrier. Pass NULL to skip.
 * @param opt Reserved; pass SIF_DEFAULT
 *
 * @return Newly allocated array of n_radii void counts in the ascending order
 * of radii, released with sif_free_aligned, or NULL on invalid input or a
 * failed factorisation.
 */
SIF_NODISCARD uint64_t* sif_ep_first_crossing_counts(const sif_real* radii,
  uint32_t n_radii, const double* cov, const sif_real* barrier,
  const sif_real* collapse, uint64_t n_paths, uint64_t seed,
  uint64_t* collapse_counts, sif_option opt);

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
 * @note With a `collapse` barrier, f is the multiplicity of voids that are
 * not inside a cloud; the walks a cloud crushed are in neither the numerator
 * nor anywhere else, so f integrates to less than without it.
 *
 * @param counts Optional output, n_radii entries: the raw first-crossing
 * counts this multiplicity was built from, exactly as
 * sif_ep_first_crossing_counts would have returned them. Pass NULL to skip.
 *
 * Remaining parameters (`collapse` included), and the barrier and
 * reproducibility conventions, are as sif_ep_first_crossing_counts.
 *
 * @return Newly allocated array of n_radii - 1 values, released with
 * sif_free_aligned, or NULL on invalid input or a failed factorisation.
 */
SIF_NODISCARD sif_real* sif_ep_multiplicity_function(const sif_real* radii,
  uint32_t n_radii, const double* cov, const sif_real* barrier,
  const sif_real* collapse, uint64_t n_paths, uint64_t seed, uint64_t* counts,
  sif_option opt);

/**
 * @brief Report on where an emulated call sat relative to what the emulator
 * was trained over. Optional; pass NULL to sif_ep_multiplicity_function_emu if
 * the answer is all that is wanted.
 *
 * Leaving the trained region is not an error and does not fail the call. It
 * costs accuracy -- `expected_error` carries the measured figure either way --
 * and a sampler should be able to notice that without parsing the log or
 * aborting a chain over one proposal.
 */
typedef struct {
  /* Zero if either check below fired. */
  int in_domain;

  /* Bins with at least one input outside the range the fit covered. Bins
   * above the trained nu are not counted: nothing crosses there, so a grid
   * started further out than the training walks is not out of domain. */
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
 * The up-crossing rate of Verza et al. (2024) eq. (3.15) corrected by a small
 * trained network. The correction multiplies a hazard rather than the
 * multiplicity itself, and the result is rebuilt through the survival
 * recursion, so it is non-negative and integrates to at most one whatever the
 * network predicts.
 *
 * @note Takes `sigma`, not the packed covariance that
 * sif_ep_multiplicity_function needs. That is deliberate rather than an
 * oversight: the emulator reads only the diagonal, so it needs n numbers where
 * the Monte Carlo needs n(n+1)/2, and its cost is linear rather than quadratic
 * in the radius count.
 *
 * @note Returns the answer the Monte Carlo CONVERGES to, not the answer it
 * gives on the caller's grid. Those differ: a first-crossing walk sampled at 65
 * radii sits about 0.1% below its converged value, at 129 within 0.05%. The
 * emulated result hardly depends on the sampling: the same problem on 65, 129
 * and 257 radii agrees to 0.02%.
 *
 * @note Void-in-cloud is a factor on top: without `collapse` the result is
 * the single-barrier one, bit for bit, and compares with the Monte Carlo run
 * without one.
 *
 * @note No cosmological parameter enters: the cosmology is seen only through
 * sigma and deriv_variance, i.e. through nu = B / sigma along the walk and the
 * shape of P(k). Trained over CDM-like spectra and Sheth-Mo-Tormen barriers
 * (constant barriers included), for walks starting anywhere from ~2 to
 * 250 Mpc/h -- the largest radius of the grid, placed as described under
 * `radii`. Other spectral families and barrier shapes are outside the
 * contract. The `domain` report catches the ones far off (a strong
 * warm-dark-matter cutoff, a barrier rising with sigma) but not every mild
 * one: a power-law spectrum or a weak cutoff can pass 0.25-0.3% wrong, since
 * what sets them apart is the shape of the whole trajectory rather than any
 * pointwise value.
 *
 * @param radii Smoothing radii, strictly positive and strictly increasing, at
 * least 4. Place the largest where nu = B / sigma is ~3.5 or more: few walks
 * then start above the barrier, and the answer does not depend on exactly
 * where the grid starts (further out costs nothing). Keep nu above ~0.21 at
 * the smallest. `domain` reports a start too low and an end too deep.
 * @param n_radii Number of radii
 * @param sigma R.m.s. density contrast per radius, strictly positive. Take it
 * from sif_delta_covariance_pk so the barrier and the walk share one variance.
 * @param barrier n_radii barrier heights, in the ascending order of radii,
 * **strictly positive** -- stricter than the Monte Carlo, which takes any
 * finite value. The features here are built from log(B / sigma), and the
 * correction was trained over nu from 0.21 upward, so the convention is a
 * positive barrier crossed from below. That is what sif_ep_barrier_smt()
 * produces; a void threshold enters through its magnitude, as it does in the
 * BBKS routines.
 * @param deriv_variance n_radii entries: <(d delta / dS)^2> from
 * sif_delta_covariance_pk. REQUIRED, unlike the Monte Carlo path, which has no
 * use for it: differencing it off a covariance converges only at first order,
 * which would make the answer depend on how finely `radii` was sampled -- the
 * one property this entry point exists to avoid.
 * @param collapse Optional, n_radii collapse thresholds delta_c (constant or
 * moving), strictly positive, as sif_ep_first_crossing_counts takes them:
 * with it the result is the multiplicity of voids NOT inside a cloud, the
 * single-barrier answer times an emulated void-in-cloud factor K <= 1. NULL
 * for the single void barrier.
 * @param cov The packed covariance between the radii, as
 * sif_delta_covariance_pk returns it alongside sigma. Required with
 * `collapse`, ignored without: the void-in-cloud baseline conditions on how a
 * walk ending on the void barrier was correlated with its larger scales, which
 * the diagonal alone cannot say. O(n_radii^2) work, under a millisecond at
 * 128 radii.
 * @param domain Optional report; pass NULL to skip
 * @param opt SIF_EP_EMU_AT_RADII (default): the first-crossing density per
 * unit R at each of the n_radii radii. SIF_EP_EMU_BIN_CENTRES: its average
 * over each bin, n_radii - 1 values on the bin centres -- the form a Monte
 * Carlo histogram measures, and so what sif_ep_multiplicity_function returns;
 * use it to compare the two. The forms agree to second order in the spacing
 * (0.01-0.06% at 129 radii).
 *
 * @return Newly allocated array of n_radii values (or n_radii - 1 under
 * SIF_EP_EMU_BIN_CENTRES), released with sif_free_aligned, or NULL on invalid
 * input. An input outside the trained region is NOT invalid: the answer is
 * returned, `domain` records it, and the log names the quantity responsible.
 */
SIF_NODISCARD sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, const sif_real* collapse, const double* cov,
  sif_emu_domain_t* domain, sif_option opt);

#endif /* SIF_MODEL_EXCURSION_SET_H */
