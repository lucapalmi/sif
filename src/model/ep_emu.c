/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The emulated multiplicity function: the semi-analytic up-crossing baseline,
 * corrected per bin by a small trained network.
 *
 * The baseline and the eight features below are shared verbatim with the code
 * the network was fitted against (tools/ep_model.py). That is the whole
 * correctness argument for this file: a feature computed a little differently
 * here is not an approximation of the trained model, it is a different model.
 * tests/test_ep_emu.c pins every one of them against stored values from the
 * Python reference.
 *
 * Costs a few tens of microseconds against seconds for the Monte Carlo in
 * excursion_set.c, and returns the answer that Monte Carlo CONVERGES to rather
 * than the one it gives on the caller's grid.
 */

#include "sif/model/excursion_set.h"

#include "math/nn.h"
#include "model/ep_emu_weights.h"
#include "model/ep_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

#define TAG "ep_emu"

/* Column order of the feature matrix. Must match ep_emu_weights.h, which
 * records the same list as the order the network was fitted in. */
enum {
  F_NU = 0,
  F_GAMMA2,
  F_Y,
  F_DLNNU_DLNS,
  F_NU_BACK,
  F_LAG,
  F_CUM_LAM,
  F_NU_ORIGIN,
  F_COUNT
};

/* The lookback: nu where the variance was half its current value. */
#define EP_EMU_LOOKBACK 0.5

/* --- Validation --- */

static int validate(const sif_real* radii, uint32_t n_radii,
  const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance) {

  if (!radii || !sigma || !barrier || !deriv_variance) {
    SIF_LOG_ERROR(TAG,
      "radii, sigma, barrier and deriv_variance are all required; the "
      "derivative variance has no fallback here, since differencing it off a "
      "covariance converges only at first order and would make the result "
      "depend on how finely the radii were sampled");
    return SIF_ERR_INVALID;
  }

  /*
   * Four, not three. The three-point stencil runs over the BIN CENTRES, of
   * which there are n_radii - 1, so three radii leave it two points and it
   * reads off the end. The Python the network was fitted with has the same
   * floor -- numpy's gradient with edge_order=2 refuses two points -- so this
   * is the smallest grid the model is defined on, not merely the smallest one
   * this implementation happens to accept.
   */
  if (n_radii < 4) {
    SIF_LOG_ERROR(TAG,
      "%u radii; the local description differentiates over a three-point "
      "stencil on the %u bin centres, so it needs at least four radii",
      n_radii, n_radii > 0 ? n_radii - 1 : 0);
    return SIF_ERR_INVALID;
  }

  if (n_radii > SIF_COV_MAX_RADII) {
    SIF_LOG_ERROR(
      TAG, "%u radii exceeds the maximum of %d", n_radii, SIF_COV_MAX_RADII);
    return SIF_ERR_INVALID;
  }

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(
        TAG, "radius %u is %g, must be strictly positive", i, (double)radii[i]);
      return SIF_ERR_INVALID;
    }
    if (i > 0 && !(radii[i] > radii[i - 1])) {
      SIF_LOG_ERROR(TAG,
        "radii are not strictly increasing at index %u (%g after %g)", i,
        (double)radii[i], (double)radii[i - 1]);
      return SIF_ERR_INVALID;
    }
    if (!(sigma[i] > 0.0f)) {
      SIF_LOG_ERROR(TAG,
        "sigma at radius %u (%g) is %g, must be strictly "
        "positive",
        i, (double)radii[i], (double)sigma[i]);
      return SIF_ERR_INVALID;
    }
    if (!(deriv_variance[i] > 0.0)) {
      SIF_LOG_ERROR(TAG,
        "the derivative variance at radius %u (%g) is %g, must be strictly "
        "positive",
        i, (double)radii[i], deriv_variance[i]);
      return SIF_ERR_INVALID;
    }
    if (!isfinite((double)barrier[i])) {
      SIF_LOG_ERROR(TAG, "the barrier at radius %u (%g) is not finite", i,
        (double)radii[i]);
      return SIF_ERR_INVALID;
    }
  }

  return SIF_OK;
}

/* --- The features --- */

/*
 * df/dx on a non-uniform grid, second order, matching numpy's gradient with
 * edge_order=2 -- which is what the trainer used, so the stencil is part of
 * the model rather than a choice made here.
 */
static double grad(const double* fv, const double* x, uint32_t n, uint32_t i) {

  if (i == 0) {
    const double h1 = x[1] - x[0];
    const double h2 = x[2] - x[1];
    return -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * fv[0] +
           (h1 + h2) / (h1 * h2) * fv[1] - h1 / (h2 * (h1 + h2)) * fv[2];
  }

  if (i + 1 == n) {
    const double h1 = x[n - 2] - x[n - 3];
    const double h2 = x[n - 1] - x[n - 2];
    return h2 / (h1 * (h1 + h2)) * fv[n - 3] -
           (h1 + h2) / (h1 * h2) * fv[n - 2] +
           (h1 + 2.0 * h2) / (h2 * (h1 + h2)) * fv[n - 1];
  }

  const double h1 = x[i] - x[i - 1];
  const double h2 = x[i + 1] - x[i];
  return (h1 * h1 * fv[i + 1] - h2 * h2 * fv[i - 1] -
           (h1 * h1 - h2 * h2) * fv[i]) /
         (h1 * h2 * (h1 + h2));
}

/*
 * Linear interpolation of `v` at `t`, where `x` DESCENDS with the index.
 * numpy's interp wants ascending abscissae and the trainer reversed both
 * arrays to get it; walking down from the top here is the same thing without
 * the copies.
 */
static double interp_desc(
  const double* x, const double* v, uint32_t n, double t) {

  if (t >= x[0])
    return v[0];
  if (t <= x[n - 1])
    return v[n - 1];

  uint32_t lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if (x[mid] >= t)
      lo = mid;
    else
      hi = mid;
  }

  const double span = x[lo] - x[hi];
  const double w = (span != 0.0) ? (t - x[hi]) / span : 0.0;
  return v[hi] + w * (v[lo] - v[hi]);
}

/*
 * The eight inputs, on the n - 1 bin centres, row-major.
 *
 * Every one of these is a property of the walk rather than of the grid, which
 * is what lets the result be read on any radius sampling: the emulated answer
 * moves by under 0.25% between 64 and 256 radii.
 */
static int features(
  const sif_ep_features_t* f, uint32_t n, const double* lam, double* X) {

  const uint32_t n_bins = n - 1;

  /* Bin centres of the per-radius quantities, and of S itself. */
  double* S_mid = malloc((size_t)n_bins * sizeof(double));
  double* nu_mid = malloc((size_t)n_bins * sizeof(double));
  double* ln_nu = malloc((size_t)n_bins * sizeof(double));
  double* ln_S = malloc((size_t)n_bins * sizeof(double));
  double* cum_lam = malloc((size_t)n_bins * sizeof(double));

  if (!S_mid || !nu_mid || !ln_nu || !ln_S || !cum_lam) {
    SIF_LOG_ERROR(TAG, "failed to allocate the feature workspace");
    free(S_mid);
    free(nu_mid);
    free(ln_nu);
    free(ln_S);
    free(cum_lam);
    return SIF_ERR_ALLOC;
  }

  for (uint32_t i = 0; i < n_bins; i++) {
    S_mid[i] = 0.5 * (f->S[i] + f->S[i + 1]);
    nu_mid[i] = 0.5 * (f->nu[i] + f->nu[i + 1]);
    ln_nu[i] = log(nu_mid[i]);
    ln_S[i] = log(S_mid[i]);
  }

  /* Hazard accumulated BEFORE entering each bin: the walk enters at the
   * largest radius (highest index) and works down, so this is the reverse
   * cumulative sum, excluding the bin itself. It is the one number that
   * summarizes the whole prior history, and it is what the surviving
   * population is conditioned on. */
  double cum = 0.0;
  for (int32_t i = (int32_t)n_bins - 1; i >= 0; i--) {
    cum_lam[i] = cum;
    cum += lam[i];
  }

  const double S_origin = f->S[n - 1];
  const double nu_origin = nu_mid[n_bins - 1];

  for (uint32_t i = 0; i < n_bins; i++) {
    double* row = X + (size_t)i * F_COUNT;

    double target = EP_EMU_LOOKBACK * S_mid[i];
    if (target < S_origin)
      target = S_origin;

    row[F_NU] = nu_mid[i];
    row[F_GAMMA2] = 0.5 * (f->gamma2[i] + f->gamma2[i + 1]);
    row[F_Y] = 0.5 * (f->y[i] + f->y[i + 1]);
    row[F_DLNNU_DLNS] = grad(ln_nu, ln_S, n_bins, i);
    row[F_NU_BACK] = interp_desc(S_mid, nu_mid, n_bins, target);
    row[F_LAG] = log(S_mid[i] / target);
    row[F_CUM_LAM] = cum_lam[i];
    row[F_NU_ORIGIN] = nu_origin;
  }

  free(cum_lam);
  free(S_mid);
  free(nu_mid);
  free(ln_nu);
  free(ln_S);

  return SIF_OK;
}

/* --- The domain report --- */

static void report_domain(
  const double* X, uint32_t n_bins, double nu_large, sif_emu_domain_t* domain) {

  const double* lo = SIF_EP_EMU_BOX_LO;
  const double* hi = SIF_EP_EMU_BOX_HI;
  static const char* const names[F_COUNT] = {"nu", "gamma2", "y", "dlnnu_dlnS",
    "nu_back", "lag", "cum_lam", "nu_origin"};

  const double first_step = sif__ep_upper_tail(nu_large);

  uint32_t n_outside = 0;
  uint32_t worst = 0;
  double worst_excess = 0.0;

  for (uint32_t i = 0; i < n_bins; i++) {
    const double* row = X + (size_t)i * F_COUNT;
    int out = 0;
    for (uint32_t j = 0; j < F_COUNT; j++) {
      const double span = (hi[j] - lo[j]) > 0.0 ? (hi[j] - lo[j]) : 1.0;
      const double excess =
        (row[j] < lo[j]) ? (lo[j] - row[j]) / span
                         : ((row[j] > hi[j]) ? (row[j] - hi[j]) / span : 0.0);
      if (excess > 0.0)
        out = 1;
      if (excess > worst_excess) {
        worst_excess = excess;
        worst = j;
      }
    }
    n_outside += (uint32_t)out;
  }

  const int origin_low = nu_large < SIF_EP_EMU_NU_ORIGIN_MIN;

  if (origin_low) {
    SIF_LOG_WARNING(TAG,
      "nu at the largest radius is %.3f, below the %.3f the correction was "
      "trained over: %.2f%% of walks start above the barrier and never enter "
      "any bin. Extend the radius grid outward; no amount of training would "
      "fix this one",
      nu_large, (double)SIF_EP_EMU_NU_ORIGIN_MIN, 100.0 * first_step);
  }

  if (n_outside > 0) {
    SIF_LOG_WARNING(TAG,
      "%u of %u bins fall outside the region the correction was fitted over; "
      "%s is the furthest out, by %.1f%% of its trained span. The baseline is "
      "still corrected, but the accuracy below is the measured out-of-domain "
      "figure rather than the in-domain one",
      n_outside, n_bins, names[worst], 100.0 * worst_excess);
  }

  if (domain) {
    domain->in_domain = (!origin_low && n_outside == 0) ? 1 : 0;
    domain->n_bins_outside = n_outside;
    domain->nu_origin = (sif_real)nu_large;
    domain->first_step_mass = (sif_real)first_step;
    domain->expected_error =
      (sif_real)(domain->in_domain ? SIF_EP_EMU_ERROR_IN_DOMAIN
                                   : SIF_EP_EMU_ERROR_OUT_DOMAIN);
  }
}

/* --- The entry point --- */

sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, sif_emu_domain_t* domain, sif_option opt) {

  (void)opt; /* reserved */

  if (domain) {
    domain->in_domain = 0;
    domain->n_bins_outside = 0;
    domain->nu_origin = 0.0f;
    domain->first_step_mass = 0.0f;
    domain->expected_error = 0.0f;
  }

  if (validate(radii, n_radii, sigma, barrier, deriv_variance) != SIF_OK)
    return NULL;

  const uint32_t n = n_radii;
  const uint32_t n_bins = n - 1;

  sif_ep_features_t f;
  if (sif__ep_features_init(&f, n) != SIF_OK) {
    sif__ep_features_free(&f);
    return NULL;
  }

  double* S = malloc((size_t)n * sizeof(double));
  double* lam = malloc((size_t)n_bins * sizeof(double));
  double* X = malloc((size_t)n_bins * F_COUNT * sizeof(double));
  double* corr = malloc((size_t)n_bins * sizeof(double));
  sif_real* out = sif_calloc_aligned((size_t)n_bins, sizeof(sif_real));

  if (!S || !lam || !X || !corr || !out) {
    SIF_LOG_ERROR(TAG, "failed to allocate the emulator workspace");
    goto fail;
  }

  for (uint32_t i = 0; i < n; i++)
    S[i] = (double)sigma[i] * (double)sigma[i];

  if (sif__ep_features_fill_diag(&f, radii, n, S, barrier, deriv_variance) !=
      SIF_OK)
    goto fail;

  sif__ep_hazard_bins(&f, n, lam);

  if (features(&f, n, lam, X) != SIF_OK)
    goto fail;

  if (sif__nn_eval(&SIF_EP_EMU_NN, X, n_bins, corr) != SIF_OK)
    goto fail;

  const double nu_large = f.nu[n - 1];
  report_domain(X, n_bins, nu_large, domain);

  /* The correction multiplies the HAZARD. Everything downstream of this line
   * is bounded by construction: 1 - exp(-Lambda) stays in [0, 1] however
   * wrong the network is, so the multiplicity cannot come out negative and
   * cannot integrate above one. */
  for (uint32_t i = 0; i < n_bins; i++) {
    const double scaled = lam[i] * exp(corr[i]);
    lam[i] = isfinite(scaled) && scaled > 0.0 ? scaled : 0.0;
  }

  sif__ep_survival(lam, radii, n_bins, 1.0 - sif__ep_upper_tail(nu_large), out);

  free(S);
  free(lam);
  free(X);
  free(corr);
  sif__ep_features_free(&f);
  return out;

fail:
  free(S);
  free(lam);
  free(X);
  free(corr);
  sif_free_aligned(out);
  sif__ep_features_free(&f);
  return NULL;
}
