/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The emulated multiplicity function: the semi-analytic up-crossing baseline,
 * corrected per bin by a small trained network.
 *
 * The baseline and the nine features below are shared verbatim with the code
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
#include "model/ep_emu_vic_weights.h"
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
  F_GAMMA2_BACK,
  F_COUNT
};

_Static_assert(F_COUNT == SIF_EP_EMU_N_IN,
  "ep_emu.c computes a different number of features than the generated "
  "weights expect; regenerate ep_emu_weights.h with tools/ep_export_c.py");

/* The void-in-cloud candidates: the nine above, then the collapse side, in the
 * order of VIC_CANDIDATES in tools/ep_model.py. The trained network takes the
 * ones SIF_EP_VIC_INPUTS lists. */
enum {
  V_NU_C = F_COUNT,
  V_NU_C_BACK,
  V_DLNNUC_DLNS,
  V_Y_C,
  V_LOG_CUM_LAM_C,
  V_LOG_LAM_VALUE,
  V_LOG_LAM_COND,
  V_COUNT
};

_Static_assert(V_COUNT == SIF_EP_VIC_N_CAND,
  "ep_emu.c computes a different list of void-in-cloud candidates than the "
  "generated weights index; regenerate ep_emu_vic_weights.h");

/* As in ep_model.py: logs of hazards are floored, and the network output is
 * capped before the exponential, as it was in the fit. */
#define VIC_LOG_FLOOR 1e-30
#define VIC_Z_MAX     10.0

/* The lookback: nu and gamma^2 where the variance was half its current
 * value. */
#define EP_EMU_LOOKBACK 0.5

/* Ceiling on the log-correction before it is exponentiated. The trained
 * correction lives within a factor of a few of one; e^40 is far outside
 * anything the network produces on real input and still nowhere near
 * overflowing a double, so this bounds a runaway without touching any answer
 * the model actually gives. */
#define EP_EMU_CORR_MAX 40.0

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
    /*
     * Strictly positive, not merely finite. The features take log(nu) with
     * nu = B / sigma, so a non-positive barrier produces a NaN that travels
     * through the network and out into the multiplicity without tripping
     * anything. The convention here is a positive barrier crossed from below,
     * which is all sif_ep_barrier_smt() ever produces and the only regime the
     * correction was trained over -- its nu box starts at 0.23.
     *
     * The Monte Carlo in excursion_set.c is deliberately looser: it only ever
     * compares the walk against the barrier, so any sign is meaningful there.
     */
    if (!(barrier[i] > 0.0f)) {
      SIF_LOG_ERROR(TAG,
        "the barrier at radius %u (%g) is %g; the emulator needs a strictly "
        "positive barrier, since its features are built from log(B / sigma)",
        i, (double)radii[i], (double)barrier[i]);
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

/* The collapse barrier, positive like the void barrier, and the covariance
 * its baseline needs. */
static int validate_collapse(const sif_real* radii, uint32_t n,
  const sif_real* collapse, const double* cov) {

  if (!cov) {
    SIF_LOG_ERROR(TAG,
      "collapse given without cov: the void-in-cloud baseline conditions on "
      "the covariance between scales (pass the packed cov from "
      "sif_delta_covariance_pk)");
    return SIF_ERR_INVALID;
  }
  for (uint32_t i = 0; i < n; i++) {
    if (!(collapse[i] > 0.0f && isfinite((double)collapse[i]))) {
      SIF_LOG_ERROR(TAG,
        "collapse barrier at radius %u (%g): %g (expected positive and "
        "finite)",
        i, (double)radii[i], (double)collapse[i]);
      return SIF_ERR_INVALID;
    }
    if (!(cov[SIF_COV_INDEX(i, i)] > 0.0)) {
      SIF_LOG_ERROR(TAG,
        "cov diagonal at radius %u (%g): %g (expected positive)", i,
        (double)radii[i], cov[SIF_COV_INDEX(i, i)]);
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
 * The nine inputs, on the n - 1 bin centres, row-major.
 *
 * Every one of these is a property of the walk rather than of the grid, which
 * is what lets the result be read on any radius sampling. Two first versions
 * were not -- cum_lam stopped at the bin's upper edge and nu_origin was read at
 * the outermost bin centre -- and the answer moved by 0.5% between 65 and 513
 * radii. Taken at the bin centre and at the largest radius, it moves by under
 * 0.05% once the spacing is d ln S <~ 0.02.
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
  double* g2_mid = malloc((size_t)n_bins * sizeof(double));

  if (!S_mid || !nu_mid || !ln_nu || !ln_S || !cum_lam || !g2_mid) {
    SIF_LOG_ERROR(TAG, "failed to allocate the feature workspace");
    free(S_mid);
    free(nu_mid);
    free(ln_nu);
    free(ln_S);
    free(cum_lam);
    free(g2_mid);
    return SIF_ERR_ALLOC;
  }

  for (uint32_t i = 0; i < n_bins; i++) {
    S_mid[i] = 0.5 * (f->S[i] + f->S[i + 1]);
    nu_mid[i] = 0.5 * (f->nu[i] + f->nu[i + 1]);
    g2_mid[i] = 0.5 * (f->gamma2[i] + f->gamma2[i + 1]);
    ln_nu[i] = log(nu_mid[i]);
    ln_S[i] = log(S_mid[i]);
  }

  /* Hazard accumulated up to each bin's CENTRE: everything above the bin plus
   * half the bin. The walk enters at the largest radius (highest index) and
   * works down, so this is a reverse cumulative sum. Measured at the centre,
   * where the other features are; stopping at the bin's upper edge made it,
   * and through it the answer, depend on the bin width. */
  double cum = 0.0;
  for (int32_t i = (int32_t)n_bins - 1; i >= 0; i--) {
    cum_lam[i] = cum + 0.5 * lam[i];
    cum += lam[i];
  }

  /* The lookback is clamped at the walk's origin, the largest radius. */
  const double S_origin = f->S[n - 1];

  /* nu at the largest RADIUS, where the walk starts. The outermost bin
   * centre sits half a grid step inside it and so moved with the sampling. */
  const double nu_origin = f->nu[n - 1];

  for (uint32_t i = 0; i < n_bins; i++) {
    double* row = X + (size_t)i * F_COUNT;

    double target = EP_EMU_LOOKBACK * S_mid[i];
    if (target < S_origin)
      target = S_origin;

    row[F_NU] = nu_mid[i];
    row[F_GAMMA2] = g2_mid[i];
    row[F_Y] = 0.5 * (f->y[i] + f->y[i + 1]);
    row[F_DLNNU_DLNS] = grad(ln_nu, ln_S, n_bins, i);
    row[F_NU_BACK] = interp_desc(S_mid, nu_mid, n_bins, target);
    row[F_LAG] = log(S_mid[i] / target);
    row[F_CUM_LAM] = cum_lam[i];
    row[F_NU_ORIGIN] = nu_origin;
    /* gamma^2 peaks near R ~ 10 Mpc/h, so its local value is reached from two
     * sides; where it was half a variance ago says which side the walk came
     * from. */
    row[F_GAMMA2_BACK] = interp_desc(S_mid, g2_mid, n_bins, target);
  }

  free(g2_mid);
  free(cum_lam);
  free(S_mid);
  free(nu_mid);
  free(ln_nu);
  free(ln_S);

  return SIF_OK;
}

/* --- The domain report --- */

static void report_domain(const double* X, const sif_real* radii,
  uint32_t n_bins, double nu_large, sif_emu_domain_t* domain) {

  const double* lo = SIF_EP_EMU_BOX_LO;
  const double* hi = SIF_EP_EMU_BOX_HI;
  static const char* const names[F_COUNT] = {"nu", "gamma2", "y", "dlnnu_dlnS",
    "nu_back", "lag", "cum_lam", "nu_origin", "gamma2_back"};

  const double first_step = sif__ep_upper_tail(nu_large);

  uint32_t n_outside = 0;
  uint32_t worst = 0;
  double worst_excess = 0.0;
  double worst_value = 0.0;
  uint32_t first_out = 0, last_out = 0;

  for (uint32_t i = 0; i < n_bins; i++) {
    const double* row = X + (size_t)i * F_COUNT;
    /* Above the trained nu nothing crosses (less than 1e-8 of the walks), so
     * whatever the correction does there cannot reach the answer: the network
     * was measured to stay within a few per cent of one out to nu = 50. Such
     * bins are where a grid started further out than training walks; they are
     * not a gap in what the model knows. */
    if (row[F_NU] > hi[F_NU])
      continue;
    int out = 0;
    for (uint32_t j = 0; j < F_COUNT; j++) {
      /* The upper edges of nu_back and nu_origin are where training walks
       * started, not a limit of the physics: nu_origin is held at it before
       * the network (see the caller), and nu_back passes it only in bins at
       * nu > 4, where under 3e-5 of the walks cross. The low end of
       * nu_origin is the origin check below, reported once, there. */
      if (j == F_NU_ORIGIN)
        continue; /* its own check below; held at the maximum above */
      const int no_upper = (j == F_NU_BACK);
      const double span = (hi[j] - lo[j]) > 0.0 ? (hi[j] - lo[j]) : 1.0;
      const double excess =
        (row[j] < lo[j])
          ? (lo[j] - row[j]) / span
          : ((row[j] > hi[j] && !no_upper) ? (row[j] - hi[j]) / span : 0.0);
      if (excess > 0.0)
        out = 1;
      if (excess > worst_excess) {
        worst_excess = excess;
        worst = j;
        worst_value = row[j];
      }
    }
    if (out) {
      if (n_outside == 0)
        first_out = i;
      last_out = i;
      n_outside++;
    }
  }

  const int origin_low = nu_large < SIF_EP_EMU_NU_ORIGIN_MIN;

  if (origin_low) {
    SIF_LOG_WARNING(TAG,
      "walk origin too low: nu at R = %g is %.3f, %.2f%% of walks start above "
      "the barrier (expected nu >= %.3f; extend the radii outward)",
      (double)radii[n_bins], nu_large, 100.0 * first_step,
      (double)SIF_EP_EMU_NU_ORIGIN_MIN);
  }

  if (n_outside > 0) {
    SIF_LOG_WARNING(TAG,
      "%u of %u bins outside the trained range, R = %g to %g: %s = %.4g "
      "(trained %.4g to %.4g)",
      n_outside, n_bins, (double)radii[first_out], (double)radii[last_out + 1],
      names[worst], worst_value, lo[worst], hi[worst]);
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

/* --- The density at the radii --- */

/*
 * The first-crossing density per unit R AT each radius, n values:
 *
 *     f(R_i) = r(S_i) * alive(S_i) * |dS/dR|_i
 *
 * with r = f_up * exp(correction) the corrected rate per unit S, alive the
 * fraction of walks not yet crossed -- exact at a grid point, from the same
 * recursion as the bins -- and dS/dR second order in ln S over ln R. The
 * correction is fitted per bin; it is smooth and O(1), so it is carried from
 * the bin centres to the radii linearly in ln S, extrapolated over the half
 * bin at either end. tools/ep_model.py (multiplicity_at_radii) is the
 * reference.
 *
 * `lam` holds the CORRECTED per-bin hazards, `corr` the raw network output.
 */
static int density_at_radii(const sif_ep_features_t* f, const sif_real* radii,
  uint32_t n, const double* corr, const double* lam, const double* ln_k,
  double alive_origin, sif_real* out) {

  const uint32_t n_bins = n - 1;
  double* ln_mid = malloc((size_t)n_bins * sizeof(double));
  double* ln_R = malloc((size_t)n * sizeof(double));
  double* ln_S = malloc((size_t)n * sizeof(double));

  if (!ln_mid || !ln_R || !ln_S) {
    SIF_LOG_ERROR(TAG, "failed to allocate the density workspace");
    free(ln_mid);
    free(ln_R);
    free(ln_S);
    return SIF_ERR_ALLOC;
  }

  for (uint32_t i = 0; i < n_bins; i++)
    ln_mid[i] = log(0.5 * (f->S[i] + f->S[i + 1]));
  for (uint32_t i = 0; i < n; i++) {
    ln_R[i] = log((double)radii[i]);
    ln_S[i] = log(f->S[i]);
  }

  double alive = alive_origin;

  for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
    const uint32_t k = (uint32_t)i;
    const double ln_s = ln_S[k];

    /* Radius k sits between bin centres k (smaller S) and k - 1 (larger S);
     * the first and last radii extrapolate from the two nearest centres. */
    uint32_t a, b;
    if (k == 0) {
      a = 1;
      b = 0;
    } else if (k == n - 1) {
      a = n_bins - 1;
      b = n_bins - 2;
    } else {
      a = k;
      b = k - 1;
    }
    const double c_raw = corr[a] + (corr[b] - corr[a]) /
                                     (ln_mid[b] - ln_mid[a]) *
                                     (ln_s - ln_mid[a]);
    const double c = c_raw < EP_EMU_CORR_MAX ? c_raw : EP_EMU_CORR_MAX;

    if (k < n - 1)
      alive *= exp(-lam[k]);

    /* dS/dR through the logarithms: S is close to a power law in R, and
     * differenced directly it was 0.5% off at 48 radii; this way, 0.007%. */
    const double dS_dR =
      f->S[k] / (double)radii[k] * fabs(grad(ln_S, ln_R, n, k));
    double v = f->f_up[k] * exp(c) * alive * dS_dR;
    if (ln_k) {
      /* Void-in-cloud, carried to the radius as the correction is, and held
       * at or below zero: an extrapolated end must not make K exceed one. */
      const double lk = ln_k[a] + (ln_k[b] - ln_k[a]) /
                                    (ln_mid[b] - ln_mid[a]) *
                                    (ln_s - ln_mid[a]);
      v *= exp(lk < 0.0 ? lk : 0.0);
    }
    out[k] = (sif_real)(v > 0.0 ? v : 0.0);
  }

  free(ln_mid);
  free(ln_R);
  free(ln_S);
  return SIF_OK;
}

/* --- Void-in-cloud --- */

/*
 * ln K per bin: the fraction of voids whose walk was not inside a cloud at a
 * larger radius, emulated as -Lambda_cond exp(z), with Lambda_cond the
 * slope-conditioned collapse hazard (ep_vic.c) and z the network's output.
 * X holds the nine single-barrier features, as computed -- before the
 * nu_origin hold, as the trainer saw them. Counts the bins whose inputs fall
 * outside the trained region into *n_outside and names the furthest.
 */
static int vic_ln_k(const sif_real* radii, uint32_t n, const double* S,
  const sif_real* barrier, const sif_real* collapse,
  const double* deriv_variance, const double* cov, const double* X,
  double* ln_k, uint32_t* n_outside) {

  const uint32_t n_bins = n - 1;
  int status = SIF_ERR_ALLOC;

  sif_ep_features_t fc;
  if (sif__ep_features_init(&fc, n) != SIF_OK) {
    sif__ep_features_free(&fc);
    return SIF_ERR_ALLOC;
  }

  double* Bd = malloc((size_t)n * sizeof(double));
  double* Cd = malloc((size_t)n * sizeof(double));
  double* hz = malloc((size_t)n * sizeof(double));
  double* hv = malloc((size_t)n * sizeof(double));
  double* lam_c = malloc((size_t)n_bins * sizeof(double));
  double* S_mid = malloc((size_t)n_bins * sizeof(double));
  double* ln_S = malloc((size_t)n_bins * sizeof(double));
  double* nuc = malloc((size_t)n_bins * sizeof(double));
  double* ln_nuc = malloc((size_t)n_bins * sizeof(double));
  double* cand = malloc((size_t)n_bins * V_COUNT * sizeof(double));
  double* Xv = malloc((size_t)n_bins * SIF_EP_VIC_N_IN * sizeof(double));
  double* z = malloc((size_t)n_bins * sizeof(double));

  if (!Bd || !Cd || !hz || !hv || !lam_c || !S_mid || !ln_S || !nuc ||
      !ln_nuc || !cand || !Xv || !z) {
    SIF_LOG_ERROR(TAG, "failed to allocate the void-in-cloud workspace");
    goto done;
  }

  for (uint32_t i = 0; i < n; i++) {
    Bd[i] = (double)barrier[i];
    Cd[i] = (double)collapse[i];
  }

  status =
    sif__ep_features_fill_diag(&fc, radii, n, S, collapse, deriv_variance);
  if (status != SIF_OK)
    goto done;
  sif__ep_hazard_bins(&fc, n, lam_c);

  status = sif__ep_vic_hazard(cov, S, Bd, Cd, deriv_variance, n, hz);
  if (status != SIF_OK)
    goto done;

  /* The value-only baseline is a second O(n^2) pass; only if the network
   * takes it. */
  int want_value = 0;
  for (uint32_t k = 0; k < SIF_EP_VIC_N_IN; k++)
    want_value |= SIF_EP_VIC_INPUTS[k] == V_LOG_LAM_VALUE;
  if (want_value) {
    status = sif__ep_vic_hazard_value(cov, S, Bd, Cd, deriv_variance, n, hv);
    if (status != SIF_OK)
      goto done;
  } else {
    for (uint32_t i = 0; i < n; i++)
      hv[i] = 0.0;
  }

  for (uint32_t i = 0; i < n_bins; i++) {
    S_mid[i] = 0.5 * (S[i] + S[i + 1]);
    ln_S[i] = log(S_mid[i]);
    nuc[i] = 0.5 * (fc.nu[i] + fc.nu[i + 1]);
    ln_nuc[i] = log(nuc[i]);
  }

  /* Collapse hazard accumulated to each bin's centre, as cum_lam is. */
  double cum = 0.0;
  for (int32_t i = (int32_t)n_bins - 1; i >= 0; i--) {
    double* row = cand + (size_t)i * V_COUNT;
    const double c = cum + 0.5 * lam_c[i];
    cum += lam_c[i];
    row[V_LOG_CUM_LAM_C] = log(c > VIC_LOG_FLOOR ? c : VIC_LOG_FLOOR);
  }

  for (uint32_t i = 0; i < n_bins; i++) {
    double* row = cand + (size_t)i * V_COUNT;
    for (uint32_t j = 0; j < F_COUNT; j++)
      row[j] = X[(size_t)i * F_COUNT + j];
    double target = EP_EMU_LOOKBACK * S_mid[i];
    if (target < S[n - 1])
      target = S[n - 1];
    row[V_NU_C] = nuc[i];
    row[V_NU_C_BACK] = interp_desc(S_mid, nuc, n_bins, target);
    row[V_DLNNUC_DLNS] = grad(ln_nuc, ln_S, n_bins, i);
    row[V_Y_C] = 0.5 * (fc.y[i] + fc.y[i + 1]);
    const double lv = 0.5 * (hv[i] + hv[i + 1]);
    const double lc = 0.5 * (hz[i] + hz[i + 1]);
    row[V_LOG_LAM_VALUE] = log(lv > VIC_LOG_FLOOR ? lv : VIC_LOG_FLOOR);
    row[V_LOG_LAM_COND] = log(lc > VIC_LOG_FLOOR ? lc : VIC_LOG_FLOOR);
    for (uint32_t k = 0; k < SIF_EP_VIC_N_IN; k++)
      Xv[(size_t)i * SIF_EP_VIC_N_IN + k] = row[SIF_EP_VIC_INPUTS[k]];
  }

  status = sif__nn_eval(&SIF_EP_VIC_NN, Xv, n_bins, z);
  if (status != SIF_OK)
    goto done;

  /* Bins where the baseline sees no cloud have K = 1 exactly, whatever the
   * inputs, and bins above the trained nu nothing crosses: neither is a gap
   * in what the network knows. */
  uint32_t outside = 0, worst = 0;
  double worst_excess = 0.0, worst_value = 0.0;
  for (uint32_t i = 0; i < n_bins; i++) {
    const double lc = 0.5 * (hz[i] + hz[i + 1]);
    const double zz = z[i] < VIC_Z_MAX ? z[i] : VIC_Z_MAX;
    ln_k[i] = -lc * exp(zz);
    if (!(lc > 0.0) || X[(size_t)i * F_COUNT + F_NU] > SIF_EP_EMU_BOX_HI[F_NU])
      continue;
    int out = 0;
    for (uint32_t k = 0; k < SIF_EP_VIC_N_IN; k++) {
      const double v = Xv[(size_t)i * SIF_EP_VIC_N_IN + k];
      const double lo = SIF_EP_VIC_BOX_LO[k], hi = SIF_EP_VIC_BOX_HI[k];
      const double span = hi - lo > 0.0 ? hi - lo : 1.0;
      const double ex =
        v < lo ? (lo - v) / span : (v > hi ? (v - hi) / span : 0.0);
      if (ex > 0.0)
        out = 1;
      if (ex > worst_excess) {
        worst_excess = ex;
        worst = k;
        worst_value = v;
      }
    }
    outside += (uint32_t)out;
  }
  if (outside > 0) {
    SIF_LOG_WARNING(TAG,
      "%u of %u bins outside the void-in-cloud training range: input %u = "
      "%.4g (trained %.4g to %.4g)",
      outside, n_bins, worst, worst_value, SIF_EP_VIC_BOX_LO[worst],
      SIF_EP_VIC_BOX_HI[worst]);
  }
  *n_outside = outside;
  status = SIF_OK;

done:
  free(Bd);
  free(Cd);
  free(hz);
  free(hv);
  free(lam_c);
  free(S_mid);
  free(ln_S);
  free(nuc);
  free(ln_nuc);
  free(cand);
  free(Xv);
  free(z);
  sif__ep_features_free(&fc);
  return status;
}

/* --- The entry point --- */

sif_real* sif_ep_multiplicity_function_emu(const sif_real* radii,
  uint32_t n_radii, const sif_real* sigma, const sif_real* barrier,
  const double* deriv_variance, const sif_real* collapse, const double* cov,
  sif_emu_domain_t* domain, sif_option opt) {

  const int binned = (opt & SIF__EP_EMU_OUTPUT_MASK) == SIF_EP_EMU_BIN_CENTRES;

  if (domain) {
    domain->in_domain = 0;
    domain->n_bins_outside = 0;
    domain->nu_origin = 0.0f;
    domain->first_step_mass = 0.0f;
    domain->expected_error = 0.0f;
  }

  if (validate(radii, n_radii, sigma, barrier, deriv_variance) != SIF_OK)
    return NULL;
  if (collapse && validate_collapse(radii, n_radii, collapse, cov) != SIF_OK)
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
  double* ln_k = collapse ? malloc((size_t)n_bins * sizeof(double)) : NULL;
  sif_real* out =
    sif_calloc_aligned((size_t)(binned ? n_bins : n), sizeof(sif_real));

  if (!S || !lam || !X || !corr || !out || (collapse && !ln_k)) {
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

  const double nu_large = f.nu[n - 1];
  report_domain(X, radii, n_bins, nu_large, domain);

  uint32_t vic_outside = 0;
  if (collapse && vic_ln_k(radii, n, S, barrier, collapse, deriv_variance, cov,
                    X, ln_k, &vic_outside) != SIF_OK)
    goto fail;
  if (domain && vic_outside > 0) {
    domain->in_domain = 0;
    domain->n_bins_outside += vic_outside;
    domain->expected_error = (sif_real)SIF_EP_EMU_ERROR_OUT_DOMAIN;
  }

  /* A walk starting further out than any training walk is the same walk: above
   * nu ~ 3.5 nothing has crossed, so the Monte Carlo does not depend on where
   * the grid starts. The network was only shown origins up to the trained
   * maximum, and extrapolated past it moves the answer by up to 0.4% at
   * nu_origin = 8; held at the maximum, by under 0.1%. */
  const double nu_origin_max = SIF_EP_EMU_BOX_HI[F_NU_ORIGIN];
  for (uint32_t i = 0; i < n_bins; i++) {
    double* row = X + (size_t)i * F_COUNT;
    if (row[F_NU_ORIGIN] > nu_origin_max)
      row[F_NU_ORIGIN] = nu_origin_max;
  }

  if (sif__nn_eval(&SIF_EP_EMU_NN, X, n_bins, corr) != SIF_OK)
    goto fail;

  /* The correction multiplies the HAZARD. Everything downstream of this line
   * is bounded by construction: 1 - exp(-Lambda) stays in [0, 1] however
   * wrong the network is, so the multiplicity cannot come out negative and
   * cannot integrate above one. */
  for (uint32_t i = 0; i < n_bins; i++) {
    /* Bounded before the exponential, not after it. The network output is
     * unbounded, and testing the product with isfinite() afterwards is exactly
     * the pattern sif__ep_cholesky avoids: the release build carries
     * -ffast-math, under which the compiler may assume no infinity ever
     * appears and fold the predicate away. A ceiling on the exponent cannot be
     * optimised out, and at e^40 the hazard already saturates the survival to
     * one -- so anything above it is the same answer, reached honestly. */
    const double c = corr[i] < EP_EMU_CORR_MAX ? corr[i] : EP_EMU_CORR_MAX;
    const double scaled = lam[i] * exp(c);

    lam[i] = scaled > 0.0 ? scaled : 0.0;
  }

  const double alive_origin = 1.0 - sif__ep_upper_tail(nu_large);

  if (binned) {
    sif__ep_survival(lam, radii, n_bins, alive_origin, out);
    if (ln_k)
      for (uint32_t i = 0; i < n_bins; i++)
        out[i] = (sif_real)((double)out[i] * exp(ln_k[i]));
  } else if (density_at_radii(
               &f, radii, n, corr, lam, ln_k, alive_origin, out) != SIF_OK)
    goto fail;

  free(S);
  free(lam);
  free(X);
  free(corr);
  free(ln_k);
  sif__ep_features_free(&f);
  return out;

fail:
  free(S);
  free(lam);
  free(X);
  free(corr);
  free(ln_k);
  sif_free_aligned(out);
  sif__ep_features_free(&f);
  return NULL;
}

#undef TAG
