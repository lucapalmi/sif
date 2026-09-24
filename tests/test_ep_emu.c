/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The emulated multiplicity function.
 *
 * Two kinds of check, and the first is the one that matters. A trained model
 * has no closed form to compare against, so the correctness argument is that
 * this C reproduces the Python it was fitted with -- exactly, not
 * approximately. Stored reference cases pin the whole chain: the up-crossing
 * baseline, the log-linear hazard, all eight features, the network, and the
 * survival recursion. A stencil that differs in its last term, or a lookback
 * that clamps differently, changes the answer by far more than the tolerance
 * here and by far less than any physical invariant would notice.
 *
 * The rest are properties that must hold whatever the network predicts: the
 * result is a probability density and cannot go negative or integrate above
 * one, it must not depend on how finely the radii were sampled, and the domain
 * report must fire on the two conditions it exists for.
 */

#include "sif/core/system.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/utils/align.h"
#include "test_util.h"

#include "ep_emu_cases.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* sif_real holds about seven significant digits, so a result stored in it can
 * differ from the double the Python produced by that much and no more. This
 * tolerance is float rounding, not a physics allowance: the measured agreement
 * is 5e-8, and anything at 1e-5 or worse is a real divergence. */
#define REF_TOL 5e-7

static sif_real* as_real(const double* v, uint32_t n) {
  sif_real* out = malloc((size_t)n * sizeof(sif_real));
  for (uint32_t i = 0; i < n; i++)
    out[i] = (sif_real)v[i];
  return out;
}

/* --- 1. against the Python reference --- */

static void test_against_reference(void) {
  printf("against the Python model the network was fitted with\n");

  for (int c = 0; c < EP_EMU_N_CASES; c++) {
    const uint32_t n = EP_EMU_CASE_N;
    const uint32_t n_bins = n - 1;

    sif_real* radii = as_real(EP_EMU_RADII[c], n);
    sif_real* sigma = as_real(EP_EMU_SIGMA[c], n);
    sif_real* barrier = as_real(EP_EMU_BARRIER[c], n);

    sif_emu_domain_t dom;
    sif_real* f = sif_ep_multiplicity_function_emu(
      radii, n, sigma, barrier, EP_EMU_DVAR[c], &dom, SIF_DEFAULT);

    CHECK(f != NULL, "case %d returned NULL on a valid input", c);

    if (f) {
      double worst = 0.0;
      uint32_t at = 0;
      for (uint32_t i = 0; i < n_bins; i++) {
        const double want = EP_EMU_EXPECT[c][i];
        const double scale = fabs(want) > 1e-300 ? fabs(want) : 1.0;
        const double rel = fabs((double)f[i] - want) / scale;
        if (rel > worst) {
          worst = rel;
          at = i;
        }
      }

      CHECK(worst < REF_TOL,
        "case %d departs from the Python model by a relative %.3e at bin %u; "
        "at this size the cause is an implementation difference, not rounding",
        c, worst, at);

      CHECK(dom.in_domain == EP_EMU_IN_DOMAIN[c],
        "case %d reports in_domain=%d, the reference says %d", c, dom.in_domain,
        EP_EMU_IN_DOMAIN[c]);

      const double nu_rel =
        fabs((double)dom.nu_origin - EP_EMU_NU_ORIGIN[c]) / EP_EMU_NU_ORIGIN[c];
      CHECK(nu_rel < 1e-6, "case %d reports nu_origin %.6f, expected %.6f", c,
        (double)dom.nu_origin, EP_EMU_NU_ORIGIN[c]);

      printf("  case %d: worst relative departure %.2e at bin %u, "
             "in_domain %d\n",
        c, worst, at, dom.in_domain);
      sif_free_aligned(f);
    }

    free(radii);
    free(sigma);
    free(barrier);
  }
}

/* --- 2. what must hold whatever the network says --- */

static void test_invariants(void) {
  printf("the result is a probability density\n");

  for (int c = 0; c < EP_EMU_N_CASES; c++) {
    const uint32_t n = EP_EMU_CASE_N;

    sif_real* radii = as_real(EP_EMU_RADII[c], n);
    sif_real* sigma = as_real(EP_EMU_SIGMA[c], n);
    sif_real* barrier = as_real(EP_EMU_BARRIER[c], n);

    sif_real* f = sif_ep_multiplicity_function_emu(
      radii, n, sigma, barrier, EP_EMU_DVAR[c], NULL, SIF_DEFAULT);

    if (f) {
      int negative = 0, nonfinite = 0;
      double integral = 0.0;
      for (uint32_t i = 0; i + 1 < n; i++) {
        if (f[i] < 0.0f)
          negative = 1;
        if (!isfinite((double)f[i]))
          nonfinite = 1;
        integral += (double)f[i] * ((double)radii[i + 1] - (double)radii[i]);
      }
      CHECK(!negative, "case %d went negative", c);
      CHECK(!nonfinite, "case %d produced a non-finite value", c);
      CHECK(integral <= 1.0,
        "case %d integrates to %.6f, above the one path per path it can "
        "represent",
        c, integral);
      printf("  case %d: crossing fraction %.4f\n", c, integral);
      sif_free_aligned(f);
    }

    free(radii);
    free(sigma);
    free(barrier);
  }
}

/*
 * The emulator predicts a continuum quantity, so refining the radius grid must
 * not move it. This is the property the whole feature design rests on -- every
 * input is a property of the field rather than of the sampling -- and nothing
 * in the fit enforced it, which is what makes it worth asserting.
 *
 * A power-law spectrum is used so the covariance is exact and self-similar,
 * and the same physical problem is read at a radius common to all three grids.
 */
#define N_K 4000

static void test_grid_independence(void) {
  printf("independence of the radius sampling\n");

  sif_real* k = malloc(N_K * sizeof(sif_real));
  sif_real* pk = malloc(N_K * sizeof(sif_real));
  const double lo = log(1e-6), hi = log(1e4);
  for (uint32_t i = 0; i < N_K; i++) {
    const double lk = lo + (hi - lo) * i / (N_K - 1.0);
    k[i] = (sif_real)exp(lk);
    pk[i] = (sif_real)pow(exp(lk), -2.0);
  }

  /* Normalize to sigma_8 = 0.8, so nu lands where the emulator was trained
   * rather than wherever an arbitrary amplitude happens to put it. Without
   * this the multiplicity is zero on every grid and the comparison below
   * passes by comparing nothing. */
  {
    const sif_real eight[1] = {8.0f};
    sif_real s8[1];
    double* c = sif_delta_covariance_pk(
      k, pk, N_K, eight, 1, s8, NULL, NULL, SIF_DEFAULT);
    CHECK(c != NULL && s8[0] > 0.0f, "could not normalize the spectrum");
    if (c) {
      const double scale = (0.8 / (double)s8[0]) * (0.8 / (double)s8[0]);
      for (uint32_t i = 0; i < N_K; i++)
        pk[i] = (sif_real)((double)pk[i] * scale);
      sif_free_aligned(c);
    }
  }

  const uint32_t counts[3] = {48, 96, 192};
  const double r_lo = 1.0, r_hi = 30.0;
  const double r_probe = 6.0;
  double value[3] = {0.0, 0.0, 0.0};

  for (int g = 0; g < 3; g++) {
    const uint32_t n = counts[g];
    sif_real* radii = malloc(n * sizeof(sif_real));
    sif_real* sigma = malloc(n * sizeof(sif_real));
    double* dvar = malloc(n * sizeof(double));

    for (uint32_t i = 0; i < n; i++)
      radii[i] = (sif_real)(r_lo * pow(r_hi / r_lo, (double)i / (n - 1.0)));

    double* cov = sif_delta_covariance_pk(
      k, pk, N_K, radii, n, sigma, NULL, dvar, SIF_DEFAULT);
    CHECK(cov != NULL, "the covariance returned NULL at %u radii", n);

    if (cov) {
      sif_real* barrier = sif_ep_barrier_smt(sigma, n, 0.5f, 0.3f, 0.8f);
      sif_real* f = barrier ? sif_ep_multiplicity_function_emu(radii, n, sigma,
                                barrier, dvar, NULL, SIF_DEFAULT)
                            : NULL;
      CHECK(f != NULL, "the emulator returned NULL at %u radii", n);

      if (f) {
        /* Linear interpolation onto the probe radius, on bin centres. */
        for (uint32_t i = 0; i + 2 < n; i++) {
          const double a = 0.5 * ((double)radii[i] + (double)radii[i + 1]);
          const double b = 0.5 * ((double)radii[i + 1] + (double)radii[i + 2]);
          if (a <= r_probe && r_probe <= b) {
            const double w = (r_probe - a) / (b - a);
            value[g] = (1.0 - w) * (double)f[i] + w * (double)f[i + 1];
            break;
          }
        }
        sif_free_aligned(f);
      }
      sif_free_aligned(barrier);
      sif_free_aligned(cov);
    }

    free(radii);
    free(sigma);
    free(dvar);
  }

  double vmin = value[0], vmax = value[0];
  for (int g = 1; g < 3; g++) {
    if (value[g] < vmin)
      vmin = value[g];
    if (value[g] > vmax)
      vmax = value[g];
  }
  /* A zero multiplicity on every grid would give a zero spread and pass this
   * test by comparing nothing at all. */
  CHECK(vmin > 0.0,
    "the multiplicity is zero at the probe radius on at least one grid, so "
    "the comparison below is vacuous");

  const double spread = (vmax > 0.0) ? (vmax - vmin) / vmax : 1.0;

  /* The Monte Carlo itself moves by about a per cent between 50 and 100
   * radii. Anything approaching that here means a feature has picked up a
   * dependence on the grid. */
  CHECK(spread < 0.01,
    "the result moved by %.3f%% between 48 and 192 radii; a feature is "
    "measuring the grid rather than the field",
    100.0 * spread);
  printf("  f(48) %.6e, f(96) %.6e, f(192) %.6e, spread %.3f%%\n", value[0],
    value[1], value[2], 100.0 * spread);

  free(k);
  free(pk);
}

/* --- 3. the domain report --- */

static void test_domain(void) {
  printf("the domain report\n");

  const uint32_t n = EP_EMU_CASE_N;
  sif_real* radii = as_real(EP_EMU_RADII[0], n);
  sif_real* sigma = as_real(EP_EMU_SIGMA[0], n);
  sif_real* barrier = as_real(EP_EMU_BARRIER[0], n);

  sif_emu_domain_t dom;
  sif_real* f = sif_ep_multiplicity_function_emu(
    radii, n, sigma, barrier, EP_EMU_DVAR[0], &dom, SIF_DEFAULT);
  CHECK(f && dom.in_domain == 1, "a training-like input was not in domain");
  CHECK(dom.first_step_mass < 0.01f,
    "the walk origin of a training-like input reports %.3f of its paths "
    "starting above the barrier",
    (double)dom.first_step_mass);
  CHECK(dom.expected_error > 0.0f && dom.expected_error < 0.01f,
    "the in-domain accuracy is reported as %g", (double)dom.expected_error);
  sif_free_aligned(f);

  /*
   * Truncating the grid from the outside leaves the walk starting where the
   * barrier is close to the field, so a real share of paths begin above it.
   * That is the one failure the caller can fix, and the one the report exists
   * to name.
   */
  {
    const uint32_t cut = n / 2;
    sif_real* f2 = sif_ep_multiplicity_function_emu(
      radii, cut, sigma, barrier, EP_EMU_DVAR[0], &dom, SIF_DEFAULT);
    CHECK(f2 != NULL, "a truncated grid was rejected rather than reported");
    CHECK(dom.in_domain == 0,
      "a grid cut to %u radii was reported as in domain, with %.2f%% of walks "
      "starting above the barrier",
      cut, 100.0 * (double)dom.first_step_mass);
    printf("  grid cut to %u radii: nu_origin %.3f, first-step mass %.2e, "
           "in_domain %d\n",
      cut, (double)dom.nu_origin, (double)dom.first_step_mass, dom.in_domain);
    sif_free_aligned(f2);
  }

  /* A barrier far above anything trained: extrapolation, reported per bin. */
  {
    sif_real* tall = malloc((size_t)n * sizeof(sif_real));
    for (uint32_t i = 0; i < n; i++)
      tall[i] = barrier[i] * 4.0f;

    sif_real* f3 = sif_ep_multiplicity_function_emu(
      radii, n, sigma, tall, EP_EMU_DVAR[0], &dom, SIF_DEFAULT);
    CHECK(f3 != NULL, "an out-of-domain barrier was rejected rather than "
                      "reported");
    CHECK(dom.in_domain == 0 && dom.n_bins_outside > 0,
      "a barrier four times the trained scale was reported as in domain");
    printf("  barrier x4: %u of %u bins outside, expected error %.3f%%\n",
      dom.n_bins_outside, n - 1, 100.0 * (double)dom.expected_error);
    sif_free_aligned(f3);
    free(tall);
  }

  free(radii);
  free(sigma);
  free(barrier);
}

/* --- 4. invalid input --- */

static void test_guards(void) {
  printf("input validation\n");

  const uint32_t n = EP_EMU_CASE_N;
  sif_real* radii = as_real(EP_EMU_RADII[0], n);
  sif_real* sigma = as_real(EP_EMU_SIGMA[0], n);
  sif_real* barrier = as_real(EP_EMU_BARRIER[0], n);
  const double* dvar = EP_EMU_DVAR[0];
  sif_real* r;

  r = sif_ep_multiplicity_function_emu(
    NULL, n, sigma, barrier, dvar, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "NULL radii were accepted");
  sif_free_aligned(r);

  r = sif_ep_multiplicity_function_emu(
    radii, n, NULL, barrier, dvar, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL sigma was accepted");
  sif_free_aligned(r);

  r = sif_ep_multiplicity_function_emu(
    radii, n, sigma, barrier, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "a NULL derivative variance was accepted; it has no fallback here");
  sif_free_aligned(r);

  r = sif_ep_multiplicity_function_emu(
    radii, 2, sigma, barrier, dvar, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "two radii were accepted by a three-point stencil");
  sif_free_aligned(r);

  /*
   * Three radii leave only two bin centres, and the stencil runs on those --
   * it reads off the end of them. Caught by AddressSanitizer, not by any
   * result looking wrong, which is why the boundary is asserted here rather
   * than left to the reference cases (all of which are 128 radii).
   */
  r = sif_ep_multiplicity_function_emu(
    radii, 3, sigma, barrier, dvar, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "three radii were accepted, leaving the three-point stencil two bin "
    "centres to work with");
  sif_free_aligned(r);

  /* Four is the smallest grid the model is defined on, and must work. */
  r = sif_ep_multiplicity_function_emu(
    radii, 4, sigma, barrier, dvar, NULL, SIF_DEFAULT);
  CHECK(r != NULL, "four radii were rejected; that is the documented minimum");
  if (r) {
    int ok = 1;
    for (uint32_t i = 0; i < 3; i++)
      if (!(r[i] >= 0.0f) || !isfinite((double)r[i]))
        ok = 0;
    CHECK(ok, "the smallest legal grid produced a negative or non-finite bin");
  }
  sif_free_aligned(r);

  {
    sif_real* descending = malloc((size_t)n * sizeof(sif_real));
    for (uint32_t i = 0; i < n; i++)
      descending[i] = radii[n - 1 - i];
    r = sif_ep_multiplicity_function_emu(
      descending, n, sigma, barrier, dvar, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "descending radii were accepted");
    sif_free_aligned(r);
    free(descending);
  }

  {
    sif_real* bad = malloc((size_t)n * sizeof(sif_real));
    memcpy(bad, sigma, (size_t)n * sizeof(sif_real));
    bad[3] = 0.0f;
    r = sif_ep_multiplicity_function_emu(
      radii, n, bad, barrier, dvar, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a zero sigma was accepted");
    sif_free_aligned(r);
    free(bad);
  }

  printf("  all guards returned NULL\n");

  free(radii);
  free(sigma);
  free(barrier);
}

int main(void) {
  sif_config_t cfg = {
    .fft_config = NULL, .omp_config = NULL, .log_level = SIF_LOG_LEVEL_ERROR};
  if (sif_init(&cfg) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

  test_against_reference();
  test_invariants();
  test_grid_independence();
  test_domain();
  test_guards();

  sif_finalize();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall emulator checks passed\n");
  return 0;
}
