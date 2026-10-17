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
      radii, n, sigma, barrier, EP_EMU_DVAR[c], NULL, NULL, &dom, SIF_DEFAULT);
    sif_real* fb = sif_ep_multiplicity_function_emu(radii, n, sigma, barrier,
      EP_EMU_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);

    CHECK(f != NULL && fb != NULL, "case %d returned NULL on a valid input", c);

    if (f && fb) {
      /* Both forms: n values at the radii, n - 1 bin averages. */
      double worst = 0.0;
      uint32_t at = 0;
      for (uint32_t i = 0; i < n + n_bins; i++) {
        const int at_radii = i < n;
        const double want =
          at_radii ? EP_EMU_EXPECT[c][i] : EP_EMU_EXPECT_BINNED[c][i - n];
        const double got = at_radii ? (double)f[i] : (double)fb[i - n];
        const double scale = fabs(want) > 1e-300 ? fabs(want) : 1.0;
        const double rel = fabs(got - want) / scale;
        if (rel > worst) {
          worst = rel;
          at = i;
        }
      }

      CHECK(worst < REF_TOL,
        "case %d departs from the Python model by a relative %.3e at %s %u; "
        "at this size the cause is an implementation difference, not rounding",
        c, worst, at < n ? "radius" : "bin", at < n ? at : at - n);

      CHECK(dom.in_domain == EP_EMU_IN_DOMAIN[c],
        "case %d reports in_domain=%d, the reference says %d", c, dom.in_domain,
        EP_EMU_IN_DOMAIN[c]);

      const double nu_rel =
        fabs((double)dom.nu_origin - EP_EMU_NU_ORIGIN[c]) / EP_EMU_NU_ORIGIN[c];
      CHECK(nu_rel < 1e-6, "case %d reports nu_origin %.6f, expected %.6f", c,
        (double)dom.nu_origin, EP_EMU_NU_ORIGIN[c]);

      printf("  case %d: worst relative departure %.2e (radii and bins), "
             "in_domain %d\n",
        c, worst, dom.in_domain);
    }
    sif_free_aligned(f);
    sif_free_aligned(fb);

    free(radii);
    free(sigma);
    free(barrier);
  }
}

/* --- 1b. void-in-cloud, against the Python reference --- */

static void test_vic_reference(void) {
  printf("void-in-cloud against the Python model\n");

  for (int c = 0; c < EP_VIC_N_CASES; c++) {
    const uint32_t n = EP_VIC_CASE_N;
    const uint32_t n_bins = n - 1;

    sif_real* radii = as_real(EP_VIC_RADII[c], n);
    sif_real* sigma = as_real(EP_VIC_SIGMA[c], n);
    sif_real* barrier = as_real(EP_VIC_BARRIER[c], n);
    sif_real* collapse = as_real(EP_VIC_COLLAPSE[c], n);

    sif_real* f = sif_ep_multiplicity_function_emu(radii, n, sigma, barrier,
      EP_VIC_DVAR[c], collapse, EP_VIC_COV[c], NULL, SIF_DEFAULT);
    sif_real* fb = sif_ep_multiplicity_function_emu(radii, n, sigma, barrier,
      EP_VIC_DVAR[c], collapse, EP_VIC_COV[c], NULL, SIF_EP_EMU_BIN_CENTRES);
    sif_real* f1 = sif_ep_multiplicity_function_emu(radii, n, sigma, barrier,
      EP_VIC_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);

    CHECK(f && fb && f1, "void-in-cloud case %d returned NULL", c);

    if (f && fb && f1) {
      double worst = 0.0, k_min = 1.0, k_max = 0.0;
      for (uint32_t i = 0; i < n + n_bins; i++) {
        const int at_radii = i < n;
        const double want =
          at_radii ? EP_VIC_EXPECT[c][i] : EP_VIC_EXPECT_BINNED[c][i - n];
        const double got = at_radii ? (double)f[i] : (double)fb[i - n];
        /* Below 1e-30 a sif_real is subnormal or nearly so, and its few
         * remaining bits say nothing about the model. */
        if (fabs(want) < 1e-30)
          continue;
        const double rel = fabs(got - want) / fabs(want);
        if (rel > worst)
          worst = rel;
      }
      /* K = f with collapse / f without: never above one. */
      for (uint32_t i = 0; i < n_bins; i++) {
        if (!(f1[i] > 0.0f))
          continue;
        const double k = (double)fb[i] / (double)f1[i];
        k_min = k < k_min ? k : k_min;
        k_max = k > k_max ? k : k_max;
      }
      CHECK(worst < REF_TOL,
        "void-in-cloud case %d departs from the Python model by %.3e", c,
        worst);
      CHECK(k_max <= 1.0 + 1e-6, "void-in-cloud raised f: K = %.8f", k_max);
      printf("  case %d: worst relative departure %.2e, K %.4f to %.4f\n", c,
        worst, k_min, k_max);
    }
    sif_free_aligned(f);
    sif_free_aligned(fb);
    sif_free_aligned(f1);
    free(radii);
    free(sigma);
    free(barrier);
    free(collapse);
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

    /* The bound is structural in the binned form: each bin holds a
     * probability, alive * (1 - exp(-Lambda)), so they cannot sum past one.
     * The density at the radii is the same function sampled at points; it
     * must be non-negative and finite, and integrate to the same total. */
    sif_real* fb = sif_ep_multiplicity_function_emu(radii, n, sigma, barrier,
      EP_EMU_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);
    sif_real* f = sif_ep_multiplicity_function_emu(
      radii, n, sigma, barrier, EP_EMU_DVAR[c], NULL, NULL, NULL, SIF_DEFAULT);

    if (f && fb) {
      int negative = 0, nonfinite = 0;
      double integral = 0.0, trapezoid = 0.0;
      for (uint32_t i = 0; i < n; i++) {
        if (f[i] < 0.0f || (i + 1 < n && fb[i] < 0.0f))
          negative = 1;
        if (!isfinite((double)f[i]) || (i + 1 < n && !isfinite((double)fb[i])))
          nonfinite = 1;
        if (i + 1 < n) {
          const double dr = (double)radii[i + 1] - (double)radii[i];
          integral += (double)fb[i] * dr;
          trapezoid += 0.5 * ((double)f[i] + (double)f[i + 1]) * dr;
        }
      }
      CHECK(!negative, "case %d went negative", c);
      CHECK(!nonfinite, "case %d produced a non-finite value", c);
      CHECK(integral <= 1.0,
        "case %d integrates to %.6f, above the one path per path it can "
        "represent",
        c, integral);
      CHECK(fabs(trapezoid / integral - 1.0) < 1e-3,
        "case %d: the density at the radii integrates to %.6f, the bins to "
        "%.6f",
        c, trapezoid, integral);
      printf("  case %d: crossing fraction %.4f (at the radii: %.4f)\n", c,
        integral, trapezoid);
    }
    sif_free_aligned(f);
    sif_free_aligned(fb);

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
    CHECK(c != NULL && s8[0] > 0.0f, "could not normalise the spectrum");
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
                                barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT)
                            : NULL;
      CHECK(f != NULL, "the emulator returned NULL at %u radii", n);

      if (f) {
        /* Linear interpolation onto the probe radius, between radii. */
        for (uint32_t i = 0; i + 1 < n; i++) {
          const double a = (double)radii[i];
          const double b = (double)radii[i + 1];
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
    radii, n, sigma, barrier, EP_EMU_DVAR[0], NULL, NULL, &dom, SIF_DEFAULT);
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
    sif_real* f2 = sif_ep_multiplicity_function_emu(radii, cut, sigma, barrier,
      EP_EMU_DVAR[0], NULL, NULL, &dom, SIF_DEFAULT);
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

  /*
   * A barrier four times higher: the walk starts at nu ~ 18, far beyond any
   * training walk. Above nu ~ 3.5 nothing has crossed, so that is the same
   * walk started further out, and the bins where nothing crosses are not a gap
   * in what the model knows. Not flagged.
   */
  {
    sif_real* tall = malloc((size_t)n * sizeof(sif_real));
    for (uint32_t i = 0; i < n; i++)
      tall[i] = barrier[i] * 4.0f;

    sif_real* f3 = sif_ep_multiplicity_function_emu(
      radii, n, sigma, tall, EP_EMU_DVAR[0], NULL, NULL, &dom, SIF_DEFAULT);
    CHECK(f3 && dom.in_domain == 1,
      "a walk started beyond the trained origins was flagged (%u bins)",
      dom.n_bins_outside);
    printf("  barrier x4 (origin nu %.1f): in_domain %d\n",
      (double)dom.nu_origin, dom.in_domain);
    sif_free_aligned(f3);
    free(tall);
  }

  /*
   * Half the derivative variance doubles gamma^2, past anything a CDM-like
   * spectrum produces: a spectral shape the network never saw. Extrapolation,
   * reported per bin, and the answer still returned.
   */
  {
    double* dvar = malloc((size_t)n * sizeof(double));
    for (uint32_t i = 0; i < n; i++)
      dvar[i] = 0.5 * EP_EMU_DVAR[0][i];

    sif_real* f4 = sif_ep_multiplicity_function_emu(
      radii, n, sigma, barrier, dvar, NULL, NULL, &dom, SIF_DEFAULT);
    CHECK(f4 != NULL, "an out-of-domain spectrum was rejected rather than "
                      "reported");
    CHECK(dom.in_domain == 0 && dom.n_bins_outside > 0,
      "gamma^2 twice the trained range was reported as in domain");
    printf("  gamma^2 x2: %u of %u bins outside, expected error %.3f%%\n",
      dom.n_bins_outside, n - 1, 100.0 * (double)dom.expected_error);
    sif_free_aligned(f4);
    free(dvar);
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
    NULL, n, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "NULL radii were accepted");
  sif_free_aligned(r);

  /* Void-in-cloud: the collapse barrier needs the covariance, and is a
   * positive magnitude. */
  {
    const uint32_t m = EP_VIC_CASE_N;
    sif_real* vr = as_real(EP_VIC_RADII[0], m);
    sif_real* vs = as_real(EP_VIC_SIGMA[0], m);
    sif_real* vb = as_real(EP_VIC_BARRIER[0], m);
    sif_real* vc = as_real(EP_VIC_COLLAPSE[0], m);
    r = sif_ep_multiplicity_function_emu(
      vr, m, vs, vb, EP_VIC_DVAR[0], vc, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a collapse barrier without cov was accepted");
    sif_free_aligned(r);
    vc[m / 2] = 0.0f;
    r = sif_ep_multiplicity_function_emu(
      vr, m, vs, vb, EP_VIC_DVAR[0], vc, EP_VIC_COV[0], NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a zero collapse barrier was accepted");
    sif_free_aligned(r);
    free(vr);
    free(vs);
    free(vb);
    free(vc);
  }

  r = sif_ep_multiplicity_function_emu(
    radii, n, NULL, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL sigma was accepted");
  sif_free_aligned(r);

  r = sif_ep_multiplicity_function_emu(
    radii, n, sigma, barrier, NULL, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "a NULL derivative variance was accepted; it has no fallback here");
  sif_free_aligned(r);

  r = sif_ep_multiplicity_function_emu(
    radii, 2, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "two radii were accepted by a three-point stencil");
  sif_free_aligned(r);

  /*
   * Three radii leave only two bin centres, and the stencil runs on those --
   * it reads off the end of them. Caught by AddressSanitizer, not by any
   * result looking wrong, which is why the boundary is asserted here rather
   * than left to the reference cases (all of which are 128 radii).
   */
  r = sif_ep_multiplicity_function_emu(
    radii, 3, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "three radii were accepted, leaving the three-point stencil two bin "
    "centres to work with");
  sif_free_aligned(r);

  /* Four is the smallest grid the model is defined on, and must work. */
  r = sif_ep_multiplicity_function_emu(
    radii, 4, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
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
      descending, n, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "descending radii were accepted");
    sif_free_aligned(r);
    free(descending);
  }

  {
    sif_real* bad = malloc((size_t)n * sizeof(sif_real));
    memcpy(bad, sigma, (size_t)n * sizeof(sif_real));
    bad[3] = 0.0f;
    r = sif_ep_multiplicity_function_emu(
      radii, n, bad, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
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
  if (sif_system_init(&cfg) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  test_against_reference();
  test_vic_reference();
  test_invariants();
  test_grid_independence();
  test_domain();
  test_guards();

  sif_system_finalise();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall emulator checks passed\n");
  return 0;
}
