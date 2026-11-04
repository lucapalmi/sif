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
 *
 * Those run on the model at a given grid, the internal entry point. The public
 * one takes a walk from a variance kernel instead, chooses the grid itself and
 * interpolates: the last sections check the kernel against a direct
 * integral, and that the public path is the model on the grid it chose.
 */

#include "sif/core/system.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/model/variance_kernel.h"
#include "sif/utils/align.h"
#include "test_util.h"

#include "model/ep_internal.h"

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
    sif_real* f = sif__ep_multiplicity_function_emu_grid(
      radii, n, sigma, barrier, EP_EMU_DVAR[c], NULL, NULL, &dom, SIF_DEFAULT);
    sif_real* fb = sif__ep_multiplicity_function_emu_grid(radii, n, sigma,
      barrier, EP_EMU_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);

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

    sif_emu_domain_t dom;
    sif_real* f = sif__ep_multiplicity_function_emu_grid(radii, n, sigma,
      barrier, EP_VIC_DVAR[c], collapse, EP_VIC_COV[c], &dom, SIF_DEFAULT);
    sif_real* fb =
      sif__ep_multiplicity_function_emu_grid(radii, n, sigma, barrier,
        EP_VIC_DVAR[c], collapse, EP_VIC_COV[c], NULL, SIF_EP_EMU_BIN_CENTRES);
    sif_real* f1 = sif__ep_multiplicity_function_emu_grid(radii, n, sigma,
      barrier, EP_VIC_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);

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
      /* These grids start at nu_origin well above the trained 5.77. That is
       * the same walk started further out, not a gap in the training: the
       * network holds nu_origin at its maximum and the guard does not flag it
       * (it used to, and K moved by up to 2.7% unclamped). */
      CHECK(dom.in_domain == 1,
        "void-in-cloud case %d (nu_origin %.1f) was flagged out of domain, "
        "%u bins",
        c, (double)dom.nu_origin, dom.n_bins_outside);
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
    sif_real* fb = sif__ep_multiplicity_function_emu_grid(radii, n, sigma,
      barrier, EP_EMU_DVAR[c], NULL, NULL, NULL, SIF_EP_EMU_BIN_CENTRES);
    sif_real* f = sif__ep_multiplicity_function_emu_grid(
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
      sif_real* f = barrier
                      ? sif__ep_multiplicity_function_emu_grid(radii, n, sigma,
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
  sif_real* f = sif__ep_multiplicity_function_emu_grid(
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
    sif_real* f2 = sif__ep_multiplicity_function_emu_grid(radii, cut, sigma,
      barrier, EP_EMU_DVAR[0], NULL, NULL, &dom, SIF_DEFAULT);
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

    sif_real* f3 = sif__ep_multiplicity_function_emu_grid(
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

    sif_real* f4 = sif__ep_multiplicity_function_emu_grid(
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

  r = sif__ep_multiplicity_function_emu_grid(
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
    r = sif__ep_multiplicity_function_emu_grid(
      vr, m, vs, vb, EP_VIC_DVAR[0], vc, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a collapse barrier without cov was accepted");
    sif_free_aligned(r);
    vc[m / 2] = 0.0f;
    r = sif__ep_multiplicity_function_emu_grid(
      vr, m, vs, vb, EP_VIC_DVAR[0], vc, EP_VIC_COV[0], NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a zero collapse barrier was accepted");
    sif_free_aligned(r);
    free(vr);
    free(vs);
    free(vb);
    free(vc);
  }

  r = sif__ep_multiplicity_function_emu_grid(
    radii, n, NULL, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL sigma was accepted");
  sif_free_aligned(r);

  r = sif__ep_multiplicity_function_emu_grid(
    radii, n, sigma, barrier, NULL, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "a NULL derivative variance was accepted; it has no fallback here");
  sif_free_aligned(r);

  r = sif__ep_multiplicity_function_emu_grid(
    radii, 2, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "two radii were accepted by a three-point stencil");
  sif_free_aligned(r);

  /*
   * Three radii leave only two bin centres, and the stencil runs on those --
   * it reads off the end of them. Caught by AddressSanitizer, not by any
   * result looking wrong, which is why the boundary is asserted here rather
   * than left to the reference cases (all of which are 128 radii).
   */
  r = sif__ep_multiplicity_function_emu_grid(
    radii, 3, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL,
    "three radii were accepted, leaving the three-point stencil two bin "
    "centres to work with");
  sif_free_aligned(r);

  /* Four is the smallest grid the model is defined on, and must work. */
  r = sif__ep_multiplicity_function_emu_grid(
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
    r = sif__ep_multiplicity_function_emu_grid(
      descending, n, sigma, barrier, dvar, NULL, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "descending radii were accepted");
    sif_free_aligned(r);
    free(descending);
  }

  {
    sif_real* bad = malloc((size_t)n * sizeof(sif_real));
    memcpy(bad, sigma, (size_t)n * sizeof(sif_real));
    bad[3] = 0.0f;
    r = sif__ep_multiplicity_function_emu_grid(
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

/* --- 5. the variance kernel --- */

/*
 * A smooth CDM-like shape, n_s = 0.96 through the BBKS transfer function,
 * normalised below. Analytic, so the reference integral can sample it as
 * finely as it likes rather than inheriting a table's resolution.
 */
static double cdm_shape(double k) {
  const double q = k / 0.21;
  const double t =
    log(1.0 + 2.34 * q) / (2.34 * q) *
    pow(1.0 + 3.89 * q + pow(16.1 * q, 2) + pow(5.46 * q, 3) + pow(6.71 * q, 4),
      -0.25);
  return pow(k, 0.96) * t * t;
}

#define KERNEL_N_K  600
#define KERNEL_K_LO 1e-4
#define KERNEL_K_HI 100.0

/* sigma_8 = 0.8, from a direct quadrature of the shape: the kernel under test
 * must not be the one setting its own normalisation. */
static double cdm_amplitude(void) {
  const double R = 8.0;
  double s = 0.0, prev = 0.0, prev_x = 0.0;
  const uint32_t n = 400000;
  for (uint32_t i = 0; i < n; i++) {
    const double x =
      log(KERNEL_K_LO) + (log(KERNEL_K_HI) - log(KERNEL_K_LO)) * i / (n - 1.0);
    const double k = exp(x), kr = k * R;
    const double w = 3.0 * (sin(kr) - kr * cos(kr)) / (kr * kr * kr);
    const double v = k * k * k * cdm_shape(k) * w * w;
    if (i > 0)
      s += 0.5 * (v + prev) * (x - prev_x);
    prev = v;
    prev_x = x;
  }
  return 0.64 * 2.0 * M_PI * M_PI / s;
}

/*
 * sigma^2, dS/dR and <(d delta / dR)^2> at one radius by brute force: a
 * trapezoid in k, each step the smaller of 2e-4 in ln k and 1/128 of a period
 * of sin^2(kR), over the kernel's own k range.
 */
static void direct_walk(double amp, double R, double* sigma, double* dvar) {
  double S = 0.0, D1 = 0.0, D2 = 0.0;
  double prev[3] = {0.0, 0.0, 0.0}, prev_k = 0.0;
  const double dlnk = 2e-4;
  const double dk = M_PI / (128.0 * R);
  int first = 1;

  for (double k = KERNEL_K_LO;;) {
    const double kr = k * R;
    const double s = sin(kr), c = cos(kr);
    const double w =
      kr < 1e-2 ? 1.0 - kr * kr / 10.0 : 3.0 * (s - kr * c) / (kr * kr * kr);
    const double wp = kr < 1e-2 ? -kr / 5.0 + kr * kr * kr / 70.0
                                : 3.0 * (kr * kr * s - 3.0 * s + 3.0 * kr * c) /
                                    (kr * kr * kr * kr);
    /* Per unit k: k^2 P / 2 pi^2 times the window terms. */
    const double base = amp * k * k * cdm_shape(k) / (2.0 * M_PI * M_PI);
    const double v[3] = {
      base * w * w, base * 2.0 * k * w * wp, base * k * k * wp * wp};

    if (!first) {
      const double h = k - prev_k;
      S += 0.5 * (v[0] + prev[0]) * h;
      D1 += 0.5 * (v[1] + prev[1]) * h;
      D2 += 0.5 * (v[2] + prev[2]) * h;
    }
    first = 0;
    for (int a = 0; a < 3; a++)
      prev[a] = v[a];
    prev_k = k;

    if (k >= KERNEL_K_HI)
      break;
    double next = k + (k * dlnk < dk ? k * dlnk : dk);
    if (next > KERNEL_K_HI)
      next = KERNEL_K_HI;
    k = next;
  }

  *sigma = sqrt(S);
  *dvar = D2 / (D1 * D1);
}

static void test_kernel(void) {
  printf("the variance kernel against a direct integral\n");

  const double amp = cdm_amplitude();
  sif_real k[KERNEL_N_K], pk[KERNEL_N_K];
  for (uint32_t m = 0; m < KERNEL_N_K; m++) {
    k[m] = (sif_real)(KERNEL_K_LO *
                      pow(KERNEL_K_HI / KERNEL_K_LO, m / (KERNEL_N_K - 1.0)));
    pk[m] = (sif_real)(amp * cdm_shape((double)k[m]));
  }

  sif_variance_kernel_t* kernel =
    sif_variance_kernel_alloc(k, KERNEL_N_K, 1.0f, 200.0f, 0.02f, SIF_DEFAULT);
  CHECK(kernel != NULL, "the kernel returned NULL on a valid input");
  if (!kernel)
    return;

  CHECK(kernel->n_radii == 265 && fabs(kernel->radii[0] - 1.0f) < 1e-6f &&
          kernel->radii[kernel->n_radii - 1] <= 200.0f,
    "lattice 1 to 200 at dln_r 0.02: %u radii ending at %g (expected 265, "
    "at most 200)",
    kernel->n_radii, (double)kernel->radii[kernel->n_radii - 1]);

  sif_walk_t* walk = sif_walk_pk(kernel, pk);
  CHECK(walk != NULL, "the walk returned NULL on a valid spectrum");

  if (walk) {
    /*
     * On this smooth shape 600 nodes reach 1e-7 in the derivative variance
     * and 4e-8 in sigma; a CAMB spectrum, with its baryon wiggles, 3e-6. The
     * bounds leave room for the wiggles, and are still far below the 1.5e-4
     * by which a directly quadratured 8000-point table misses the derivative
     * variance at large R.
     */
    double worst_s = 0.0, worst_v = 0.0;
    const uint32_t probes[] = {0, 35, 80, 115, 160, 200, 264};
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
      const uint32_t j = probes[i];
      double s, v;
      direct_walk(amp, (double)walk->radii[j], &s, &v);
      const double es = fabs((double)walk->sigma[j] / s - 1.0);
      const double ev = fabs(walk->deriv_variance[j] / v - 1.0);
      worst_s = es > worst_s ? es : worst_s;
      worst_v = ev > worst_v ? ev : worst_v;
    }
    CHECK(worst_s < 5e-7, "sigma departs from the direct integral by %.2e",
      worst_s);
    CHECK(worst_v < 2e-6,
      "the derivative variance departs from the direct integral by %.2e",
      worst_v);
    printf("  600 nodes: sigma within %.1e, derivative variance within %.1e\n",
      worst_s, worst_v);

    /* Linear in P: twice the spectrum is twice the variance and leaves the
     * derivative variance, a ratio of two integrals, at a quarter. */
    sif_real twice[KERNEL_N_K];
    for (uint32_t m = 0; m < KERNEL_N_K; m++)
      twice[m] = 2.0f * pk[m];
    sif_walk_t* w2 = sif_walk_pk(kernel, twice);
    if (w2) {
      double worst = 0.0;
      for (uint32_t j = 0; j < walk->n_radii; j++) {
        const double e1 =
          fabs((double)w2->sigma[j] / ((double)walk->sigma[j] * M_SQRT2) - 1.0);
        const double e2 =
          fabs(w2->deriv_variance[j] / (0.5 * walk->deriv_variance[j]) - 1.0);
        worst = e1 > worst ? e1 : worst;
        worst = e2 > worst ? e2 : worst;
      }
      CHECK(worst < 1e-6, "doubling P(k) did not scale the walk: %.2e", worst);
    }
    sif_walk_free(w2);
  }

  /* Guards. */
  sif_variance_kernel_t* bad;
  bad = sif_variance_kernel_alloc(k, 3, 1.0f, 200.0f, 0.02f, SIF_DEFAULT);
  CHECK(bad == NULL, "a kernel with 3 nodes was accepted");
  sif_variance_kernel_free(bad);
  bad = sif_variance_kernel_alloc(k, KERNEL_N_K, 2.0f, 1.0f, 0.02f, 0);
  CHECK(bad == NULL, "r_max below r_min was accepted");
  sif_variance_kernel_free(bad);
  bad = sif_variance_kernel_alloc(k, KERNEL_N_K, 1.0f, 1.05f, 0.02f, 0);
  CHECK(bad == NULL, "a lattice of 3 radii was accepted");
  sif_variance_kernel_free(bad);
  bad = sif_variance_kernel_alloc(k, KERNEL_N_K, 1.0f, 200.0f, 0.0f, 0);
  CHECK(bad == NULL, "dln_r = 0 was accepted");
  sif_variance_kernel_free(bad);
  {
    sif_real kk[KERNEL_N_K];
    memcpy(kk, k, sizeof(kk));
    kk[100] = kk[99];
    bad = sif_variance_kernel_alloc(kk, KERNEL_N_K, 1.0f, 200.0f, 0.02f, 0);
    CHECK(bad == NULL, "repeated nodes were accepted");
    sif_variance_kernel_free(bad);
  }
  {
    sif_real neg[KERNEL_N_K];
    memcpy(neg, pk, sizeof(neg));
    neg[10] = -1.0f;
    sif_walk_t* w = sif_walk_pk(kernel, neg);
    CHECK(w == NULL, "a negative P(k) was accepted");
    sif_walk_free(w);
  }

  sif_walk_free(walk);
  sif_variance_kernel_free(kernel);
}

/* --- 6. the emulator from a walk --- */

static void test_walk_emulator(void) {
  printf("the emulator from a walk\n");

  const double amp = cdm_amplitude();
  sif_real k[KERNEL_N_K], pk[KERNEL_N_K];
  for (uint32_t m = 0; m < KERNEL_N_K; m++) {
    k[m] = (sif_real)(KERNEL_K_LO *
                      pow(KERNEL_K_HI / KERNEL_K_LO, m / (KERNEL_N_K - 1.0)));
    pk[m] = (sif_real)(amp * cdm_shape((double)k[m]));
  }

  sif_variance_kernel_t* kernel =
    sif_variance_kernel_alloc(k, KERNEL_N_K, 1.0f, 300.0f, 0.02f, SIF_DEFAULT);
  sif_walk_t* walk = kernel ? sif_walk_pk(kernel, pk) : NULL;
  CHECK(walk != NULL, "no walk to test the emulator on");
  if (!walk) {
    sif_variance_kernel_free(kernel);
    return;
  }

  const uint32_t n = walk->n_radii;
  /* A thesis-like void barrier: nu from ~0.3 at 10 Mpc/h up. */
  sif_real* barrier = sif_ep_barrier_smt(walk->sigma, n, 0.096f, 1.15f, 0.43f);

  /*
   * At lattice radii the interpolation is the identity, so the answer must be
   * the model on the grid the entry point chose: one point inside the smallest
   * radius out to the first with B / sigma >= 4.5, rebuilt here by hand.
   */
  const uint32_t j_lo = 120, j_hi = 200;
  sif_real at[81];
  for (uint32_t j = j_lo; j <= j_hi; j++)
    at[j - j_lo] = walk->radii[j];

  uint32_t top = j_hi + 1;
  while (top < n - 1 && (double)barrier[top] / (double)walk->sigma[top] < 4.5)
    top++;
  const uint32_t lo = j_lo - 1;

  sif_emu_domain_t dom;
  sif_real* f = sif_ep_multiplicity_function_emu(
    walk, barrier, NULL, at, 81, &dom, SIF_DEFAULT);
  sif_real* g = sif__ep_multiplicity_function_emu_grid(walk->radii + lo,
    top - lo + 1, walk->sigma + lo, barrier + lo, walk->deriv_variance + lo,
    NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(f && g, "the emulator returned NULL on a valid walk");

  if (f && g) {
    double worst = 0.0;
    for (uint32_t i = 0; i < 81; i++) {
      if (!(g[i + 1] > 0.0f))
        continue;
      const double e = fabs((double)f[i] / (double)g[i + 1] - 1.0);
      worst = e > worst ? e : worst;
    }
    CHECK(worst < 1e-5,
      "at lattice radii the walk path departs from the model on its own grid "
      "by %.2e",
      worst);
    /* Not asserted in domain: the BBKS shape peaks at gamma^2 = 0.2586,
     * just past the 0.2535 the CAMB spectra of the training reach. That is
     * the shape, and what is checked here is the path, not the network. */
    CHECK(dom.nu_origin > 4.5f,
      "the walk was started at nu %.2f (expected above 4.5)",
      (double)dom.nu_origin);
    printf("  at lattice radii: %.1e from the model on the chosen grid, "
           "nu_origin %.2f\n",
      worst, (double)dom.nu_origin);
  }
  sif_free_aligned(f);
  sif_free_aligned(g);

  /*
   * Between lattice radii, against a lattice twice as fine: the interpolation
   * and the coarser grid together, on radii neither lattice contains.
   */
  {
    sif_variance_kernel_t* fine_k = sif_variance_kernel_alloc(
      k, KERNEL_N_K, 1.0f, 300.0f, 0.01f, SIF_DEFAULT);
    sif_walk_t* fine = fine_k ? sif_walk_pk(fine_k, pk) : NULL;
    sif_real* fb = fine ? sif_ep_barrier_smt(
                            fine->sigma, fine->n_radii, 0.096f, 1.15f, 0.43f)
                        : NULL;
    sif_real radii[25];
    for (uint32_t i = 0; i < 25; i++)
      radii[i] = (sif_real)(8.3 * pow(60.0 / 8.3, i / 24.0));

    sif_real* coarse_f = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, radii, 25, NULL, SIF_DEFAULT);
    sif_real* fine_f = fb ? sif_ep_multiplicity_function_emu(
                              fine, fb, NULL, radii, 25, NULL, SIF_DEFAULT)
                          : NULL;
    CHECK(coarse_f && fine_f, "the emulator returned NULL between lattices");
    if (coarse_f && fine_f) {
      double worst = 0.0;
      for (uint32_t i = 0; i < 25; i++) {
        const double e = fabs((double)coarse_f[i] / (double)fine_f[i] - 1.0);
        worst = e > worst ? e : worst;
      }
      CHECK(worst < 5e-4,
        "dln_r 0.02 and 0.01 differ by %.2e between lattice radii", worst);
      printf("  between lattice radii, dln_r 0.02 against 0.01: %.1e\n", worst);
    }

    /* Bin averages over the same radii: positive, and between the densities
     * at their two ends, since f falls monotonically over this range. */
    sif_real* bins = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, radii, 25, NULL, SIF_EP_EMU_BIN_CENTRES);
    CHECK(bins != NULL, "bin averages returned NULL");
    if (bins && coarse_f) {
      int ok = 1;
      for (uint32_t i = 0; i < 24; i++) {
        const double a = (double)coarse_f[i], b = (double)coarse_f[i + 1];
        const double lo_v = a < b ? a : b, hi_v = a < b ? b : a;
        if (!((double)bins[i] >= lo_v * (1.0 - 1e-6) &&
              (double)bins[i] <= hi_v * (1.0 + 1e-6)))
          ok = 0;
      }
      CHECK(ok, "a bin average fell outside the densities at its ends");
    }

    sif_free_aligned(bins);
    sif_free_aligned(coarse_f);
    sif_free_aligned(fine_f);
    sif_free_aligned(fb);
    sif_walk_free(fine);
    sif_variance_kernel_free(fine_k);
  }

  /*
   * Void-in-cloud through the walk: the covariance is rebuilt from the walk's
   * spectrum, so compare against the model given one quadratured from the
   * analytic shape directly, on the same grid. At 2 to 10 Mpc/h with a
   * shallow barrier, where sigma is large enough for clouds to matter.
   */
  {
    sif_real* vb = sif_ep_barrier_smt(walk->sigma, n, 0.5f, 0.3f, 0.8f);
    sif_real* collapse = malloc(n * sizeof(sif_real));
    for (uint32_t j = 0; j < n; j++)
      collapse[j] = 1.686f;

    const uint32_t v_lo = 35, n_at = 41;
    sif_real at_vic[41];
    for (uint32_t j = 0; j < n_at; j++)
      at_vic[j] = walk->radii[v_lo + 2 * j];

    const uint32_t lo_v = v_lo - 1;
    uint32_t t = v_lo + 2 * (n_at - 1) + 1;
    while (t < n - 1 && ((double)vb[t] / (double)walk->sigma[t] < 4.5 ||
                          1.686 / (double)walk->sigma[t] < 4.5))
      t++;

    const uint32_t n_tab = 6000;
    sif_real* kt = malloc(n_tab * sizeof(sif_real));
    sif_real* pt = malloc(n_tab * sizeof(sif_real));
    for (uint32_t i = 0; i < n_tab; i++) {
      kt[i] = (sif_real)(KERNEL_K_LO *
                         pow(KERNEL_K_HI / KERNEL_K_LO, i / (n_tab - 1.0)));
      pt[i] = (sif_real)(amp * cdm_shape((double)kt[i]));
    }
    double* cov = sif_delta_covariance_pk(kt, pt, n_tab, walk->radii + lo_v,
      t - lo_v + 1, NULL, NULL, NULL, SIF_DEFAULT);

    sif_real* fv = sif_ep_multiplicity_function_emu(
      walk, vb, collapse, at_vic, n_at, NULL, SIF_DEFAULT);
    sif_real* f1 = sif_ep_multiplicity_function_emu(
      walk, vb, NULL, at_vic, n_at, NULL, SIF_DEFAULT);
    sif_real* gv = cov ? sif__ep_multiplicity_function_emu_grid(
                           walk->radii + lo_v, t - lo_v + 1, walk->sigma + lo_v,
                           vb + lo_v, walk->deriv_variance + lo_v,
                           collapse + lo_v, cov, NULL, SIF_DEFAULT)
                       : NULL;
    CHECK(fv && f1 && gv, "void-in-cloud through the walk returned NULL");
    if (fv && f1 && gv) {
      double worst = 0.0, k_max = 0.0, k_min = 1.0;
      for (uint32_t j = 0; j < n_at; j++) {
        if (!(gv[2 * j + 1] > 0.0f) || !(f1[j] > 0.0f))
          continue;
        const double e = fabs((double)fv[j] / (double)gv[2 * j + 1] - 1.0);
        const double kk = (double)fv[j] / (double)f1[j];
        worst = e > worst ? e : worst;
        k_max = kk > k_max ? kk : k_max;
        k_min = kk < k_min ? kk : k_min;
      }
      CHECK(k_min < 0.99,
        "void-in-cloud never mattered here (K >= %.4f); the comparison is "
        "vacuous",
        k_min);
      CHECK(worst < 1e-4,
        "void-in-cloud through the walk departs from the model given the "
        "direct covariance by %.2e",
        worst);
      CHECK(k_max <= 1.0 + 1e-6, "void-in-cloud raised f: K = %.8f", k_max);
      printf("  void-in-cloud: %.1e from the direct covariance, K %.4f to "
             "%.4f\n",
        worst, k_min, k_max);
    }
    sif_free_aligned(fv);
    sif_free_aligned(f1);
    sif_free_aligned(gv);
    sif_free_aligned(cov);
    sif_free_aligned(vb);
    free(kt);
    free(pt);
    free(collapse);
  }

  /* Guards. */
  {
    sif_real r;
    sif_real outside[2] = {0.5f, 10.0f};
    sif_real* res = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, outside, 2, NULL, SIF_DEFAULT);
    CHECK(res == NULL, "a radius below the lattice was accepted");
    sif_free_aligned(res);

    sif_real beyond[2] = {10.0f, 400.0f};
    res = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, beyond, 2, NULL, SIF_DEFAULT);
    CHECK(res == NULL, "a radius beyond the lattice was accepted");
    sif_free_aligned(res);

    sif_real down[2] = {20.0f, 10.0f};
    res = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, down, 2, NULL, SIF_DEFAULT);
    CHECK(res == NULL, "descending radii were accepted");
    sif_free_aligned(res);

    r = 10.0f;
    res = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, &r, 1, NULL, SIF_EP_EMU_BIN_CENTRES);
    CHECK(res == NULL, "bin averages over a single radius were accepted");
    sif_free_aligned(res);

    res = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, &r, 1, NULL, SIF_DEFAULT);
    CHECK(res != NULL, "a single radius was rejected");
    sif_free_aligned(res);

    res = sif_ep_multiplicity_function_emu(NULL, barrier, NULL, &r, 1, NULL, 0);
    CHECK(res == NULL, "a NULL walk was accepted");
    sif_free_aligned(res);
  }

  sif_free_aligned(barrier);
  sif_walk_free(walk);
  sif_variance_kernel_free(kernel);
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
  test_kernel();
  test_walk_emulator();

  sif_system_finalise();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall emulator checks passed\n");
  return 0;
}
