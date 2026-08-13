/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Excursion-set void size functions, against Jennings, Li & Hu (2013).
 *
 * Two checks carry the weight and neither involves the quadrature. The
 * logarithmic slope of sigma has a closed form for a power-law spectrum,
 * -(3+n)/2, independent of the window; and with a constant expansion factor
 * the SvdW and Vdn models differ by exactly F^3, since the only thing that
 * changes between them is which radius goes into the volume.
 */
#include "sif/core/system.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/model/size_function.h"
#include "sif/structures/delta_moments.h"
#include "sif/structures/size_function.h"
#include "sif/utils/align.h"
#include "test_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define N_K 4000

/*
 * Power-law spectrum. The range has to reach far enough below 1/R that the
 * integral has converged, which for a red slope means much lower than the
 * window scale: sigma^2 picks up k^(2+n) dk, and at n = -2.5 the low-k tail
 * still carries percent-level weight at k = 1e-5.
 */
static void power_law(sif_real* k, sif_real* pk, double slope) {
  const double lo = log(1e-8), hi = log(1e3);
  for (uint32_t i = 0; i < N_K; i++) {
    const double lk = lo + (hi - lo) * i / (N_K - 1.0);
    k[i] = (sif_real)exp(lk);
    pk[i] = (sif_real)pow(exp(lk), slope);
  }
}

/*
 * For P(k) ~ k^n, sigma^2 ~ R^-(3+n) whatever the window, since the window
 * only ever appears as W(kR) and the R dependence factors out of the integral.
 * So dln(sigma)/dln(R) = -(3+n)/2 exactly, at every radius.
 */
static void test_sigma_slope_closed_form(void) {
  printf("sigma slope vs closed form\n");

  sif_real* k = malloc(N_K * sizeof(sif_real));
  sif_real* pk = malloc(N_K * sizeof(sif_real));
  const sif_real radii[4] = {2.0f, 8.0f, 20.0f, 50.0f};

  const double slopes[3] = {-2.5, -2.0, -1.0};
  const sif_option windows[2] = {
    SIF_DELTA_FILTER_TOP_HAT, SIF_DELTA_FILTER_GAUSSIAN};
  const char* names[2] = {"top-hat", "Gaussian"};

  for (int w = 0; w < 2; w++) {
    for (int s = 0; s < 3; s++) {
      power_law(k, pk, slopes[s]);

      sif_real* got =
        sif_delta_sigma_slope_pk(k, pk, N_K, radii, 4, windows[w]);
      CHECK(got != NULL, "slope returned NULL");
      if (!got)
        continue;

      const double expected = -(3.0 + slopes[s]) / 2.0;
      double worst = 0.0;
      for (uint32_t r = 0; r < 4; r++) {
        const double rel = fabs((double)got[r] - expected) / fabs(expected);
        if (rel > worst)
          worst = rel;
      }

      printf("    %s, n=%+.1f:  expected %.4f  worst relative error %.2e\n",
        names[w], slopes[s], expected, worst);

      CHECK(worst < 1e-3, "%s n=%g: slope is off by %.3e from -(3+n)/2",
        names[w], slopes[s], worst);

      sif_free_aligned(got);
    }
  }

  free(k);
  free(pk);
}

/*
 * Rescales a spectrum to a target sigma at R = 8, so the models are exercised
 * where they predict something. An unnormalised power law leaves sigma around
 * 0.03, and exp(-delta_v^2 / 2 sigma^2) then underflows single precision at
 * every radius.
 */
static void normalize(sif_real* k, sif_real* pk, double target) {
  const sif_real eight[1] = {8.0f};
  sif_delta_moments_t* m =
    sif_delta_moments_pk(k, pk, N_K, eight, 1, 0, SIF_DELTA_FILTER_TOP_HAT);
  if (!m)
    return;

  const double s8 = (double)sif_delta_moments_sigma(m, 0)[0];
  sif_delta_moments_free(m);

  if (!(s8 > 0.0))
    return;

  const double scale = (target / s8) * (target / s8);
  for (uint32_t i = 0; i < N_K; i++)
    pk[i] = (sif_real)((double)pk[i] * scale);
}

/* The slope has to agree with a finite difference of sigma itself. */
static void test_sigma_slope_vs_finite_difference(void) {
  printf("sigma slope vs finite difference of sigma\n");

  sif_real* k = malloc(N_K * sizeof(sif_real));
  sif_real* pk = malloc(N_K * sizeof(sif_real));
  power_law(k, pk, -2.0);

  const double r0 = 12.0, h = 1e-3;
  const sif_real probe[3] = {
    (sif_real)(r0 * exp(-h)), (sif_real)r0, (sif_real)(r0 * exp(h))};

  sif_delta_moments_t* m =
    sif_delta_moments_pk(k, pk, N_K, probe, 3, 0, SIF_DELTA_FILTER_TOP_HAT);
  sif_real* slope =
    sif_delta_sigma_slope_pk(k, pk, N_K, probe, 3, SIF_DELTA_FILTER_TOP_HAT);

  CHECK(m && slope, "setup returned NULL");

  if (m && slope) {
    const sif_real* sigma = sif_delta_moments_sigma(m, 0);
    const double fd =
      (log((double)sigma[2]) - log((double)sigma[0])) / (2.0 * h);
    const double rel = fabs(fd - (double)slope[1]) / fabs(fd);

    printf("    analytic %.6f  vs  finite difference %.6f  (rel %.2e)\n",
      (double)slope[1], fd, rel);

    CHECK(rel < 1e-3, "analytic slope %g disagrees with %g (rel %.3e)",
      (double)slope[1], fd, rel);
  }

  sif_delta_moments_free(m);
  sif_free_aligned(slope);
  free(k);
  free(pk);
}

/*
 * The linear/non-linear mapping, both ways and both methods.
 *
 * The round trip is the check that carries the weight: it holds separately for
 * each method, with no reference value to agree on, so it tests the fit and
 * the root-find independently rather than against each other.
 */
static void test_delta_mapping(void) {
  printf("linear <-> non-linear density contrast\n");

  const double barriers[5] = {-2.7, -1.8, -1.24, -0.8, -0.5};
  const sif_option methods[2] = {SIF_SPHERICAL_B94, SIF_SPHERICAL_EXACT};
  const char* names[2] = {"B94", "exact"};

  for (int m = 0; m < 2; m++) {
    double worst_trip = 0.0;

    for (int i = 0; i < 5; i++) {
      const sif_real dnl =
        sif_spherical_map_nonlinear((sif_real)barriers[i], methods[m]);

      CHECK(dnl < 0.0f && dnl > -1.0f,
        "%s: delta_L=%g gave delta_NL=%g, outside (-1, 0)", names[m],
        barriers[i], (double)dnl);

      const sif_real back = sif_spherical_map_linear(dnl, methods[m]);
      const double rel = fabs((double)back - barriers[i]) / fabs(barriers[i]);
      if (rel > worst_trip)
        worst_trip = rel;
    }

    CHECK(worst_trip < 1e-5, "%s: the round trip is off by %.3e", names[m],
      worst_trip);
    printf(
      "  %-5s round trip worst relative error %.2e\n", names[m], worst_trip);
  }

  /*
   * The fit against what it is a fit to. Bernardeau quotes 0.2%, and it is
   * quoted on the expansion factor rather than on the contrast, so that is
   * what gets compared.
   */
  double worst_f = 0.0;
  for (int i = 0; i < 5; i++) {
    const double a = (double)sif_spherical_map_nonlinear(
      (sif_real)barriers[i], SIF_SPHERICAL_B94);
    const double b = (double)sif_spherical_map_nonlinear(
      (sif_real)barriers[i], SIF_SPHERICAL_EXACT);

    const double fa = pow(1.0 + a, -1.0 / 3.0);
    const double fb = pow(1.0 + b, -1.0 / 3.0);
    const double rel = fabs(fa - fb) / fb;
    if (rel > worst_f)
      worst_f = rel;

    printf("    delta_L=%+.2f  ->  F = %.4f (B94) vs %.4f (exact)\n",
      barriers[i], fa, fb);
  }
  CHECK(worst_f < 3e-3,
    "the fit and the exact solution disagree by %.3e on the expansion factor, "
    "beyond the 0.2%% the fit claims",
    worst_f);

  /* Shell crossing, the one value the literature pins down. */
  for (int m = 0; m < 2; m++) {
    const double d = (double)sif_spherical_map_nonlinear(-2.7f, methods[m]);
    const double f = pow(1.0 + d, -1.0 / 3.0);
    CHECK(fabs(f - 1.69) < 0.02,
      "%s: shell crossing gave F=%g, expected about 1.69", names[m], f);

    /* And the non-linear contrast the reference implementations default to. */
    CHECK(fabs(d - (-0.795)) < 0.005,
      "%s: shell crossing gave delta_NL=%g, expected about -0.795", names[m],
      d);
  }

  /* The linear limit: a barely underdense region has not evolved. */
  for (int m = 0; m < 2; m++) {
    const double tiny = -1e-4;
    const double d =
      (double)sif_spherical_map_nonlinear((sif_real)tiny, methods[m]);
    CHECK(fabs(d - tiny) < 1e-6,
      "%s: delta_L=%g should map to nearly itself, gave %g", names[m], tiny, d);
  }

  /* Out of the expanding branch, both directions, both methods. */
  for (int m = 0; m < 2; m++) {
    CHECK(sif_spherical_map_nonlinear(0.5f, methods[m]) == 0.0f,
      "%s: a positive linear contrast was accepted", names[m]);
    CHECK(sif_spherical_map_nonlinear(0.0f, methods[m]) == 0.0f,
      "%s: a zero linear contrast was accepted", names[m]);
    CHECK(sif_spherical_map_linear(-1.0f, methods[m]) == 0.0f,
      "%s: total evacuation was accepted", names[m]);
    CHECK(sif_spherical_map_linear(0.5f, methods[m]) == 0.0f,
      "%s: a positive non-linear contrast was accepted", names[m]);
  }
}

/* The multiplicity function has to be continuous across its branch. */
static void test_multiplicity_continuity(void) {
  printf("multiplicity function across the x = 0.276 branch\n");

  const sif_real delta_c = 1.686f;
  const double barriers[3] = {-2.7, -1.8, -0.8};

  for (int i = 0; i < 3; i++) {
    const double abs_dv = fabs(barriers[i]);
    const double dcal = abs_dv / ((double)delta_c + abs_dv);

    /* sigma at which x sits exactly on the switch. */
    const double sigma_switch = 0.276 * abs_dv / dcal;

    const sif_real probe[2] = {(sif_real)(sigma_switch * (1.0 - 1e-6)),
      (sif_real)(sigma_switch * (1.0 + 1e-6))};

    sif_real* f =
      sif_svdw_multiplicity_function(probe, 2, (sif_real)barriers[i], delta_c);
    CHECK(f != NULL, "multiplicity returned NULL");

    if (f) {
      const double rel = fabs((double)f[1] - f[0]) / (double)f[0];
      printf("    delta_v=%+.1f  D=%.3f  f(-)=%.5f  f(+)=%.5f  (%.2f%% jump)\n",
        barriers[i], dcal, (double)f[0], (double)f[1], 100.0 * rel);

      /* The two forms are matched at the switch by construction, so the step
       * measures the approximation, not a coding error. */
      CHECK(rel < 0.02, "delta_v=%g: the branch jumps by %.2f%% at x = 0.276",
        barriers[i], 100.0 * rel);

      CHECK(f[0] > 0.0f && f[1] > 0.0f, "the multiplicity went non-positive");
      sif_free_aligned(f);
    }
  }

  /* Guards. */
  const sif_real sigma[1] = {1.0f};
  CHECK(sif_svdw_multiplicity_function(sigma, 1, 2.7f, 1.686f) == NULL,
    "a positive delta_v was accepted");
  CHECK(sif_svdw_multiplicity_function(sigma, 1, -2.7f, -1.0f) == NULL,
    "a negative delta_c was accepted");
}

/*
 * SvdW and Vdn share everything except the volume they divide by, so with a
 * constant expansion factor they differ by exactly F^3 at every radius.
 */
static void test_svdw_vdn_ratio(void) {
  printf("SvdW / Vdn = F^3\n");

  sif_real* k = malloc(N_K * sizeof(sif_real));
  sif_real* pk = malloc(N_K * sizeof(sif_real));
  power_law(k, pk, -2.0);
  normalize(k, pk, 0.8);

  const uint32_t n_r = 30;
  sif_real* radii = malloc(n_r * sizeof(sif_real));
  for (uint32_t i = 0; i < n_r; i++)
    radii[i] =
      (sif_real)exp(log(1.0) + (log(25.0) - log(1.0)) * i / (n_r - 1.0));

  const sif_real dv = -2.7f, dc = 1.686f;

  /* The same expansion factor the models derive internally when passed 0. */
  const double f =
    pow(1.0 + (double)sif_spherical_map_nonlinear(dv, SIF_SPHERICAL_B94),
      -1.0 / 3.0);

  sif_size_function_t* sw =
    sif_size_function_svdw(k, pk, N_K, radii, n_r, dv, dc, SIF_DEFAULT);
  sif_size_function_t* vdn =
    sif_size_function_vdn(k, pk, N_K, radii, n_r, dv, dc, SIF_DEFAULT);

  CHECK(sw && vdn, "size functions returned NULL");

  if (sw && vdn) {
    const double expected = f * f * f;
    double worst = 0.0;

    for (uint32_t i = 0; i < n_r; i++) {
      CHECK(sw->vsf[i] > 0.0f && vdn->vsf[i] > 0.0f,
        "a size function went non-positive at R=%g", (double)radii[i]);

      const double ratio = (double)sw->vsf[i] / (double)vdn->vsf[i];
      const double rel = fabs(ratio - expected) / expected;
      if (rel > worst)
        worst = rel;
    }

    printf("    F=%.4f  F^3=%.4f  worst deviation of the ratio %.2e\n", f,
      expected, worst);

    CHECK(worst < 1e-5,
      "SvdW/Vdn is not F^3 = %g at every radius (worst relative %.3e)",
      expected, worst);

    /* Vdn is the smaller of the two, since it divides by the larger volume. */
    CHECK(vdn->vsf[0] < sw->vsf[0],
      "Vdn is not below SvdW; the volumes may be swapped");

    /* The container describes itself. */
    CHECK(
      sw->n_bins == n_r && sw->r_min == radii[0] && sw->r_max == radii[n_r - 1],
      "the size function container is inconsistent");
  }

  /* The linear convention is the log one over R. */
  sif_size_function_t* per_r =
    sif_size_function_vdn(k, pk, N_K, radii, n_r, dv, dc, SIF_VSF_BIN_LINEAR);
  if (per_r && vdn) {
    double worst = 0.0;
    for (uint32_t i = 0; i < n_r; i++) {
      const double expect = (double)vdn->vsf[i] / (double)radii[i];
      const double rel = fabs((double)per_r->vsf[i] - expect) / expect;
      if (rel > worst)
        worst = rel;
    }
    CHECK(worst < 1e-5, "dn/dR is not dn/dlnR over R (worst %.3e)", worst);
  }

  /*
   * The spherical-evolution option has to reach the mapping. Both methods
   * agree to well under a percent on the expansion factor, so the two curves
   * must differ -- proving the flag is threaded through -- but only slightly.
   */
  sif_size_function_t* forced =
    sif_size_function_svdw(k, pk, N_K, radii, n_r, dv, dc, SIF_SPHERICAL_EXACT);
  if (forced && sw) {
    double worst = 0.0;
    int differs = 0;
    for (uint32_t i = 0; i < n_r; i++) {
      if (forced->vsf[i] != sw->vsf[i])
        differs = 1;
      const double rel =
        fabs((double)forced->vsf[i] - (double)sw->vsf[i]) / (double)sw->vsf[i];
      if (rel > worst)
        worst = rel;
    }
    CHECK(differs,
      "SIF_SPHERICAL_EXACT gave the same curve as B94; the option is not "
      "reaching the mapping");
    CHECK(worst < 0.02,
      "the two spherical-evolution methods give size functions differing by "
      "%.3e, far more than their agreement on the expansion factor allows",
      worst);
    printf("    B94 vs exact: worst relative difference %.2e\n", worst);
  }

  sif_size_function_free(sw);
  sif_size_function_free(vdn);
  sif_size_function_free(per_r);
  sif_size_function_free(forced);
  free(k);
  free(pk);
  free(radii);
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg,
    .omp_config = NULL,
    .verbose = false,
    .log_level = SIF_LOG_LEVEL_ERROR};
  sif_init(&cfg);

  test_sigma_slope_closed_form();
  test_sigma_slope_vs_finite_difference();
  test_delta_mapping();
  test_multiplicity_continuity();
  test_svdw_vdn_ratio();

  sif_finalize();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall VSF model checks passed\n");
  return 0;
}
