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
#include "sif/model/deltamoments.h"
#include "sif/model/sizefunction.h"
#include "sif/structures/deltamoments.h"
#include "sif/structures/sizefunction.h"
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
static void power_law(real_t* k, real_t* pk, double slope) {
  const double lo = log(1e-8), hi = log(1e3);
  for (uint32_t i = 0; i < N_K; i++) {
    const double lk = lo + (hi - lo) * i / (N_K - 1.0);
    k[i] = (real_t)exp(lk);
    pk[i] = (real_t)pow(exp(lk), slope);
  }
}

/*
 * For P(k) ~ k^n, sigma^2 ~ R^-(3+n) whatever the window, since the window
 * only ever appears as W(kR) and the R dependence factors out of the integral.
 * So dln(sigma)/dln(R) = -(3+n)/2 exactly, at every radius.
 */
static void test_sigma_slope_closed_form(void) {
  printf("sigma slope vs closed form\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  const real_t radii[4] = {2.0f, 8.0f, 20.0f, 50.0f};

  const double slopes[3] = {-2.5, -2.0, -1.0};
  const sif_option_t windows[2] = {
    SIF_DELTA_FILTER_TOP_HAT, SIF_DELTA_FILTER_GAUSSIAN};
  const char* names[2] = {"top-hat", "Gaussian"};

  for (int w = 0; w < 2; w++) {
    for (int s = 0; s < 3; s++) {
      power_law(k, pk, slopes[s]);

      real_t* got = sif_sigma_slope_pk(k, pk, N_K, radii, 4, windows[w]);
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
static void normalize(real_t* k, real_t* pk, double target) {
  const real_t eight[1] = {8.0f};
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
    pk[i] = (real_t)((double)pk[i] * scale);
}

/* The slope has to agree with a finite difference of sigma itself. */
static void test_sigma_slope_vs_finite_difference(void) {
  printf("sigma slope vs finite difference of sigma\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  power_law(k, pk, -2.0);

  const double r0 = 12.0, h = 1e-3;
  const real_t probe[3] = {
    (real_t)(r0 * exp(-h)), (real_t)r0, (real_t)(r0 * exp(h))};

  sif_delta_moments_t* m =
    sif_delta_moments_pk(k, pk, N_K, probe, 3, 0, SIF_DELTA_FILTER_TOP_HAT);
  real_t* slope =
    sif_sigma_slope_pk(k, pk, N_K, probe, 3, SIF_DELTA_FILTER_TOP_HAT);

  CHECK(m && slope, "setup returned NULL");

  if (m && slope) {
    const real_t* sigma = sif_delta_moments_sigma(m, 0);
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

/* The expansion factor inverts the barrier relation it was derived from. */
static void test_expansion_factor(void) {
  printf("expansion factor\n");

  const double c = SIF_SPHERICAL_EXPANSION_C;
  const double barriers[4] = {-2.7, -1.8, -1.24, -0.8};

  for (int i = 0; i < 4; i++) {
    const double f = (double)sif_expansion_factor((real_t)barriers[i]);

    /* Invert: delta_v = c [1 - (r_NL/r_L)^(3/c)]. */
    const double back = c * (1.0 - pow(f, 3.0 / c));
    CHECK(fabs(back - barriers[i]) < 1e-4,
      "delta_v=%g gave F=%g which inverts to %g", barriers[i], f, back);

    /* And the nonlinear density it implies, rho_v/rho_m = F^-3. */
    printf("    delta_v=%+.2f  ->  F=%.4f  (rho_v/rho_m = %.3f)\n",
      barriers[i], f, 1.0 / (f * f * f));
  }

  /* Shell crossing: the reference quotes about 1.7. */
  const double f_sc = (double)sif_expansion_factor(-2.7f);
  CHECK(fabs(f_sc - 1.69) < 0.02,
    "shell crossing gave F=%g, expected about 1.69", f_sc);

  CHECK(sif_expansion_factor(0.5f) == 0.0f,
    "a positive barrier was accepted");
}

/* The multiplicity function has to be continuous across its branch. */
static void test_multiplicity_continuity(void) {
  printf("multiplicity function across the x = 0.276 branch\n");

  const real_t delta_c = 1.686f;
  const double barriers[3] = {-2.7, -1.8, -0.8};

  for (int i = 0; i < 3; i++) {
    const double abs_dv = fabs(barriers[i]);
    const double dcal = abs_dv / ((double)delta_c + abs_dv);

    /* sigma at which x sits exactly on the switch. */
    const double sigma_switch = 0.276 * abs_dv / dcal;

    const real_t probe[2] = {(real_t)(sigma_switch * (1.0 - 1e-6)),
      (real_t)(sigma_switch * (1.0 + 1e-6))};

    real_t* f = sif_multiplicity_function_svdw(
      probe, 2, (real_t)barriers[i], delta_c);
    CHECK(f != NULL, "multiplicity returned NULL");

    if (f) {
      const double rel = fabs((double)f[1] - f[0]) / (double)f[0];
      printf("    delta_v=%+.1f  D=%.3f  f(-)=%.5f  f(+)=%.5f  (%.2f%% jump)\n",
        barriers[i], dcal, (double)f[0], (double)f[1], 100.0 * rel);

      /* The two forms are matched at the switch by construction, so the step
       * measures the approximation, not a coding error. */
      CHECK(rel < 0.02,
        "delta_v=%g: the branch jumps by %.2f%% at x = 0.276", barriers[i],
        100.0 * rel);

      CHECK(f[0] > 0.0f && f[1] > 0.0f, "the multiplicity went non-positive");
      sif_free_aligned(f);
    }
  }

  /* Guards. */
  const real_t sigma[1] = {1.0f};
  CHECK(sif_multiplicity_function_svdw(sigma, 1, 2.7f, 1.686f) == NULL,
    "a positive delta_v was accepted");
  CHECK(sif_multiplicity_function_svdw(sigma, 1, -2.7f, -1.0f) == NULL,
    "a negative delta_c was accepted");
}

/*
 * SvdW and Vdn share everything except the volume they divide by, so with a
 * constant expansion factor they differ by exactly F^3 at every radius.
 */
static void test_svdw_vdn_ratio(void) {
  printf("SvdW / Vdn = F^3\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  power_law(k, pk, -2.0);
  normalize(k, pk, 0.8);

  const uint32_t n_r = 30;
  real_t* radii = malloc(n_r * sizeof(real_t));
  for (uint32_t i = 0; i < n_r; i++)
    radii[i] = (real_t)exp(log(1.0) + (log(25.0) - log(1.0)) * i / (n_r - 1.0));

  const real_t dv = -2.7f, dc = 1.686f;
  const double f = (double)sif_expansion_factor(dv);

  sif_size_function_t* sw =
    sif_size_function_svdw(k, pk, N_K, radii, n_r, dv, dc, 0.0f, SIF_DEFAULT);
  sif_size_function_t* vdn =
    sif_size_function_vdn(k, pk, N_K, radii, n_r, dv, dc, 0.0f, SIF_DEFAULT);

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
    CHECK(sw->n_bins == n_r && sw->r_min == radii[0] &&
            sw->r_max == radii[n_r - 1],
      "the size function container is inconsistent");
  }

  /* The linear convention is the log one over R. */
  sif_size_function_t* per_r = sif_size_function_vdn(
    k, pk, N_K, radii, n_r, dv, dc, 0.0f, SIF_VSF_BIN_LINEAR);
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

  /* An explicit factor overrides the derived one, and 1.7 is the value the
   * reference actually uses alongside delta_v = -2.7. */
  sif_size_function_t* forced =
    sif_size_function_svdw(k, pk, N_K, radii, n_r, dv, dc, 1.7f, SIF_DEFAULT);
  if (forced && sw) {
    CHECK(forced->vsf[0] != sw->vsf[0],
      "an explicit expansion factor did not override the derived one");
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
  sif_config_t cfg = {.fft_config = &fftcfg, .omp_config = NULL,
                      .verbose = false, .log_level = SIF_LOG_LEVEL_ERROR};
  sif_init(&cfg);

  test_sigma_slope_closed_form();
  test_sigma_slope_vs_finite_difference();
  test_expansion_factor();
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
