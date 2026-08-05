/*
 * Peak statistics of a Gaussian random field, following Bardeen, Bond, Kaiser
 * & Szalay (1986), in the presentation of Wu, Phys. Dark Universe 30 (2020)
 * 100654, eqs. (17)-(19).
 *
 * The number density of maxima depends on the field only through its spectral
 * moments, which is what makes it the natural prediction to hold a
 * phase-randomized surrogate against: everything BBKS knows about a field
 * survives phase randomization, so any disagreement with a directly counted
 * catalogue is phase information.
 */

#include "sif/model/bbks.h"

#include "structures/results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define __TAG "bbks"

/* The fit is calibrated for CDM-like spectra; outside this band it degrades. */
#define __BBKS_GAMMA_MIN 0.4
#define __BBKS_GAMMA_MAX 0.7

/* Simpson intervals for the exact G. The integrand is smooth and the range is
 * scaled to the kernel width, so this is far past converged: doubling it moves
 * the result by less than a part in 10^9. */
#define __BBKS_QUAD_INTERVALS 512

/* Half-width of the quadrature range, in units of the kernel's sigma. */
#define __BBKS_QUAD_SIGMAS 12.0

/* Simpson intervals for the cumulative integral over nu. */
#define __BBKS_NU_INTERVALS 2048

/* --- The fitted form, BBKS eq. (4.4) --- */

static double __g_fitted(double g, double w) {
  const double g2 = g * g;

  /* BBKS eq. (4.5). The 9 - 5 gamma^2 denominator stays positive for any
   * gamma in (0, 1), so no guard is needed there. */
  const double d = 9.0 - 5.0 * g2;

  const double a1 = 2.5 / d;
  const double b1 = 432.0 / (sqrt(10.0 * M_PI) * d * d * sqrt(d));
  const double c1 = 1.84 + 1.13 * pow(1.0 - g2, 5.72);
  const double c2 = 8.91 + 1.27 * exp(6.51 * g2);
  const double c3 = 2.58 * exp(1.05 * g2);

  /* BBKS eq. (4.4). */
  const double numerator =
    w * w * w - 3.0 * g2 * w + (b1 * w * w + c1) * exp(-a1 * w * w);
  const double denominator = 1.0 + c2 * exp(-c3 * w);

  return numerator / denominator;
}

/* --- The exact form, Wu (2020) eqs. (18)-(19) --- */

/*
 * The curvature weight, Wu (2020) eq. (19). x is the trace of the second
 * derivative tensor in units of sigma_2, and f(0) = 0 exactly: the two
 * constant terms of the exponential bracket cancel.
 */
static inline double __bbks_f(double x) {
  static const double root_5_2 = 1.5811388300841898;  /* sqrt(5/2)    */
  static const double root_2_5pi = 0.3568248232305542; /* sqrt(2/(5 pi)) */

  const double x2 = x * x;

  const double polynomial = 0.5 * (x2 * x - 3.0 * x) *
                            (erf(x * root_5_2) + erf(0.5 * x * root_5_2));

  const double exponential =
    root_2_5pi * ((7.75 * x2 + 1.6) * exp(-0.625 * x2) +
                   (0.5 * x2 - 1.6) * exp(-2.5 * x2));

  return polynomial + exponential;
}

static double __g_exact(double g, double w) {

  /* Width of the Gaussian kernel in eq. (18). */
  const double s = sqrt(1.0 - g * g);

  /*
   * The integral runs over x > 0 only -- that lower limit is the maximum
   * condition. The upper limit is pushed past where the kernel has died; for
   * w far negative the kernel barely overlaps x > 0 at all, and the remaining
   * sliver near the origin is what the range below still captures.
   */
  const double lo = 0.0;
  double hi = w + __BBKS_QUAD_SIGMAS * s;
  if (hi < __BBKS_QUAD_SIGMAS * s)
    hi = __BBKS_QUAD_SIGMAS * s;

  const int n = __BBKS_QUAD_INTERVALS; /* even, so Simpson applies */
  const double h = (hi - lo) / n;
  const double norm = 1.0 / sqrt(2.0 * M_PI * (1.0 - g * g));
  const double inv_two_var = 1.0 / (2.0 * (1.0 - g * g));

  double sum = 0.0;

  for (int i = 0; i <= n; i++) {
    const double x = lo + i * h;
    const double d = x - w;
    const double integrand = __bbks_f(x) * exp(-d * d * inv_two_var);

    /* Composite Simpson weights: 1, 4, 2, 4, ..., 4, 1. */
    const double weight = (i == 0 || i == n) ? 1.0 : ((i & 1) ? 4.0 : 2.0);

    sum += weight * integrand;
  }

  return norm * sum * h / 3.0;
}

/*
 * The two forms share one entry point rather than two, so that swapping
 * between them is a flag on the call rather than a different symbol -- which
 * is what lets the number-density routines below thread the choice straight
 * through from their own options.
 */
real_t sif_g_bbks(real_t gamma, real_t w, sif_option_t opt) {
  const double g = (double)gamma;

  if (!(g > 0.0) || !(g < 1.0)) {
    SIF_LOG_ERROR(__TAG, "gamma is %g, must lie strictly in (0, 1)", g);
    return (real_t)0.0;
  }

  return (real_t)(((opt & __SIF_BBKS_G_MASK) == SIF_BBKS_G_EXACT)
                    ? __g_exact(g, (double)w)
                    : __g_fitted(g, (double)w));
}

/* --- The differential number density of maxima --- */

real_t* sif_differential_number_density_bbks(const real_t* nu,
  const real_t* gamma, const real_t* r_star, uint32_t size, sif_option_t opt) {

  if (!nu || !gamma || !r_star || size == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to sif_differential_number_density_bbks");
    return NULL;
  }

  for (uint32_t i = 0; i < size; i++) {
    if (!(r_star[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG, "r_star[%u] is %g, must be strictly positive", i,
        (double)r_star[i]);
      return NULL;
    }
    if (!(gamma[i] > 0.0f) || !(gamma[i] < 1.0f)) {
      SIF_LOG_ERROR(__TAG, "gamma[%u] is %g, must lie strictly in (0, 1)", i,
        (double)gamma[i]);
      return NULL;
    }
  }

  const bool exact = (opt & __SIF_BBKS_G_MASK) == SIF_BBKS_G_EXACT;

  /* Only the fit has a calibration band to fall outside of. One warning for
   * the whole array rather than one per entry: a caller sweeping R will
   * usually cross the band somewhere. */
  if (!exact) {
    uint32_t n_outside = 0;
    for (uint32_t i = 0; i < size; i++) {
      if (gamma[i] < __BBKS_GAMMA_MIN || gamma[i] > __BBKS_GAMMA_MAX)
        n_outside++;
    }
    if (n_outside > 0) {
      SIF_LOG_WARNING(__TAG,
        "%u of %u entries have gamma outside [%g, %g], where the BBKS fit to G "
        "is less accurate; SIF_BBKS_G_EXACT has no such restriction",
        n_outside, size, __BBKS_GAMMA_MIN, __BBKS_GAMMA_MAX);
    }
  }

  real_t* out = sif_calloc_aligned((size_t)size, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the number density array");
    return NULL;
  }

  const double four_pi_sq = 4.0 * M_PI * M_PI; /* (2 pi)^2 */

#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < size; i++) {
    const double n = (double)nu[i];
    const double rs = (double)r_star[i];
    const real_t w = (real_t)(gamma[i] * n);

    const double g = exact ? __g_exact((double)gamma[i], (double)w)
                           : __g_fitted((double)gamma[i], (double)w);

    const double density = exp(-0.5 * n * n) / (four_pi_sq * rs * rs * rs) * g;

    /* The fitted G dips slightly negative at nu well below zero, where the
     * exact G is small and positive. A negative number density is not a
     * result, so it is reported as zero. */
    out[i] = (real_t)(density > 0.0 ? density : 0.0);
  }

  return out;
}

/* --- The cumulative density of maxima above a threshold --- */

/*
 * Width of the nu integral, measured from where the integrand actually peaks.
 *
 * Writing nu = nu_t + u, the Gaussian factor becomes
 * exp(-nu_t^2/2) exp(-nu_t u) exp(-u^2/2), so the decay scale in u is 1/nu_t
 * once nu_t is large and order unity otherwise. Taking the larger of the two
 * covers both regimes.
 */
static inline double __bbks_nu_span(double nu_t) {
  const double scale = nu_t > 1.0 ? nu_t : 1.0;
  const double span = 40.0 / scale;
  return span > 12.0 ? span : 12.0;
}

real_t* sif_cumulative_number_density_bbks(
  real_t delta, const sif_delta_moments_t* moments, sif_option_t opt) {

  if (!moments || !moments->sigma || moments->n_radii == 0) {
    SIF_LOG_ERROR(__TAG, "invalid moment set");
    return NULL;
  }

  if (moments->order < 2) {
    SIF_LOG_ERROR(__TAG,
      "the cumulative density needs sigma_0 through sigma_2, but the moment "
      "set only reaches order %u",
      moments->order);
    return NULL;
  }

  const real_t* sigma_0 = sif_delta_moments_sigma(moments, 0);

  real_t* gamma = sif_gamma_moments(moments);
  real_t* r_star = sif_r_star_moments(moments);
  real_t* out = sif_calloc_aligned((size_t)moments->n_radii, sizeof(real_t));

  if (!gamma || !r_star || !out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the cumulative density arrays");
    sif_free_aligned(gamma);
    sif_free_aligned(r_star);
    sif_free_aligned(out);
    return NULL;
  }

  const bool exact = (opt & __SIF_BBKS_G_MASK) == SIF_BBKS_G_EXACT;
  const double four_pi_sq = 4.0 * M_PI * M_PI; /* (2 pi)^2 */

  uint32_t n_outside = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : n_outside)
  for (uint32_t r = 0; r < moments->n_radii; r++) {
    const double s0 = (double)sigma_0[r];
    const double g = (double)gamma[r];
    const double rs = (double)r_star[r];

    /* A radius whose moments collapsed -- shot-noise subtraction can do that
     * -- has no threshold to speak of. Reported as zero and flagged below. */
    if (!(s0 > 0.0) || !(g > 0.0) || !(g < 1.0) || !(rs > 0.0)) {
      n_outside++;
      continue;
    }

    /*
     * The threshold enters through its magnitude. BBKS counts maxima, and for
     * a Gaussian field the density of minima below -|delta| equals that of
     * maxima above +|delta|, so one integral serves both signs: a caller
     * passing a void threshold gets the void count without flipping anything.
     * It also keeps nu_t non-negative, which is what lets the quadrature range
     * below be anchored on nu_t alone.
     */
    const double nu_t = fabs((double)delta) / s0;

    const double nu_lo = nu_t;
    const double nu_hi = nu_t + __bbks_nu_span(nu_t);

    const int n = __BBKS_NU_INTERVALS; /* even, so Simpson applies */
    const double h = (nu_hi - nu_lo) / n;

    double sum = 0.0;

    for (int i = 0; i <= n; i++) {
      const double nu = nu_lo + i * h;
      const real_t w = (real_t)(g * nu);

      const double gv = exact ? __g_exact(g, (double)w)
                              : __g_fitted(g, (double)w);

      const double integrand = exp(-0.5 * nu * nu) * gv;

      /* Composite Simpson weights: 1, 4, 2, 4, ..., 4, 1. */
      const double weight = (i == 0 || i == n) ? 1.0 : ((i & 1) ? 4.0 : 2.0);

      sum += weight * integrand;
    }

    const double density = sum * h / 3.0 / (four_pi_sq * rs * rs * rs);

    /* The fitted G goes slightly negative well below zero, which can drag a
     * low-threshold integral down; a negative count is not a result. */
    out[r] = (real_t)(density > 0.0 ? density : 0.0);
  }

  if (n_outside > 0) {
    SIF_LOG_WARNING(__TAG,
      "%u of %u radii had a non-positive sigma_0, gamma or R_star and were "
      "reported as zero density",
      n_outside, moments->n_radii);
  }

  if (!exact) {
    uint32_t n_band = 0;
    for (uint32_t r = 0; r < moments->n_radii; r++) {
      if (gamma[r] < __BBKS_GAMMA_MIN || gamma[r] > __BBKS_GAMMA_MAX)
        n_band++;
    }
    if (n_band > 0) {
      SIF_LOG_WARNING(__TAG,
        "%u of %u radii have gamma outside [%g, %g], where the BBKS fit to G "
        "is less accurate; SIF_BBKS_G_EXACT has no such restriction",
        n_band, moments->n_radii, __BBKS_GAMMA_MIN, __BBKS_GAMMA_MAX);
    }
  }

  sif_free_aligned(gamma);
  sif_free_aligned(r_star);

  return out;
}

/* --- The size function implied by the cumulative density --- */

/*
 * Second-order finite difference on a non-uniform grid.
 *
 * Central in the interior, one-sided at the two ends, so the endpoints keep
 * the same order as the rest rather than degrading to first order where the
 * curve is often most interesting.
 */
static void __derivative(
  const double* x, const double* y, uint32_t n, double* out) {

  if (n == 2) {
    const double slope = (y[1] - y[0]) / (x[1] - x[0]);
    out[0] = slope;
    out[1] = slope;
    return;
  }

  for (uint32_t i = 1; i + 1 < n; i++) {
    const double h1 = x[i] - x[i - 1];
    const double h2 = x[i + 1] - x[i];

    out[i] = -h2 / (h1 * (h1 + h2)) * y[i - 1] +
             (h2 - h1) / (h1 * h2) * y[i] +
             h1 / (h2 * (h1 + h2)) * y[i + 1];
  }

  {
    const double h1 = x[1] - x[0];
    const double h2 = x[2] - x[1];
    out[0] = -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * y[0] +
             (h1 + h2) / (h1 * h2) * y[1] - h1 / (h2 * (h1 + h2)) * y[2];
  }

  {
    const uint32_t l = n - 1;
    const double h1 = x[l - 1] - x[l - 2];
    const double h2 = x[l] - x[l - 1];
    out[l] = h2 / (h1 * (h1 + h2)) * y[l - 2] -
             (h1 + h2) / (h1 * h2) * y[l - 1] +
             (2.0 * h2 + h1) / (h2 * (h1 + h2)) * y[l];
  }
}

sif_size_function_t* sif_size_function_bbks(
  real_t delta, const sif_delta_moments_t* moments, sif_option_t opt) {

  if (!moments || moments->n_radii < 2) {
    SIF_LOG_ERROR(__TAG,
      "the size function is a derivative in R and needs at least two radii");
    return NULL;
  }

  const uint32_t n = moments->n_radii;

  for (uint32_t r = 1; r < n; r++) {
    if (!(moments->radii[r] > moments->radii[r - 1])) {
      SIF_LOG_ERROR(__TAG,
        "radii must be strictly increasing to differentiate in R; index %u is "
        "%g after %g",
        r, (double)moments->radii[r], (double)moments->radii[r - 1]);
      return NULL;
    }
  }

  if (n < 5) {
    SIF_LOG_WARNING(__TAG,
      "differentiating over only %u radii; the result is a finite difference "
      "on that grid and is no better than the sampling allows",
      n);
  }

  real_t* cumulative = sif_cumulative_number_density_bbks(delta, moments, opt);
  if (!cumulative)
    return NULL;

  double* ln_r = malloc((size_t)n * sizeof(double));
  double* ln_c = malloc((size_t)n * sizeof(double));
  double* slope = malloc((size_t)n * sizeof(double));
  sif_size_function_t* out = __sif_size_function_alloc(n);

  if (!ln_r || !ln_c || !slope || !out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the size function");
    free(ln_r);
    free(ln_c);
    free(slope);
    sif_size_function_free(out);
    sif_free_aligned(cumulative);
    return NULL;
  }

  /* counts and err stay zero: a model counts nothing and carries no Poisson
   * uncertainty. `options` records which derivative convention `vsf` holds,
   * so the units travel with the data instead of with the caller. */
  out->options = opt;
  out->r_min = moments->radii[0];
  out->r_max = moments->radii[n - 1];

  memcpy(out->r_centers, moments->radii, (size_t)n * sizeof(real_t));
  __sif_edges_from_centers(moments->radii, n, out->r_edges);

  /*
   * Differentiated as C * dlnC/dlnR rather than by differencing C directly.
   * C falls over many decades, so a finite difference in the raw values is
   * dominated by the largest entry in each stencil and loses the shape.
   *
   * A radius whose C underflowed carries no usable slope, and one bad entry
   * would poison its two neighbours through the stencil. Those are reported
   * as zero rather than propagated.
   */
  uint32_t n_dead = 0;
  for (uint32_t r = 0; r < n; r++) {
    ln_r[r] = log((double)moments->radii[r]);
    if (cumulative[r] > 0.0f) {
      ln_c[r] = log((double)cumulative[r]);
    } else {
      ln_c[r] = 0.0;
      n_dead++;
    }
  }

  if (n_dead == n) {
    SIF_LOG_WARNING(__TAG,
      "the cumulative density underflowed at every radius; the size function "
      "is identically zero");
    free(ln_r);
    free(ln_c);
    free(slope);
    sif_free_aligned(cumulative);
    return out;
  }

  __derivative(ln_r, ln_c, n, slope);

  const bool per_radius =
    (opt & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR;

  for (uint32_t r = 0; r < n; r++) {
    if (!(cumulative[r] > 0.0f))
      continue;

    /* Neighbours of a dead entry would be differenced against a fabricated
     * log, so they are dropped too. */
    const bool touches_dead =
      (r > 0 && !(cumulative[r - 1] > 0.0f)) ||
      (r + 1 < n && !(cumulative[r + 1] > 0.0f));
    if (touches_dead)
      continue;

    /* dC/dlnR = C dlnC/dlnR, and dC/dR = (dC/dlnR) / R. The sign is flipped
     * so the result is a positive number density: C decreases with R. */
    double value = -(double)cumulative[r] * slope[r];
    if (per_radius)
      value /= (double)moments->radii[r];

    out->vsf[r] = (real_t)(value > 0.0 ? value : 0.0);
  }

  if (n_dead > 0) {
    SIF_LOG_WARNING(__TAG,
      "the cumulative density underflowed at %u of %u radii; those entries "
      "and their neighbours are reported as zero",
      n_dead, n);
  }

  free(ln_r);
  free(ln_c);
  free(slope);
  sif_free_aligned(cumulative);

  return out;
}
