/*
 * Spectral moments evaluated from a model power spectrum: the continuum
 * counterpart of the grid estimator in measure/. No box, no realization, no
 * cosmic variance -- and no FFT, which is why this side of the split carries
 * its own window rather than borrowing the filter machinery.
 */

#include "sif/model/deltamoments.h"

#include "structures/results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define __TAG "delta"

/* high_k_fraction above this is reported: past it the moment is dominated by
 * the far end of the tabulated k range rather than by the spectrum. */
#define __HIGH_K_WARN_LEVEL 0.1

static int __validate_order(uint8_t order) {
  if (order > SIF_MAX_MOMENT_ORDER) {
    SIF_LOG_ERROR(__TAG, "moment order %u exceeds the maximum of %d", order,
      SIF_MAX_MOMENT_ORDER);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* Fourier transform of the smoothing window, as a function of x = kR. */
static inline double __window(bool gaussian, double x) {
  if (gaussian)
    return exp(-0.5 * x * x);

  /* Top-hat. The series limit avoids the 0/0 at small x, where the naive form
   * loses all its significant digits. */
  if (x < 1e-4)
    return 1.0 - x * x / 10.0;
  return 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
}

static int __validate_pk_table(
  const real_t* k, const real_t* pk, uint32_t n_points) {

  if (!k || !pk || n_points < 2) {
    SIF_LOG_ERROR(__TAG, "the P(k) table needs at least two samples");
    return SIF_ERR_INVALID;
  }

  for (uint32_t i = 0; i < n_points; i++) {
    if (!(k[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG,
        "k[%u] is %g; the quadrature runs in log k, so all wavenumbers must be "
        "strictly positive",
        i, (double)k[i]);
      return SIF_ERR_INVALID;
    }
    if (i > 0 && !(k[i] > k[i - 1])) {
      SIF_LOG_ERROR(__TAG, "k is not strictly increasing at index %u", i);
      return SIF_ERR_INVALID;
    }
  }

  return SIF_OK;
}

sif_delta_moments_t* sif_delta_moments_pk(const real_t* k,
  const real_t* pk, uint32_t n_points, const real_t* radii, uint32_t n_radii,
  uint8_t order, sif_option_t opt) {

  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to sif_delta_moments_pk");
    return NULL;
  }

  if (__validate_order(order) != SIF_OK)
    return NULL;

  if (__validate_pk_table(k, pk, n_points) != SIF_OK)
    return NULL;

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG, "radius %u is %g, must be strictly positive", i,
        (double)radii[i]);
      return NULL;
    }
  }

  const bool gaussian =
    (opt & __SIF_DELTA_FILTER_MASK) == SIF_DELTA_FILTER_GAUSSIAN;

  sif_delta_moments_t* m = __sif_delta_moments_alloc(n_radii, order);
  if (!m)
    return NULL;

  memcpy(m->radii, radii, (size_t)n_radii * sizeof(real_t));

  SIF_LOG_INFO(__TAG,
    "evaluating sigma_0..sigma_%u over %u radii from a %u-point P(k) (%s "
    "window)",
    order, n_radii, n_points, gaussian ? "Gaussian" : "top-hat");

  /* sigma_j^2 = (1 / 2 pi^2) integral dk k^(2j+2) P(k) W^2(kR).
   *
   * Quadratured as a trapezoid in log k, since P(k) tables are sampled that
   * way and a linear-k trapezoid would badly under-resolve the low-k decade.
   * The change of variable contributes the extra factor of k. */
  const double prefactor = 1.0 / (2.0 * M_PI * M_PI);
  const double k_high = 0.5 * (double)k[n_points - 1];
  const uint8_t n_moments = m->n_moments;

#pragma omp parallel for schedule(static)
  for (uint32_t r = 0; r < n_radii; r++) {
    const double R = (double)radii[r];

    double total[SIF_MAX_MOMENT_ORDER + 1] = {0.0};
    double high[SIF_MAX_MOMENT_ORDER + 1] = {0.0};
    double prev[SIF_MAX_MOMENT_ORDER + 1] = {0.0};

    for (uint32_t i = 0; i < n_points; i++) {
      const double ki = (double)k[i];
      const double w = __window(gaussian, ki * R);

      /* k^2 * P(k) * W^2, times the k from d(ln k); the k^(2j) weight is
       * folded in one order at a time below. */
      const double base = ki * ki * ki * (double)pk[i] * w * w;
      const double dlnk = i > 0 ? log(ki) - log((double)k[i - 1]) : 0.0;

      double k2j = 1.0;
      for (uint8_t j = 0; j < n_moments; j++) {
        const double f = base * k2j;

        if (i > 0) {
          const double piece = 0.5 * (f + prev[j]) * dlnk;
          total[j] += piece;
          if (ki > k_high)
            high[j] += piece;
        }

        prev[j] = f;
        k2j *= ki * ki;
      }
    }

    /* The warnings live outside the parallel loop so their order is stable. */
    for (uint8_t j = 0; j < n_moments; j++) {
      const double s = total[j] * prefactor;
      const size_t at = m->offsets[j] + r;

      m->sigma[at] = (real_t)(s > 0.0 ? sqrt(s) : 0.0);
      m->high_k_fraction[at] = (real_t)(total[j] > 0.0 ? high[j] / total[j] : 0.0);
    }
  }

  for (uint32_t r = 0; r < n_radii; r++) {
    for (uint8_t j = 0; j < n_moments; j++) {
      const size_t at = m->offsets[j] + r;

      if (m->sigma[at] <= 0.0f) {
        SIF_LOG_WARNING(__TAG,
          "sigma_%u at radius %g integrated to zero; check that the P(k) table "
          "covers the relevant scales",
          j, (double)radii[r]);
      }
      if (m->high_k_fraction[at] > __HIGH_K_WARN_LEVEL) {
        SIF_LOG_WARNING(__TAG,
          "at R = %g, %.1f%% of the sigma_%u integral comes from the top half "
          "of the tabulated k range; the table is truncating the moment",
          (double)radii[r], 100.0 * (double)m->high_k_fraction[at], j);
      }
    }
  }

  SIF_LOG_INFO(__TAG, "sigma_0..sigma_%u evaluated from P(k)", order);

  return m;
}


/* --- Logarithmic slope of sigma --- */

/*
 * dW/dx at x = kR.
 *
 * The top-hat form has a cubic cancellation in the numerator over x^4, so
 * below the crossover the series is used instead: the closed form loses every
 * significant digit there while the two leading series terms are good to
 * around 1e-10.
 */
static inline double __window_slope(bool gaussian, double x) {
  if (gaussian)
    return -x * exp(-0.5 * x * x);

  if (x < 1e-2)
    return -x / 5.0 + x * x * x / 70.0;

  const double x2 = x * x;
  return 3.0 * (x2 * sin(x) - 3.0 * sin(x) + 3.0 * x * cos(x)) / (x2 * x2);
}

real_t* sif_sigma_slope_pk(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, sif_option_t opt) {

  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to sif_sigma_slope_pk");
    return NULL;
  }

  if (__validate_pk_table(k, pk, n_points) != SIF_OK)
    return NULL;

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG, "radius %u is %g, must be strictly positive", i,
        (double)radii[i]);
      return NULL;
    }
  }

  const bool gaussian =
    (opt & __SIF_DELTA_FILTER_MASK) == SIF_DELTA_FILTER_GAUSSIAN;

  real_t* out = sif_calloc_aligned((size_t)n_radii, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the slope array");
    return NULL;
  }

  /*
   * dln(sigma)/dln(R) = (1/2) dln(sigma^2)/dln(R), and differentiating
   * W^2(kR) under the integral gives 2 W W' (kR). The 1/2 and the 2 cancel,
   * as does the 1/(2 pi^2), leaving a ratio of two integrals over the same
   * measure -- which is why this needs no separate normalization.
   */
#pragma omp parallel for schedule(static)
  for (uint32_t r = 0; r < n_radii; r++) {
    const double R = (double)radii[r];

    double numerator = 0.0, denominator = 0.0;
    double prev_num = 0.0, prev_den = 0.0;

    for (uint32_t i = 0; i < n_points; i++) {
      const double ki = (double)k[i];
      const double x = ki * R;
      const double w = __window(gaussian, x);

      /* k^2 P W^2, times the k from d(ln k). */
      const double base = ki * ki * ki * (double)pk[i];

      const double den = base * w * w;
      const double num = base * w * __window_slope(gaussian, x) * x;

      if (i > 0) {
        const double dlnk = log(ki) - log((double)k[i - 1]);
        numerator += 0.5 * (num + prev_num) * dlnk;
        denominator += 0.5 * (den + prev_den) * dlnk;
      }

      prev_num = num;
      prev_den = den;
    }

    out[r] = (real_t)(denominator > 0.0 ? numerator / denominator : 0.0);
  }

  for (uint32_t r = 0; r < n_radii; r++) {
    if (out[r] == 0.0f) {
      SIF_LOG_WARNING(__TAG,
        "the slope at radius %g came out zero; the P(k) table may not cover "
        "the relevant scales",
        (double)radii[r]);
    }
  }

  return out;
}
