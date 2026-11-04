/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The variance of the smoothed field as a precomputed linear map from P(k).
 *
 * With x = ln k and Delta(k) = k^3 / 2 pi^2, the three integrals are
 *
 *     sigma^2(R)              = int dx Delta W^2(kR)          P(k)
 *     dS/dR                   = int dx Delta 2 k W W'(kR)      P(k)
 *     <(d delta / dR)^2>      = int dx Delta k^2 W'(kR)^2      P(k)
 *
 * and P(k) between the nodes is the cubic in ln k through the four nearest
 * nodes, sum_m P_m L_m(x). Substituting gives weights K_jm, one integral of a
 * window term against a basis function per (radius, node), and every later
 * spectrum costs sum_m K_jm P_m.
 *
 * The weights are integrated with 4-point Gauss-Legendre on sub-intervals of
 * each node interval, cut so that neither ln k nor the phase kR moves far
 * across one. The phase is what matters at large k: W' goes as sin(kR) / kR
 * there, its square turns over every pi / R in k, and the sub-intervals follow
 * it at each radius separately rather than at the largest one, which keeps
 * the build cost proportional to the sum of the lattice radii rather than to
 * their number times the largest.
 */

#include "sif/model/variance_kernel.h"

#include "sif/model/delta_moments.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "utils/logger_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define TAG "delta"

/*
 * Largest sub-interval, in ln k and in phase kR. Against the same kernel built
 * with both halved, these leave the derivative variance within 1e-9 on a CDM
 * spectrum: the phase limit is a quarter period of sin^2(kR), the ln k limit
 * matters only where kR < 1 and the window is flat.
 */
#define SUB_DLNK  0.02
#define SUB_PHASE (0.25 * 3.14159265358979323846)

/* Rows of the weight matrix per lattice radius: sigma^2, dS/dR and
 * <(d delta / dR)^2>. A fourth, the last over the top half of the k range
 * only, is the truncation diagnostic, kept for the smallest radius alone:
 * that is where the high-k tail weighs most, and a full row per radius would
 * be a third more memory to stream on every spectrum. */
#define N_ROWS 3

/* Fraction of <(d delta / dR)^2> from the top half of the k range above
 * which the range is reported as truncating it, as sif_delta_covariance_pk
 * does. */
#define HIGH_K_WARN_LEVEL 0.1

/* 4-point Gauss-Legendre on [-1, 1]. */
static const double GL_X[4] = {-0.8611363115940526, -0.3399810435848563,
  0.3399810435848563, 0.8611363115940526};
static const double GL_W[4] = {0.3478548451374538, 0.6521451548625461,
  0.6521451548625461, 0.3478548451374538};

/*
 * W(x) and dW/dx at x = kR together, from one sine and one cosine. The
 * series below the crossovers are the ones delta_moments.c uses: the closed
 * forms cancel to nothing at small x.
 */
static inline void window_pair(bool gaussian, double x, double* w, double* wp) {

  if (gaussian) {
    *w = exp(-0.5 * x * x);
    *wp = -x * *w;
    return;
  }

  if (x < 1e-2) {
    *w = 1.0 - x * x / 10.0;
    *wp = -x / 5.0 + x * x * x / 70.0;
    return;
  }

  const double s = sin(x);
  const double c = cos(x);
  const double x2 = x * x;
  *w = 3.0 * (s - x * c) / (x2 * x);
  *wp = 3.0 * (x2 * s - 3.0 * s + 3.0 * x * c) / (x2 * x2);
}

/* --- The kernel --- */

static int validate_kernel_args(const sif_real* k, uint32_t n_k, sif_real r_min,
  sif_real r_max, sif_real dln_r) {

  if (!k || n_k < 4) {
    SIF_LOG_ERROR(TAG,
      "k: %u nodes (expected at least 4: P(k) between nodes is the cubic "
      "through the four nearest)",
      k ? n_k : 0);
    return SIF_ERR_INVALID;
  }

  for (uint32_t m = 0; m < n_k; m++) {
    if (!(k[m] > 0.0f) || !isfinite((double)k[m])) {
      SIF_LOG_ERROR(
        TAG, "k[%u]: %g (expected positive and finite)", m, (double)k[m]);
      return SIF_ERR_INVALID;
    }
    if (m > 0 && !(k[m] > k[m - 1])) {
      SIF_LOG_ERROR(TAG, "k[%u]: %g after %g (expected strictly increasing)", m,
        (double)k[m], (double)k[m - 1]);
      return SIF_ERR_INVALID;
    }
  }

  if (!(r_min > 0.0f) || !(r_max > r_min) || !isfinite((double)r_max)) {
    SIF_LOG_ERROR(TAG,
      "radius lattice: r_min %g, r_max %g (expected 0 < r_min < r_max)",
      (double)r_min, (double)r_max);
    return SIF_ERR_INVALID;
  }

  if (!(dln_r > 0.0f) || !isfinite((double)dln_r)) {
    SIF_LOG_ERROR(
      TAG, "dln_r: %g (expected positive and finite)", (double)dln_r);
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/*
 * One lattice radius: every node interval, cut into sub-intervals, four
 * Gauss-Legendre points each, accumulated into this radius's rows. Rows of
 * different radii never overlap, which is what lets the caller run radii in
 * parallel with a result independent of the thread count.
 */
static void kernel_row(const double* logk, const sif_real* k, uint32_t n_k,
  double R, bool gaussian, double k_high, double* rows[N_ROWS], double* high) {

  const double prefactor =
    1.0 / (2.0 * 3.14159265358979323846 * 3.14159265358979323846);

  for (uint32_t iv = 0; iv + 1 < n_k; iv++) {
    const double a = logk[iv];
    const double b = logk[iv + 1];
    const double span_k = (double)k[iv + 1] - (double)k[iv];

    double n_sub = ceil((b - a) / SUB_DLNK);
    const double n_phase = ceil(span_k * R / SUB_PHASE);
    if (n_phase > n_sub)
      n_sub = n_phase;
    if (n_sub < 1.0)
      n_sub = 1.0;

    /* The four nodes the cubic through this interval runs through: the
     * interval's ends and one either side, shifted inward at the ends of the
     * table, so x always lies inside the stencil. */
    uint32_t s = (iv > 0) ? iv - 1 : 0;
    if (s + 4 > n_k)
      s = n_k - 4;

    const double* xs = logk + s;
    double denom[4];
    for (int p = 0; p < 4; p++) {
      denom[p] = 1.0;
      for (int q = 0; q < 4; q++)
        if (q != p)
          denom[p] *= xs[p] - xs[q];
    }

    const double h = (b - a) / n_sub;
    const uint32_t n_steps = (uint32_t)n_sub;

    for (uint32_t step = 0; step < n_steps; step++) {
      const double mid = a + (step + 0.5) * h;

      for (int g = 0; g < 4; g++) {
        const double x = mid + 0.5 * h * GL_X[g];
        const double kk = exp(x);
        const double weight = GL_W[g] * 0.5 * h * kk * kk * kk * prefactor;

        double w, wp;
        window_pair(gaussian, kk * R, &w, &wp);

        const double f0 = weight * w * w;
        const double f1 = weight * 2.0 * kk * w * wp;
        const double f2 = weight * kk * kk * wp * wp;

        for (int p = 0; p < 4; p++) {
          double basis = 1.0;
          for (int q = 0; q < 4; q++)
            if (q != p)
              basis *= x - xs[q];
          basis /= denom[p];

          rows[0][s + p] += f0 * basis;
          rows[1][s + p] += f1 * basis;
          rows[2][s + p] += f2 * basis;
          if (high && kk > k_high)
            high[s + p] += f2 * basis;
        }
      }
    }
  }
}

sif_variance_kernel_t* sif_variance_kernel_alloc(const sif_real* k,
  uint32_t n_k, sif_real r_min, sif_real r_max, sif_real dln_r,
  sif_option opt) {

  if (validate_kernel_args(k, n_k, r_min, r_max, dln_r) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  /* The small slack keeps r_max itself on the lattice when it is meant to
   * be, despite the rounding in the logarithm. */
  const double span = log((double)r_max / (double)r_min) / (double)dln_r;
  const double n_lattice = floor(span + 1e-9) + 1.0;

  if (n_lattice < 4.0 || n_lattice > (double)SIF_COV_MAX_RADII) {
    SIF_LOG_ERROR(TAG,
      "radius lattice %g to %g at dln_r %g: %.0f radii (expected 4 to %d)",
      (double)r_min, (double)r_max, (double)dln_r, n_lattice,
      SIF_COV_MAX_RADII);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  const uint32_t n_radii = (uint32_t)n_lattice;

  sif_variance_kernel_t* kernel = calloc(1, sizeof(*kernel));
  double* logk = malloc((size_t)n_k * sizeof(double));
  if (!kernel || !logk) {
    SIF_LOG_ERROR(TAG, "failed to allocate the variance kernel");
    sif__error_status_set(SIF_ERR_ALLOC);
    free(kernel);
    free(logk);
    return NULL;
  }

  kernel->n_k = n_k;
  kernel->n_radii = n_radii;
  kernel->dln_r = (double)dln_r;
  kernel->window = opt & SIF__DELTA_FILTER_MASK;
  kernel->k = sif_malloc_aligned((size_t)n_k * sizeof(sif_real));
  kernel->radii = sif_malloc_aligned((size_t)n_radii * sizeof(sif_real));
  kernel->_weights =
    sif_calloc_aligned(((size_t)N_ROWS * n_radii + 1) * n_k, sizeof(double));

  if (!kernel->k || !kernel->radii || !kernel->_weights) {
    SIF_LOG_ERROR(TAG,
      "failed to allocate the variance kernel (%u radii x %u nodes)", n_radii,
      n_k);
    sif__error_status_set(SIF_ERR_ALLOC);
    free(logk);
    sif_variance_kernel_free(kernel);
    return NULL;
  }

  memcpy(kernel->k, k, (size_t)n_k * sizeof(sif_real));
  for (uint32_t m = 0; m < n_k; m++)
    logk[m] = log((double)k[m]);

  /* The weights are taken at the radii as stored, so that a caller reading
   * kernel->radii gets the radius the variance was computed at, not one a
   * rounding away from it. */
  for (uint32_t j = 0; j < n_radii; j++)
    kernel->radii[j] =
      (sif_real)((double)r_min * exp((double)j * (double)dln_r));

  const bool gaussian = kernel->window == SIF_DELTA_FILTER_GAUSSIAN;
  const double k_high = 0.5 * (double)k[n_k - 1];

  SIF_LOG_TRACE(TAG,
    "building the variance kernel: %u radii from %g to %g, %u nodes from %g "
    "to %g (%s window)",
    n_radii, (double)kernel->radii[0], (double)kernel->radii[n_radii - 1], n_k,
    (double)k[0], (double)k[n_k - 1], gaussian ? "Gaussian" : "top-hat");

  /* Cost grows with the radius, so the largest radii are dealt out first. */
#pragma omp parallel for schedule(dynamic, 1)
  for (uint32_t jj = 0; jj < n_radii; jj++) {
    const uint32_t j = n_radii - 1 - jj;
    double* rows[N_ROWS];
    for (int r = 0; r < N_ROWS; r++)
      rows[r] = kernel->_weights + ((size_t)r * n_radii + j) * n_k;

    double* high =
      (j == 0) ? kernel->_weights + (size_t)N_ROWS * n_radii * n_k : NULL;

    kernel_row(
      logk, k, n_k, (double)kernel->radii[j], gaussian, k_high, rows, high);
  }

  free(logk);
  return kernel;
}

void sif_variance_kernel_free(sif_variance_kernel_t* kernel) {
  if (!kernel)
    return;
  sif_free_aligned(kernel->k);
  sif_free_aligned(kernel->radii);
  sif_free_aligned(kernel->_weights);
  free(kernel);
}

/* --- The walk --- */

/* Four partial sums, combined in a fixed order: the compiler may not reorder a
 * floating-point reduction itself, so without them the loop runs one
 * multiply-add at a time. The answer still depends on nothing but the data. */
static double dot(const double* w, const double* p, uint32_t n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  uint32_t m = 0;
  for (; m + 4 <= n; m += 4) {
    s0 += w[m] * p[m];
    s1 += w[m + 1] * p[m + 1];
    s2 += w[m + 2] * p[m + 2];
    s3 += w[m + 3] * p[m + 3];
  }
  for (; m < n; m++)
    s0 += w[m] * p[m];
  return (s0 + s1) + (s2 + s3);
}

sif_walk_t* sif_walk_pk(
  const sif_variance_kernel_t* kernel, const sif_real* pk) {

  if (!kernel || !pk) {
    SIF_LOG_ERROR(TAG, "sif_walk_pk: kernel and pk are required");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  const uint32_t n_k = kernel->n_k;
  const uint32_t n = kernel->n_radii;

  for (uint32_t m = 0; m < n_k; m++) {
    if (!(pk[m] >= 0.0f) || !isfinite((double)pk[m])) {
      SIF_LOG_ERROR(TAG,
        "pk[%u] at k = %g: %g (expected non-negative and finite)", m,
        (double)kernel->k[m], (double)pk[m]);
      sif__error_status_set(SIF_ERR_INVALID);
      return NULL;
    }
  }

  sif_walk_t* walk = calloc(1, sizeof(*walk));
  double* p = malloc((size_t)n_k * sizeof(double));
  double* rows = malloc(((size_t)N_ROWS * n + 1) * sizeof(double));
  if (!walk || !p || !rows) {
    SIF_LOG_ERROR(TAG, "failed to allocate the walk");
    sif__error_status_set(SIF_ERR_ALLOC);
    free(walk);
    free(p);
    free(rows);
    return NULL;
  }

  walk->n_radii = n;
  walk->n_k = n_k;
  walk->window = kernel->window;
  walk->radii = sif_malloc_aligned((size_t)n * sizeof(sif_real));
  walk->sigma = sif_malloc_aligned((size_t)n * sizeof(sif_real));
  walk->deriv_variance = sif_malloc_aligned((size_t)n * sizeof(double));
  walk->k = sif_malloc_aligned((size_t)n_k * sizeof(sif_real));
  walk->pk = sif_malloc_aligned((size_t)n_k * sizeof(sif_real));

  if (!walk->radii || !walk->sigma || !walk->deriv_variance || !walk->k ||
      !walk->pk) {
    SIF_LOG_ERROR(TAG, "failed to allocate the walk (%u radii)", n);
    sif__error_status_set(SIF_ERR_ALLOC);
    free(p);
    free(rows);
    sif_walk_free(walk);
    return NULL;
  }

  memcpy(walk->radii, kernel->radii, (size_t)n * sizeof(sif_real));
  memcpy(walk->k, kernel->k, (size_t)n_k * sizeof(sif_real));
  memcpy(walk->pk, pk, (size_t)n_k * sizeof(sif_real));

  for (uint32_t m = 0; m < n_k; m++)
    p[m] = (double)pk[m];

  /* Each row is one dot product summed in a fixed order, so the result does
   * not depend on how the rows are shared out. Below a few hundred thousand
   * multiply-adds the threads cost more than they save. */
  const uint32_t n_rows = N_ROWS * n + 1;
  const double work = (double)n_rows * (double)n_k;
#pragma omp parallel for schedule(static) if (work > 2e5)
  for (uint32_t r = 0; r < n_rows; r++)
    rows[r] = dot(kernel->_weights + (size_t)r * n_k, p, n_k);

  free(p);

  const double* S = rows;
  const double* ds_dr = rows + n;
  const double* dd_dr2 = rows + 2 * (size_t)n;
  const double high = rows[(size_t)N_ROWS * n];

  for (uint32_t j = 0; j < n; j++) {
    if (!(S[j] > 0.0) || !(ds_dr[j] != 0.0) || !(dd_dr2[j] > 0.0)) {
      SIF_LOG_ERROR(TAG,
        "walk at R = %g: sigma^2 %g, dS/dR %g, <(d delta / dR)^2> %g "
        "(expected nonzero; P(k) has no support on the scales of this radius)",
        (double)walk->radii[j], S[j], ds_dr[j], dd_dr2[j]);
      sif__error_status_set(SIF_ERR_RANGE);
      free(rows);
      sif_walk_free(walk);
      return NULL;
    }

    /* delta' = (d delta / dR) / (dS / dR) by the chain rule. */
    walk->sigma[j] = (sif_real)sqrt(S[j]);
    walk->deriv_variance[j] = dd_dr2[j] / (ds_dr[j] * ds_dr[j]);
  }

  /*
   * The one way this can still go wrong without the input being invalid: the
   * derivative variance falls only as P(k) sin^2(kR) at large k, so a k range
   * that converges sigma can stop short for it. Checked at the smallest
   * radius, where the tail weighs most, so it is one line per spectrum at
   * most rather than one per radius.
   */
  const double frac = high / dd_dr2[0];
  if (frac > HIGH_K_WARN_LEVEL) {
    SIF_LOG_WARNING(TAG,
      "walk at R = %g: %.1f%% of <(d delta / dR)^2> from k > %g, the top half "
      "of the k range (expected below %.0f%%; extend k upward)",
      (double)walk->radii[0], 100.0 * frac, 0.5 * (double)kernel->k[n_k - 1],
      100.0 * HIGH_K_WARN_LEVEL);
  }

  free(rows);
  return walk;
}

void sif_walk_free(sif_walk_t* walk) {
  if (!walk)
    return;
  sif_free_aligned(walk->radii);
  sif_free_aligned(walk->sigma);
  sif_free_aligned(walk->deriv_variance);
  sif_free_aligned(walk->k);
  sif_free_aligned(walk->pk);
  free(walk);
}

#undef TAG
#undef SUB_DLNK
#undef SUB_PHASE
#undef N_ROWS
#undef HIGH_K_WARN_LEVEL
