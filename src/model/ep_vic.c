/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Void-in-cloud for the emulator: the baseline of the factor K.
 *
 * K is the fraction of the voids at a radius whose walk was not inside a
 * cloud -- above the collapse barrier -- at any larger radius. Its baseline is
 * the expected number of collapse crossings at larger radii, conditional on
 * the walk making its void at this radius with a given slope, averaged over
 * the up-crossing slopes. A transcription of slope_collapse_hazard in
 * tools/ep_model.py, which the network was fitted against: every stencil,
 * floor and quadrature node below is part of the model, not a choice.
 */

#include "model/ep_internal.h"

#include "sif/model/delta_moments.h"

#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

#define TAG "ep_emu"

/* As in ep_model.py. */
#define VIC_RATE_FLOOR   1e-300
#define VIC_PINNED       1e-12
#define VIC_EXCESS_LIMIT 40.0
#define VIC_SLOPE_SPAN   8.0
#define VIC_SLOPE_NODES  8

/* SIF_PI is a sif_real; the density is taken in double, as numpy does. */
#define VIC_PI 3.14159265358979323846

/* Below this, exp() returns exactly zero in double precision. */
#define VIC_EXP_ZERO -746.0

/* 8-point Gauss-Legendre on [-1, 1]. */
static const double GL8_X[VIC_SLOPE_NODES] = {-0.9602898564975363,
  -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
  0.1834346424956498, 0.5255324099163290, 0.7966664774136267,
  0.9602898564975363};
static const double GL8_W[VIC_SLOPE_NODES] = {0.1012285362903763,
  0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
  0.3626837833783620, 0.3137066458778873, 0.2223810344533745,
  0.1012285362903763};

/* numpy.gradient with edge_order=2 at point i of v(x), the elements read with
 * a stride so a matrix row or column can be passed in place. */
static double gradient(
  const double* v, size_t stride, const double* x, uint32_t n, uint32_t i) {

  if (i == 0) {
    const double h1 = x[1] - x[0];
    const double h2 = x[2] - x[1];
    return -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * v[0] +
           (h1 + h2) / (h1 * h2) * v[stride] -
           h1 / (h2 * (h1 + h2)) * v[2 * stride];
  }
  if (i + 1 == n) {
    const double h1 = x[n - 2] - x[n - 3];
    const double h2 = x[n - 1] - x[n - 2];
    return h2 / (h1 * (h1 + h2)) * v[(n - 3) * stride] -
           (h1 + h2) / (h1 * h2) * v[(n - 2) * stride] +
           (h1 + 2.0 * h2) / (h2 * (h1 + h2)) * v[(n - 1) * stride];
  }
  const double h1 = x[i] - x[i - 1];
  const double h2 = x[i + 1] - x[i];
  return (h1 * h1 * v[(i + 1) * stride] - h2 * h2 * v[(i - 1) * stride] -
           (h1 * h1 - h2 * h2) * v[i * stride]) /
         (h1 * h2 * (h1 + h2));
}

/* dv/dS through the logarithms: (v / S) d ln v / d ln S. */
static void log_slope(const double* v, const double* S, uint32_t n, double* lnv,
  double* lnS, double* out) {

  for (uint32_t i = 0; i < n; i++) {
    lnv[i] = log(v[i]);
    lnS[i] = log(S[i]);
  }
  for (uint32_t i = 0; i < n; i++)
    out[i] = v[i] / S[i] * gradient(lnv, 1, lnS, n, i);
}

int sif__ep_vic_hazard_value(const double* cov, const double* S,
  const double* B, const double* C, const double* dvar, uint32_t n,
  double* out) {

  double* work = malloc((size_t)5 * n * sizeof(double));
  if (!work) {
    SIF_LOG_ERROR(TAG, "failed to allocate the void-in-cloud workspace");
    return SIF_ERR_ALLOC;
  }
  double* dC = work;
  double* lnv = work + n;
  double* lnS = work + 2 * (size_t)n;
  double* col = work + 3 * (size_t)n; /* C(s_j, S_i), j > i */
  double* rate = work + 4 * (size_t)n;

  log_slope(C, S, n, lnv, lnS, dC);

  for (uint32_t i = 0; i < n; i++)
    out[i] = 0.0;

  for (uint32_t i = 0; i + 3 < n; i++) {
    const uint32_t L = n - 1 - i;
    const double* s = S + i + 1;
    for (uint32_t k = 0; k < L; k++)
      col[k] = cov[SIF_COV_INDEX(i + 1 + k, i)];
    const double Si = S[i];
    for (uint32_t k = 0; k < L; k++) {
      const uint32_t j = i + 1 + k;
      const double c = col[k];
      const double dc = gradient(col, 1, s, L, k);
      const double m = -B[i] * c / Si;
      const double v_raw = s[k] - c * c / Si;
      double r = 0.0;
      if (v_raw > VIC_PINNED * s[k]) {
        const double v = v_raw > VIC_RATE_FLOOR ? v_raw : VIC_RATE_FLOOR;
        const double mp = -B[i] * dc / Si;
        double vp = dvar[j] - dc * dc / Si;
        if (vp < VIC_RATE_FLOOR)
          vp = VIC_RATE_FLOOR;
        const double cv = 0.5 - c * dc / Si;
        const double mu = mp + cv / v * (C[j] - m) - dC[j];
        double v2 = vp - cv * cv / v;
        if (v2 < VIC_RATE_FLOOR)
          v2 = VIC_RATE_FLOOR;
        const double sd = sqrt(v2);
        const double d = C[j] - m;
        const double e = -0.5 * d * d / v;
        if (e > VIC_EXP_ZERO) {
          const double dens = exp(e) / sqrt(2.0 * VIC_PI * v);
          const double y = -mu / sd;
          r = dens * (fabs(y) <= VIC_EXCESS_LIMIT ? sd * sif__ep_mean_excess(y)
                                                  : (mu > 0.0 ? mu : 0.0));
        }
      }
      rate[k] = r;
    }
    double lam = 0.0;
    for (uint32_t k = 0; k + 1 < L; k++)
      lam += 0.5 * (rate[k + 1] + rate[k]) * (s[k] - s[k + 1]);
    out[i] = lam;
  }

  free(work);
  return SIF_OK;
}

int sif__ep_vic_hazard(const double* cov, const double* S, const double* B,
  const double* C, const double* dvar, uint32_t n, double* out) {

  const size_t nn = (size_t)n * n;
  double* M = malloc(nn * sizeof(double));  /* C(s_j, S_i) at [j * n + i] */
  double* A = malloc(nn * sizeof(double));  /* dC/dS, along i */
  double* Dm = malloc(nn * sizeof(double)); /* dC/ds, along j */
  double* E = malloc(nn * sizeof(double));  /* d2C/ds dS */
  double* work = malloc((size_t)4 * n * sizeof(double));

  if (!M || !A || !Dm || !E || !work) {
    SIF_LOG_ERROR(TAG, "failed to allocate the void-in-cloud workspace");
    free(M);
    free(A);
    free(Dm);
    free(E);
    free(work);
    return SIF_ERR_ALLOC;
  }

  double* dB = work;
  double* dC = work + n;
  double* lnv = work + 2 * (size_t)n;
  double* lnS = work + 3 * (size_t)n;

  for (uint32_t j = 0; j < n; j++)
    for (uint32_t i = 0; i < n; i++)
      M[(size_t)j * n + i] =
        j >= i ? cov[SIF_COV_INDEX(j, i)] : cov[SIF_COV_INDEX(i, j)];

  /* np.gradient(M, S, axis=1), (axis=0), and of A along axis 0. */
#pragma omp parallel for schedule(static) if (n > 64)
  for (uint32_t j = 0; j < n; j++)
    for (uint32_t i = 0; i < n; i++)
      A[(size_t)j * n + i] = gradient(M + (size_t)j * n, 1, S, n, i);
#pragma omp parallel for schedule(static) if (n > 64)
  for (uint32_t j = 0; j < n; j++)
    for (uint32_t i = 0; i < n; i++) {
      Dm[(size_t)j * n + i] = gradient(M + i, n, S, n, j);
      E[(size_t)j * n + i] = gradient(A + i, n, S, n, j);
    }

  log_slope(B, S, n, lnv, lnS, dB);
  log_slope(C, S, n, lnv, lnS, dC);

  for (uint32_t i = 0; i < n; i++)
    out[i] = 0.0;

  /* Each radius independent of the others, and each written once: the same
   * answer, to the bit, whatever the thread count. Dynamic, because the work
   * per radius falls with i. */
#pragma omp parallel for schedule(dynamic, 4) if (n > 64)
  for (uint32_t i = 0; i < n - 3; i++) {
    const double Si = S[i];
    const double det = Si * dvar[i] - 0.25;
    if (!(det > 0.0))
      continue;
    /* Sigma_Z^-1 for Z = (x_S, x'_S). */
    const double z00 = dvar[i] / det, z01 = -0.5 / det, z11 = Si / det;

    const double mu_u = B[i] * 0.5 / Si;
    double var_u = dvar[i] - 0.25 / Si;
    if (var_u < VIC_RATE_FLOOR)
      var_u = VIC_RATE_FLOOR;
    const double sg_u = sqrt(var_u);
    const double lo = dB[i];
    const double hi = (mu_u > lo ? mu_u : lo) + VIC_SLOPE_SPAN * sg_u;

    double u[VIC_SLOPE_NODES], w[VIC_SLOPE_NODES], lam_u[VIC_SLOPE_NODES];
    double w_sum = 0.0;
    for (int q = 0; q < VIC_SLOPE_NODES; q++) {
      u[q] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * GL8_X[q];
      const double t = (u[q] - mu_u) / sg_u;
      w[q] = GL8_W[q] * (u[q] - lo) * exp(-0.5 * t * t);
      w_sum += w[q];
      lam_u[q] = 0.0;
    }
    if (!(w_sum > 0.0))
      continue;

    /* Over the larger radii once, all slope nodes together: the conditional
     * moments depend on the pair of scales alone. The trapezoid in s runs from
     * S_{i+1} out to the origin, term by term in the order it always had. */
    double prev[VIC_SLOPE_NODES];
    for (uint32_t j = i + 1; j < n; j++) {
      const size_t ji = (size_t)j * n + i;
      const double s = S[j];
      const double a0 = M[ji], a1 = A[ji], b0 = Dm[ji], b1 = E[ji];
      const double g00 = a0 * z00 + a1 * z01;
      const double g01 = a0 * z01 + a1 * z11;
      const double g10 = b0 * z00 + b1 * z01;
      const double g11 = b0 * z01 + b1 * z11;
      double c11 = s - (g00 * a0 + g01 * a1);
      const double c12 = 0.5 - (g00 * b0 + g01 * b1);
      const double c22 = dvar[j] - (g10 * b0 + g11 * b1);
      const int pinned = !(c11 > VIC_PINNED * s);
      if (c11 < VIC_RATE_FLOOR)
        c11 = VIC_RATE_FLOOR;
      double v2 = c22 - c12 * c12 / c11;
      if (v2 < VIC_RATE_FLOOR)
        v2 = VIC_RATE_FLOOR;
      const double sd2 = sqrt(v2);
      const double lev = -C[j];
      const double root = sqrt(2.0 * VIC_PI * c11);

      for (int q = 0; q < VIC_SLOPE_NODES; q++) {
        double r = 0.0;
        if (!pinned) {
          const double m1 = g00 * B[i] + g01 * u[q];
          const double d = lev - m1;
          const double e = -0.5 * d * d / c11;
          /* exp() of anything below -746 is exactly zero, and so is the rate:
           * skipping the mean excess there changes no bit of the answer. */
          if (e > VIC_EXP_ZERO) {
            const double m2 = g10 * B[i] + g11 * u[q];
            const double g = -dC[j] - (m2 + c12 / c11 * d);
            const double y = -g / sd2;
            const double excess = fabs(y) <= VIC_EXCESS_LIMIT
                                    ? sd2 * sif__ep_mean_excess(y)
                                    : (g > 0.0 ? g : 0.0);
            r = exp(e) / root * excess;
          }
        }
        if (j > i + 1)
          lam_u[q] += 0.5 * (r + prev[q]) * (S[j - 1] - S[j]);
        prev[q] = r;
      }
    }

    double lam_min = lam_u[0];
    for (int q = 1; q < VIC_SLOPE_NODES; q++)
      if (lam_u[q] < lam_min)
        lam_min = lam_u[q];
    /* -ln E_u[exp(-Lambda(u))]: shifted so a large Lambda cannot underflow,
     * and through expm1/log1p so a tiny one is not lost to rounding. */
    double acc = 0.0;
    for (int q = 0; q < VIC_SLOPE_NODES; q++)
      acc += w[q] / w_sum * expm1(-(lam_u[q] - lam_min));
    out[i] = lam_min - log1p(acc);
  }

  free(M);
  free(A);
  free(Dm);
  free(E);
  free(work);
  return SIF_OK;
}

#undef TAG
