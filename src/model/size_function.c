/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Excursion-set void size functions: Sheth & van de Weygaert (2004) and the
 * volume-conserving Vdn variant, in the notation of Jennings, Li & Hu, MNRAS
 * 434 (2013) 2167, arXiv:1304.6087.
 *
 * Both models share one multiplicity function and one Lagrangian abundance.
 * They part company only at the mapping to Eulerian radii: SvdW conserves
 * number, Vdn conserves volume, and in the code that is the single choice of
 * which radius goes into the volume denominator.
 */

#include "sif/model/size_function.h"

#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/results_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define TAG "vsf_model"

/* --- Spherical evolution --- */

/*
 * Both contrasts are functions of the single parameter eta of the
 * Einstein-de Sitter expansion solution, so converting between them means
 * eliminating eta, which has no closed form.
 */

/* Below this the closed forms lose more than half their digits to
 * cancellation, while the series are exact to double precision. */
#define EDS_SERIES_SWITCH 1e-2

#define EDS_ETA_MIN 1e-8
#define EDS_ETA_MAX 30.0

#define BRENT_TOL      1e-12
#define BRENT_MAX_ITER 100

/* sinh(eta) - eta, whose leading terms cancel exactly. */
static double sinh_minus(double eta) {
  if (eta < EDS_SERIES_SWITCH) {
    const double e2 = eta * eta;
    return eta * e2 / 6.0 * (1.0 + e2 / 20.0 + e2 * e2 / 840.0);
  }
  return sinh(eta) - eta;
}

/* delta_L(eta), decreasing from 0. */
static double eds_linear(double eta) {
  return -0.15 * pow(6.0 * sinh_minus(eta), 2.0 / 3.0);
}

/* delta_NL(eta), decreasing from 0 to -1. */
static double eds_nonlinear(double eta) {
  if (eta < EDS_SERIES_SWITCH) {
    /* (9/2) (eta^3/6)^2 / (eta^2/2)^3 is exactly 1, so the ratio is taken
     * between the bracketed corrections rather than the raw differences,
     * which would cancel away nine digits before dividing. */
    const double e2 = eta * eta;
    const double num = 1.0 + e2 / 20.0 + e2 * e2 / 840.0;
    const double den = 1.0 + e2 / 12.0 + e2 * e2 / 360.0;
    return num * num / (den * den * den) - 1.0;
  }
  const double s = sinh(eta) - eta;
  const double c = cosh(eta) - 1.0;
  return 4.5 * s * s / (c * c * c) - 1.0;
}

/* Brent's method on g(eta) == target. Both g above are smooth and strictly
 * monotone. */
static int brent(
  double (*g)(double), double target, double lo, double hi, double* root) {

  double a = lo, b = hi;
  double fa = g(a) - target;
  double fb = g(b) - target;

  if (!(fa * fb < 0.0))
    return SIF_ERR_RANGE;

  if (fabs(fa) < fabs(fb)) {
    double t = a;
    a = b;
    b = t;
    t = fa;
    fa = fb;
    fb = t;
  }

  double c = a, fc = fa, d = 0.0;
  bool bisected = true;

  for (int it = 0; it < BRENT_MAX_ITER; it++) {
    if (fb == 0.0 || fabs(b - a) < BRENT_TOL * (fabs(b) + 1.0))
      break;

    double s;
    if (fa != fc && fb != fc) {
      s = a * fb * fc / ((fa - fb) * (fa - fc)) +
          b * fa * fc / ((fb - fa) * (fb - fc)) +
          c * fa * fb / ((fc - fa) * (fc - fb));
    } else {
      s = b - fb * (b - a) / (fb - fa);
    }

    const double bound = 0.25 * (3.0 * a + b);
    bool take_bisection = !(s > fmin(bound, b) && s < fmax(bound, b)) ||
                          (bisected && fabs(s - b) >= 0.5 * fabs(b - c)) ||
                          (!bisected && fabs(s - b) >= 0.5 * fabs(c - d)) ||
                          (bisected && fabs(b - c) < BRENT_TOL) ||
                          (!bisected && fabs(c - d) < BRENT_TOL);

    if (take_bisection)
      s = 0.5 * (a + b);
    bisected = take_bisection;

    const double fs = g(s) - target;
    d = c;
    c = b;
    fc = fb;

    if (fa * fs < 0.0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }

    if (fabs(fa) < fabs(fb)) {
      double t = a;
      a = b;
      b = t;
      t = fa;
      fa = fb;
      fb = t;
    }
  }

  *root = b;
  return SIF_OK;
}

sif_real sif_spherical_map_nonlinear(sif_real delta_linear, sif_option opt) {
  const double d = (double)delta_linear;

  if (!(d < 0.0)) {
    SIF_LOG_ERROR(TAG,
      "delta_linear is %g, must be strictly negative; only the expanding "
      "branch of the spherical solution is implemented",
      d);
    return (sif_real)0.0;
  }

  if ((opt & SIF__SPHERICAL_MASK) == SIF_SPHERICAL_B94) {
    const double c = SIF_SPHERICAL_EXPANSION_C;
    return (sif_real)(pow(1.0 - d / c, -c) - 1.0);
  }

  double eta;
  if (brent(eds_linear, d, EDS_ETA_MIN, EDS_ETA_MAX, &eta) != SIF_OK) {
    SIF_LOG_ERROR(TAG,
      "delta_linear = %g lies outside the range the parametric solution is "
      "bracketed over",
      d);
    return (sif_real)0.0;
  }

  return (sif_real)eds_nonlinear(eta);
}

sif_real sif_spherical_map_linear(sif_real delta_nonlinear, sif_option opt) {
  const double d = (double)delta_nonlinear;

  if (!(d < 0.0) || !(d > -1.0)) {
    SIF_LOG_ERROR(TAG,
      "delta_nonlinear is %g, must lie strictly between -1 and 0; -1 is total "
      "evacuation and is reached only asymptotically",
      d);
    return (sif_real)0.0;
  }

  if ((opt & SIF__SPHERICAL_MASK) == SIF_SPHERICAL_B94) {
    const double c = SIF_SPHERICAL_EXPANSION_C;
    return (sif_real)(c * (1.0 - pow(1.0 + d, -1.0 / c)));
  }

  double eta;
  if (brent(eds_nonlinear, d, EDS_ETA_MIN, EDS_ETA_MAX, &eta) != SIF_OK) {
    SIF_LOG_ERROR(TAG,
      "delta_nonlinear = %g lies outside the range the parametric solution is "
      "bracketed over",
      d);
    return (sif_real)0.0;
  }

  return (sif_real)eds_linear(eta);
}

/* --- The two size functions --- */

/*
 * Both models evaluate the Lagrangian abundance
 *
 *     dn_L/dln r_L = f_ln(sigma) / V(r_L) * dln(sigma^-1)/dln r_L
 *
 * at r_L = r / F, and report it against the Eulerian r. SvdW conserves number
 * and leaves the amplitude alone; Vdn conserves volume and therefore divides
 * by V(r) instead of V(r_L). With a constant F the Jacobian dln r_L/dln r is
 * one, so the two differ by exactly F^3.
 */
static sif_size_function_t* size_function(const sif_real* k, const sif_real* pk,
  uint32_t n_points, const sif_real* radii, uint32_t n_radii, sif_real delta_v,
  sif_real delta_c, sif_option opt, bool conserve_volume) {

  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "no radii to evaluate");
    return NULL;
  }

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(
        TAG, "radius %u is %g, must be strictly positive", i, (double)radii[i]);
      return NULL;
    }
  }

  /* The expansion factor, (1 + delta_NL)^(-1/3): the same mass occupies
   * 1/(1 + delta_NL) times the volume. */
  const sif_real delta_nl = sif_spherical_map_nonlinear(delta_v, opt);
  if (!(delta_nl < 0.0f)) /* outside the expanding branch; already logged */
    return NULL;
  const double factor = pow(1.0 + (double)delta_nl, -1.0 / 3.0);

  /* The barriers are calibrated against top-hat smoothing, so the window is
   * not the caller's to choose here. */
  const sif_option moment_opt = SIF_DELTA_FILTER_TOP_HAT;

  sif_real* lagrangian = malloc((size_t)n_radii * sizeof(sif_real));
  if (!lagrangian) {
    SIF_LOG_ERROR(TAG, "failed to allocate the Lagrangian radii");
    return NULL;
  }
  for (uint32_t i = 0; i < n_radii; i++)
    lagrangian[i] = (sif_real)((double)radii[i] / factor);

  sif_delta_moments_t* moments =
    sif_delta_moments_pk(k, pk, n_points, lagrangian, n_radii, 0, moment_opt);
  sif_real* slope =
    sif_delta_sigma_slope_pk(k, pk, n_points, lagrangian, n_radii, moment_opt);

  if (!moments || !slope) {
    free(lagrangian);
    sif_delta_moments_free(moments);
    sif_free_aligned(slope);
    return NULL;
  }

  const sif_real* sigma = sif_delta_moments_sigma(moments, 0);
  sif_real* f =
    sif_svdw_multiplicity_function(sigma, n_radii, delta_v, delta_c);

  sif_size_function_t* out = sif__size_function_alloc(n_radii);

  if (!f || !out) {
    free(lagrangian);
    sif_delta_moments_free(moments);
    sif_free_aligned(slope);
    sif_free_aligned(f);
    sif_size_function_free(out);
    return NULL;
  }

  out->options = opt;
  out->r_min = radii[0];
  out->r_max = radii[n_radii - 1];
  memcpy(out->r_centers, radii, (size_t)n_radii * sizeof(sif_real));
  sif__edges_from_centers(radii, n_radii, out->r_edges);

  const bool per_radius = (opt & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR;
  const double four_thirds_pi = 4.0 / 3.0 * SIF_PI;

  for (uint32_t i = 0; i < n_radii; i++) {
    const double r_e = (double)radii[i];
    const double r_l = (double)lagrangian[i];

    /* dln(sigma^-1)/dln r_L = -dln(sigma)/dln r_L, positive since sigma
     * falls with scale. */
    const double jacobian = -(double)slope[i];

    const double volume_radius = conserve_volume ? r_e : r_l;
    const double volume =
      four_thirds_pi * volume_radius * volume_radius * volume_radius;

    double value = (double)f[i] * jacobian / volume;
    if (per_radius)
      value /= r_e;

    out->vsf[i] = (sif_real)(value > 0.0 ? value : 0.0);
  }

  free(lagrangian);
  sif_delta_moments_free(moments);
  sif_free_aligned(slope);
  sif_free_aligned(f);

  return out;
}

sif_size_function_t* sif_size_function_svdw(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_v, sif_real delta_c, sif_option opt) {

  return size_function(
    k, pk, n_points, radii, n_radii, delta_v, delta_c, opt, false);
}

sif_size_function_t* sif_size_function_vdn(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_v, sif_real delta_c, sif_option opt) {

  return size_function(
    k, pk, n_points, radii, n_radii, delta_v, delta_c, opt, true);
}
