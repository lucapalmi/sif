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
 *
 * The EVM model replaces both ends of that chain: a moving barrier calibrated
 * on the z = 99 catalogues, whose first crossing comes from the emulator, and
 * a volume-conserving mapping times a fitted correction.
 */

#include "sif/model/size_function.h"

#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/model/variance_kernel.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/results_internal.h"
#include "utils/logger_internal.h"

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

/* --- The container --- */

/* Strictly increasing, not merely positive: r_min, r_max and the bin edges are
 * all read off the ends of this array, and sif__edges_from_centres builds an
 * edge grid that assumes it. sif_size_function_bbks() enforces the same thing
 * before it differentiates. */
static int check_radii(const sif_real* radii, uint32_t n_radii) {
  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "no radii to evaluate");
    sif__error_status_set(SIF_ERR_INVALID);
    return SIF_ERR_INVALID;
  }

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(
        TAG, "radius %u is %g, must be strictly positive", i, (double)radii[i]);
      sif__error_status_set(SIF_ERR_INVALID);
      return SIF_ERR_INVALID;
    }
    if (i > 0 && !(radii[i] > radii[i - 1])) {
      SIF_LOG_ERROR(TAG,
        "radii must be strictly increasing; index %u is %g after %g", i,
        (double)radii[i], (double)radii[i - 1]);
      sif__error_status_set(SIF_ERR_INVALID);
      return SIF_ERR_INVALID;
    }
  }

  return SIF_OK;
}

/* A size function on `radii`, with only the abundance left to fill. */
static sif_size_function_t* alloc_on_radii(
  const sif_real* radii, uint32_t n_radii, sif_option opt) {

  sif_size_function_t* out = sif__size_function_alloc(n_radii);
  if (!out)
    return NULL;

  out->options = opt;
  out->r_min = radii[0];
  out->r_max = radii[n_radii - 1];
  memcpy(out->r_centres, radii, (size_t)n_radii * sizeof(sif_real));
  sif__edges_from_centres(radii, n_radii, out->r_edges);

  return out;
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

  if (check_radii(radii, n_radii) != SIF_OK)
    return NULL;

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

  sif_size_function_t* out = alloc_on_radii(radii, n_radii, opt);

  if (!f || !out) {
    free(lagrangian);
    sif_delta_moments_free(moments);
    sif_free_aligned(slope);
    sif_free_aligned(f);
    sif_size_function_free(out);
    return NULL;
  }

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

/* --- The EVM size function --- */

/*
 * The barrier, B(sigma) = alpha [1 + (beta / sigma)^gamma] with
 *
 *     alpha = A |delta_L|^B q^(1 - B),  beta = beta0 q,  gamma = gamma0,
 *
 * and q = sigma12 / sigma12(z = 99): calibrated at z = 99, where q = 1, and
 * carried to any other epoch or cosmology by linear growth rescaling the
 * field. The values are the maximum a posteriori of the calibration chain.
 */
#define EVM_BARRIER_A      1.295609294830281
#define EVM_BARRIER_B      1.3782884464438276
#define EVM_BARRIER_BETA0  0.014867575268555028
#define EVM_BARRIER_GAMMA0 0.42590405524560554

/* sigma at R = 12 Mpc of the calibration cosmology, at z = 99 and z = 0. They
 * are part of the model: the same two numbers for every cosmology. */
#define EVM_SIGMA12_Z99 0.010642019592327554
#define EVM_SIGMA12_Z0  0.8247254042480003

/* The correction's constants, the maximum a posteriori of their chain, and
 * the ln(nu) its variable u is measured from. */
#define EVM_MAPPING_A   0.19421695931546248
#define EVM_MAPPING_C1  0.2242333261503064
#define EVM_MAPPING_C2  0.7311577281984404
#define EVM_MAPPING_C3  0.8140954933390583
#define EVM_MAPPING_C4  1.6801368020718135
#define EVM_LN_NU_PIVOT 0.3483074

/* What the model was fitted over. delta_L at z = 99 is the threshold the
 * barrier was calibrated on; its lower end keeps the 1% margin the fit's own
 * cut allowed. */
#define EVM_SIGMA12_MIN   0.39
#define EVM_SIGMA12_MAX   1.0
#define EVM_DELTA_Z99_MIN 0.00297
#define EVM_DELTA_Z99_MAX 0.045

/* The lattice sif_size_function_evm builds, which the calibration used too. */
#define EVM_LATTICE_R_MIN 0.3
#define EVM_LATTICE_R_MAX 1000.0
#define EVM_LATTICE_DLN_R 0.02

/* sigma12 is sigma at 12 Mpc, whatever h makes that in Mpc/h. */
#define EVM_SIGMA12_RADIUS_MPC 12.0

/*
 * sigma at a radius inside the walk's lattice: the straight line in ln sigma
 * against ln R. The correction was fitted to sigma interpolated like this, and
 * the lattice is fine enough that a cubic would move it by about 1e-5.
 */
static double walk_sigma_at(const sif_walk_t* walk, double radius) {
  const uint32_t n = walk->n_radii;
  const double ln_r0 = log((double)walk->radii[0]);
  const double step =
    (log((double)walk->radii[n - 1]) - ln_r0) / (double)(n - 1);
  const double t = (log(radius) - ln_r0) / step;

  int64_t c = (int64_t)floor(t);
  if (c < 0)
    c = 0;
  if (c > (int64_t)n - 2)
    c = (int64_t)n - 2;

  const double u = t - (double)c;
  return exp((1.0 - u) * log((double)walk->sigma[c]) +
             u * log((double)walk->sigma[c + 1]));
}

/*
 * ln T, the correction to the volume-conserving mapping. A symbolic-regression
 * formula rather than a derived law, so its terms have no individual meaning:
 *
 *     u = ln(B / sigma) - pivot, at the Eulerian radius
 *     w = ln(1 + delta_NL)
 *     l = ln(sigma12 / sigma12(z = 0))
 *     s = ln sigma, at the Eulerian radius
 */
static double evm_ln_correction(double u, double w, double l, double s) {
  const double q = w - l * l * w - l * u;
  const double e = u + exp(w - l * (l - q * q));
  const double u2 = u * u;
  const double shape = u2 - e * e * (EVM_MAPPING_C1 * u2 - EVM_MAPPING_C2) *
                              (u2 + u + w - exp(-l * w));
  const double offset = u - exp(w) + EVM_MAPPING_C4;

  return EVM_MAPPING_A *
         (l * (w - l * u * exp(l)) + (l * w + s + EVM_MAPPING_C3) * shape +
           offset * offset);
}

sif_size_function_t* sif_size_function_evm_kernel(
  const sif_variance_kernel_t* kernel, const sif_real* pk,
  const sif_real* radii, uint32_t n_radii, sif_real delta_v, sif_real h,
  sif_option opt) {

  if (!kernel || !pk) {
    SIF_LOG_ERROR(TAG, "kernel and pk are required");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (check_radii(radii, n_radii) != SIF_OK)
    return NULL;
  if (!(h > 0.0f)) {
    SIF_LOG_ERROR(TAG, "h: %g (expected strictly positive)", (double)h);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if ((kernel->window & SIF__DELTA_FILTER_MASK) != SIF_DELTA_FILTER_TOP_HAT) {
    SIF_LOG_ERROR(TAG, "kernel: Gaussian window (expected top-hat)");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  const double radius12 = EVM_SIGMA12_RADIUS_MPC * (double)h;
  const double lattice_lo = (double)kernel->radii[0];
  const double lattice_hi = (double)kernel->radii[kernel->n_radii - 1];

  if (!(radius12 >= lattice_lo && radius12 <= lattice_hi)) {
    SIF_LOG_ERROR(TAG, "12 Mpc = %g Mpc/h: outside the lattice, %g to %g",
      radius12, lattice_lo, lattice_hi);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  /* The emulator checks the Lagrangian radii, which are the smaller. */
  if (!((double)radii[n_radii - 1] <= lattice_hi)) {
    SIF_LOG_ERROR(TAG, "radius %g: beyond the lattice, which ends at %g",
      (double)radii[n_radii - 1], lattice_hi);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  const sif_real delta_nl = sif_spherical_map_nonlinear(delta_v, opt);
  if (!(delta_nl < 0.0f)) /* outside the expanding branch; already logged */
    return NULL;

  sif_walk_t* walk = sif_walk_pk(kernel, pk);
  if (!walk)
    return NULL;

  const double sigma12 = walk_sigma_at(walk, radius12);
  const double q = sigma12 / EVM_SIGMA12_Z99;
  const double depth = -(double)delta_v;

  if (sigma12 < EVM_SIGMA12_MIN || sigma12 > EVM_SIGMA12_MAX) {
    SIF_LOG_WARNING(TAG, "sigma12: %g (calibrated over %g to %g)", sigma12,
      EVM_SIGMA12_MIN, EVM_SIGMA12_MAX);
  }
  if (depth / q < EVM_DELTA_Z99_MIN || depth / q > EVM_DELTA_Z99_MAX) {
    SIF_LOG_WARNING(TAG,
      "delta_v at z = 99: %g (calibrated over %g to %g in magnitude)",
      depth / q, EVM_DELTA_Z99_MIN, EVM_DELTA_Z99_MAX);
  }

  const double alpha =
    EVM_BARRIER_A * pow(depth, EVM_BARRIER_B) * pow(q, 1.0 - EVM_BARRIER_B);
  const double beta = EVM_BARRIER_BETA0 * q;
  const double gamma = EVM_BARRIER_GAMMA0;

  /* The same mass in a volume 1 / (1 + delta_NL) times larger. */
  const double density = 1.0 + (double)delta_nl;
  const double shrink = cbrt(density);

  sif_real* barrier = sif_ep_barrier_smt(walk->sigma, walk->n_radii,
    (sif_real)alpha, (sif_real)beta, (sif_real)gamma);
  sif_real* lagrangian = malloc((size_t)n_radii * sizeof(sif_real));
  sif_real* f = NULL;

  if (barrier && lagrangian) {
    for (uint32_t i = 0; i < n_radii; i++)
      lagrangian[i] = (sif_real)((double)radii[i] * shrink);

    /* Not `opt`: its spherical bit is the emulator's output bit. */
    f = sif_ep_multiplicity_function_emu(
      walk, barrier, NULL, lagrangian, n_radii, NULL, SIF_EP_EMU_AT_RADII);
  } else if (barrier) {
    SIF_LOG_ERROR(TAG, "failed to allocate the Lagrangian radii");
  }

  sif_size_function_t* out = f ? alloc_on_radii(radii, n_radii, opt) : NULL;

  if (out) {
    const bool per_radius = (opt & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR;
    const double w = log(density);
    const double l = log(sigma12 / EVM_SIGMA12_Z0);

    for (uint32_t i = 0; i < n_radii; i++) {
      /* Past the last radius the emulator resolves f is exactly zero, and ln T
       * has long since left the range it was fitted over: at nu ~ 40 it is in
       * the hundreds, which only the Gaussian fall of f holds down. */
      if (!(f[i] > 0.0f)) {
        out->vsf[i] = 0.0f;
        continue;
      }

      const double r_e = (double)radii[i];
      const double r_l = r_e * shrink;
      const double sigma = walk_sigma_at(walk, r_e);
      const double nu = alpha * (1.0 + pow(beta / sigma, gamma)) / sigma;
      const double ln_t =
        evm_ln_correction(log(nu) - EVM_LN_NU_PIVOT, w, l, log(sigma));

      /* dn/dln R_L = R_L f / V(R_L), then (1 + delta_NL) for the volume. */
      double value = (double)f[i] * r_l /
                     (4.0 / 3.0 * SIF_PI * r_l * r_l * r_l) * density *
                     exp(ln_t);
      if (per_radius)
        value /= r_e;

      out->vsf[i] = (sif_real)(value > 0.0 ? value : 0.0);
    }
  }

  sif_free_aligned(barrier);
  sif_free_aligned(f);
  free(lagrangian);
  sif_walk_free(walk);

  return out;
}

sif_size_function_t* sif_size_function_evm(const sif_real* k,
  const sif_real* pk, uint32_t n_points, const sif_real* radii,
  uint32_t n_radii, sif_real delta_v, sif_real h, sif_option opt) {

  if (check_radii(radii, n_radii) != SIF_OK)
    return NULL;

  const sif_real delta_nl = sif_spherical_map_nonlinear(delta_v, opt);
  if (!(delta_nl < 0.0f)) /* outside the expanding branch; already logged */
    return NULL;

  /* The calibration's lattice, stretched for radii or an h beyond it. */
  const double r_small = (double)radii[0] * cbrt(1.0 + (double)delta_nl);
  const double r_large = (double)radii[n_radii - 1];
  const double radius12 = EVM_SIGMA12_RADIUS_MPC * (double)h;
  const double r_min = fmin(EVM_LATTICE_R_MIN, 0.9 * fmin(r_small, radius12));
  const double r_max = fmax(EVM_LATTICE_R_MAX, 1.1 * fmax(r_large, radius12));

  sif_variance_kernel_t* kernel =
    sif_variance_kernel_alloc(k, n_points, (sif_real)r_min, (sif_real)r_max,
      (sif_real)EVM_LATTICE_DLN_R, SIF_DELTA_FILTER_TOP_HAT);
  if (!kernel)
    return NULL;

  sif_size_function_t* out =
    sif_size_function_evm_kernel(kernel, pk, radii, n_radii, delta_v, h, opt);

  sif_variance_kernel_free(kernel);
  return out;
}

#undef TAG
