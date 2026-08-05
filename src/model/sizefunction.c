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

#include "sif/model/sizefunction.h"

#include "sif/model/deltamoments.h"
#include "structures/results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define __TAG "vsf_model"

/* Below this the mode series is replaced by its analytic small-x limit; above
 * it the series converges. Jennings, Li & Hu (2013) eq. (8). */
#define __SVDW_X_SWITCH 0.276

/* The Gaussian factor kills the series long before this; the cap only stops a
 * pathological D from spinning. */
#define __SVDW_MAX_TERMS 64

real_t sif_expansion_factor(real_t delta_v) {
  const double d = (double)delta_v;

  if (!(d < 0.0)) {
    SIF_LOG_ERROR(__TAG, "delta_v is %g, must be strictly negative", d);
    return (real_t)0.0;
  }

  const double c = SIF_SPHERICAL_EXPANSION_C;
  return (real_t)pow(1.0 - d / c, c / 3.0);
}

/* --- Multiplicity function --- */

static double __f_ln_sigma(double sigma, double abs_dv, double dcal) {
  /* D = |delta_v| / (delta_c + |delta_v|), x = (D / |delta_v|) sigma. */
  const double x = dcal / abs_dv * sigma;

  if (x <= __SVDW_X_SWITCH) {
    /* The series does not converge as x -> 0: sum_j j sin(j pi D) diverges,
     * so the limit has to come from the analytic form. */
    return sqrt(2.0 / M_PI) * (abs_dv / sigma) *
           exp(-0.5 * abs_dv * abs_dv / (sigma * sigma));
  }

  double sum = 0.0;
  for (int j = 1; j <= __SVDW_MAX_TERMS; j++) {
    const double jpx = j * M_PI * x;
    const double term =
      exp(-0.5 * jpx * jpx) * j * M_PI * x * x * sin(j * M_PI * dcal);

    sum += term;

    /* The envelope is monotonic past the first term, so once it is negligible
     * everything after it is too, regardless of the sine. */
    if (j > 1 && fabs(exp(-0.5 * jpx * jpx) * j * M_PI * x * x) < 1e-16)
      break;
  }

  return 2.0 * sum;
}

real_t* sif_multiplicity_function_svdw(
  const real_t* sigma, uint32_t n, real_t delta_v, real_t delta_c) {

  if (!sigma || n == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to multiplicity_function_svdw");
    return NULL;
  }

  if (!(delta_v < 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "delta_v is %g, must be strictly negative", (double)delta_v);
    return NULL;
  }

  if (!(delta_c > 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "delta_c is %g, must be strictly positive", (double)delta_c);
    return NULL;
  }

  const double abs_dv = fabs((double)delta_v);
  const double dcal = abs_dv / ((double)delta_c + abs_dv);

  if (dcal >= 0.75) {
    SIF_LOG_WARNING(__TAG,
      "D = %g exceeds the 3/4 the reference validates; the series is summed to "
      "convergence but the underlying approximation is outside its tested "
      "range",
      dcal);
  }

  real_t* out = sif_calloc_aligned((size_t)n, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the multiplicity array");
    return NULL;
  }

  uint32_t n_bad = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_bad)
  for (uint32_t i = 0; i < n; i++) {
    if (!(sigma[i] > 0.0f)) {
      n_bad++;
      continue;
    }
    const double f = __f_ln_sigma((double)sigma[i], abs_dv, dcal);
    out[i] = (real_t)(f > 0.0 ? f : 0.0);
  }

  if (n_bad > 0) {
    SIF_LOG_WARNING(__TAG,
      "%u of %u sigma values were not positive and were reported as zero",
      n_bad, n);
  }

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
static sif_size_function_t* __size_function(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, real_t delta_v,
  real_t delta_c, real_t expansion_factor, sif_option_t opt,
  bool conserve_volume) {

  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(__TAG, "no radii to evaluate");
    return NULL;
  }

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG, "radius %u is %g, must be strictly positive", i,
        (double)radii[i]);
      return NULL;
    }
  }

  double factor = (double)expansion_factor;
  if (!(factor > 0.0)) {
    const real_t derived = sif_expansion_factor(delta_v);
    if (!(derived > 0.0f))
      return NULL;
    factor = (double)derived;
    SIF_LOG_INFO(__TAG, "expansion factor derived from delta_v = %g: %g",
      (double)delta_v, factor);
  }

  /* The barriers are calibrated against top-hat smoothing, so the window is
   * not the caller's to choose here. */
  const sif_option_t moment_opt = SIF_DELTA_FILTER_TOP_HAT;

  real_t* lagrangian = malloc((size_t)n_radii * sizeof(real_t));
  if (!lagrangian) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the Lagrangian radii");
    return NULL;
  }
  for (uint32_t i = 0; i < n_radii; i++)
    lagrangian[i] = (real_t)((double)radii[i] / factor);

  sif_delta_moments_t* moments = sif_delta_moments_pk(
    k, pk, n_points, lagrangian, n_radii, 0, moment_opt);
  real_t* slope =
    sif_sigma_slope_pk(k, pk, n_points, lagrangian, n_radii, moment_opt);

  if (!moments || !slope) {
    free(lagrangian);
    sif_delta_moments_free(moments);
    sif_free_aligned(slope);
    return NULL;
  }

  const real_t* sigma = sif_delta_moments_sigma(moments, 0);
  real_t* f = sif_multiplicity_function_svdw(sigma, n_radii, delta_v, delta_c);

  sif_size_function_t* out = __sif_size_function_alloc(n_radii);

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
  memcpy(out->r_centers, radii, (size_t)n_radii * sizeof(real_t));
  __sif_edges_from_centers(radii, n_radii, out->r_edges);

  const bool per_radius = (opt & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR;
  const double four_thirds_pi = 4.0 / 3.0 * M_PI;

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

    out->vsf[i] = (real_t)(value > 0.0 ? value : 0.0);
  }

  free(lagrangian);
  sif_delta_moments_free(moments);
  sif_free_aligned(slope);
  sif_free_aligned(f);

  return out;
}

sif_size_function_t* sif_size_function_svdw(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, real_t delta_v,
  real_t delta_c, real_t expansion_factor, sif_option_t opt) {

  return __size_function(k, pk, n_points, radii, n_radii, delta_v, delta_c,
    expansion_factor, opt, false);
}

sif_size_function_t* sif_size_function_vdn(const real_t* k, const real_t* pk,
  uint32_t n_points, const real_t* radii, uint32_t n_radii, real_t delta_v,
  real_t delta_c, real_t expansion_factor, sif_option_t opt) {

  return __size_function(k, pk, n_points, radii, n_radii, delta_v, delta_c,
    expansion_factor, opt, true);
}
