#include "sif/structures/deltamoments.h"

#include "results_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

#define __TAG "delta"

sif_delta_moments_t* __sif_delta_moments_alloc(uint32_t n_radii, uint8_t order) {

  sif_delta_moments_t* m = calloc(1, sizeof(sif_delta_moments_t));
  if (!m) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the delta moments struct");
    return NULL;
  }

  m->n_radii = n_radii;
  m->order = order;
  m->n_moments = (uint8_t)(order + 1);

  const size_t total = (size_t)m->n_moments * n_radii;

  m->radii = sif_malloc_aligned((size_t)n_radii * sizeof(real_t));
  m->sigma = sif_calloc_aligned(total, sizeof(real_t));
  m->high_k_fraction = sif_calloc_aligned(total, sizeof(real_t));
  m->offsets = sif_malloc_aligned(((size_t)m->n_moments + 1) * sizeof(uint32_t));

  if (!m->radii || !m->sigma || !m->high_k_fraction || !m->offsets) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the delta moments arrays");
    sif_delta_moments_free(m);
    return NULL;
  }

  for (uint8_t j = 0; j <= m->n_moments; j++) {
    m->offsets[j] = (uint32_t)j * n_radii;
  }

  return m;
}

void sif_delta_moments_free(sif_delta_moments_t* moments) {
  if (!moments)
    return;
  sif_free_aligned(moments->radii);
  sif_free_aligned(moments->sigma);
  sif_free_aligned(moments->high_k_fraction);
  sif_free_aligned(moments->offsets);
  free(moments);
}

const real_t* sif_delta_moments_sigma(
  const sif_delta_moments_t* moments, uint8_t order) {

  if (!moments || !moments->sigma || order > moments->order)
    return NULL;
  return moments->sigma + moments->offsets[order];
}

/*
 * gamma and R_star both need sigma_0..sigma_2 together, so they share the same
 * precondition and the same "not enough orders" failure.
 */
static const real_t* __require_order_2(
  const sif_delta_moments_t* moments, const char* what) {

  if (!moments || !moments->sigma) {
    SIF_LOG_ERROR(__TAG, "cannot compute %s from an empty moment set", what);
    return NULL;
  }

  if (moments->order < 2) {
    SIF_LOG_ERROR(__TAG,
      "%s needs sigma_0 through sigma_2, but the moment set only reaches "
      "order %u",
      what, moments->order);
    return NULL;
  }

  return moments->sigma;
}

real_t* sif_gamma_moments(const sif_delta_moments_t* moments) {

  if (!__require_order_2(moments, "gamma"))
    return NULL;

  const real_t* s0 = sif_delta_moments_sigma(moments, 0);
  const real_t* s1 = sif_delta_moments_sigma(moments, 1);
  const real_t* s2 = sif_delta_moments_sigma(moments, 2);

  real_t* out = sif_calloc_aligned((size_t)moments->n_radii, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the gamma array");
    return NULL;
  }

  for (uint32_t r = 0; r < moments->n_radii; r++) {
    if (s0[r] > 0.0f && s2[r] > 0.0f) {
      out[r] = (real_t)((double)s1[r] * s1[r] / ((double)s0[r] * s2[r]));
    } else {
      SIF_LOG_WARNING(__TAG,
        "gamma at radius %g is undefined: sigma_0 = %g, sigma_2 = %g",
        (double)moments->radii[r], (double)s0[r], (double)s2[r]);
    }
  }

  return out;
}

real_t* sif_r_star_moments(const sif_delta_moments_t* moments) {

  if (!__require_order_2(moments, "R_star"))
    return NULL;

  const real_t* s1 = sif_delta_moments_sigma(moments, 1);
  const real_t* s2 = sif_delta_moments_sigma(moments, 2);

  real_t* out = sif_calloc_aligned((size_t)moments->n_radii, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the R_star array");
    return NULL;
  }

  const double root3 = sqrt(3.0);

  for (uint32_t r = 0; r < moments->n_radii; r++) {
    if (s2[r] > 0.0f) {
      out[r] = (real_t)(root3 * (double)s1[r] / s2[r]);
    } else {
      SIF_LOG_WARNING(__TAG, "R_star at radius %g is undefined: sigma_2 = %g",
        (double)moments->radii[r], (double)s2[r]);
    }
  }

  return out;
}
