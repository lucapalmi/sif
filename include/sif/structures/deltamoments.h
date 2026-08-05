#ifndef __SIF_STRUCT_DELTAMOMENTS_H__
#define __SIF_STRUCT_DELTAMOMENTS_H__

#include <stdint.h>

#include "sif/core/macros.h"

/* Highest spectral moment order the estimators accept. */
#define SIF_MAX_MOMENT_ORDER 4

/*
 * @brief Spectral moments sigma_0 .. sigma_order, at several smoothing radii.
 *
 * sigma_j carries units of length^-j. Produced either by measuring a gridded
 * field (sif_delta_moments_grid) or by integrating a model power spectrum
 * (sif_delta_moments_pk); released with sif_delta_moments_free.
 */
typedef struct {
  uint32_t n_radii;
  uint8_t order;     /* highest order computed */
  uint8_t n_moments; /* order + 1 */

  real_t* radii; /* n_radii */

  /* n_moments * n_radii, order-major: all radii for j = 0, then j = 1, ... */
  real_t* sigma;

  /* Same layout as sigma. Fraction of each sum or integral coming from the top
   * half of the available k range: above half Nyquist for the grid estimator,
   * above half of k_max for the P(k) one. */
  real_t* high_k_fraction;

  /* n_moments + 1 entries. offsets[j] is where order j's block of n_radii
   * values starts in sigma and high_k_fraction. */
  uint32_t* offsets;
} sif_delta_moments_t;

void sif_delta_moments_free(sif_delta_moments_t* moments);

/*
 * @brief Borrowed pointer to the n_radii values of sigma_order.
 *
 * @return Pointer into `moments`, not to be freed, or NULL if that order was
 * not computed.
 */
const real_t* sif_delta_moments_sigma(
  const sif_delta_moments_t* moments, uint8_t order);

/*
 * @brief Spectral parameter, sigma_1^2 / (sigma_0 sigma_2).
 *
 * @param moments Moment set with order >= 2
 *
 * @return Newly allocated array of n_radii values, released with
 * sif_free_aligned, or NULL if the moments do not reach order 2.
 */
NODISCARD real_t* sif_gamma_moments(const sif_delta_moments_t* moments);

/*
 * @brief Coherence scale, sqrt(3) sigma_1 / sigma_2.
 *
 * @param moments Moment set with order >= 2
 *
 * @return Newly allocated array of n_radii lengths, released with
 * sif_free_aligned, or NULL if the moments do not reach order 2.
 */
NODISCARD real_t* sif_r_star_moments(const sif_delta_moments_t* moments);

#endif /* __SIF_STRUCT_DELTAMOMENTS_H__ */
