#ifndef __SIF_STRUCT_DELTADISTRIBUTION_H__
#define __SIF_STRUCT_DELTADISTRIBUTION_H__

#include <stdint.h>

#include "sif/core/macros.h"

/*
 * @brief PDF of the smoothed density contrast, one row per smoothing radius.
 *
 * Produced by sif_delta_distribution_grid, released with
 * sif_delta_distribution_free.
 */
typedef struct {
  uint32_t n_radii;
  uint32_t n_bins;

  /* Grid cells behind each row. Samples outside delta_bounds are counted here
   * but not histogrammed, so a row integrates to the fraction in range. */
  uint64_t n_samples;

  real_t* radii;         /* n_radii */
  real_t* delta_edges;   /* n_bins + 1 */
  real_t* distributions; /* n_radii * n_bins, row-major */
} sif_delta_distribution_t;

void sif_delta_distribution_free(sif_delta_distribution_t* dist);

#endif /* __SIF_STRUCT_DELTADISTRIBUTION_H__ */
