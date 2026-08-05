#ifndef __SIF_MEASURE_DELTADISTRIBUTION_H__
#define __SIF_MEASURE_DELTADISTRIBUTION_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/deltadistribution.h"
#include "sif/structures/grid.h"

/*
 * @brief Measures the PDF of the smoothed density contrast on a grid.
 *
 * @param grid Overdensity field, from sif_grid_compute_overdensity
 * @param radii Smoothing radii, each at least two cell lengths and at most
 * half the box
 * @param n_radii Number of radii
 * @param n_bins Number of histogram bins
 * @param delta_bounds {min_delta, max_delta}, strictly increasing
 * @param seed Seed for the phase shuffle, unused when not shuffling
 * @param opt SIF_DELTA_SHUFFLE_*, SIF_DELTA_FILTER_*, SIF_DELTA_KEEP_CIC_WINDOW
 *
 * @return Newly allocated distribution, or NULL on invalid input or failure.
 */
NODISCARD sif_delta_distribution_t* sif_delta_distribution_grid(
  const sif_grid_t* grid, const real_t* radii, uint32_t n_radii,
  uint32_t n_bins, const real_t delta_bounds[2], uint64_t seed,
  sif_option_t opt);

#endif /* __SIF_MEASURE_DELTADISTRIBUTION_H__ */
