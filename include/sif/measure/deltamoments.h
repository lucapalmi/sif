#ifndef __SIF_MEASURE_DELTAMOMENTS_H__
#define __SIF_MEASURE_DELTAMOMENTS_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/deltamoments.h"
#include "sif/structures/grid.h"

/*
 * @brief Measures the spectral moments from a gridded density field.
 *
 * @param grid Overdensity field, from sif_grid_compute_overdensity
 * @param radii Smoothing radii, each at least two cell lengths and at most
 * half the box
 * @param n_radii Number of radii
 * @param order Highest moment order, at most SIF_MAX_MOMENT_ORDER
 * @param n_tracers Tracer count behind the grid, used to subtract the Poisson
 * term. Pass 0 to leave it in. Assumes unweighted tracers.
 * @param seed Seed for the phase shuffle, unused when not shuffling
 * @param opt SIF_DELTA_SHUFFLE_*, SIF_DELTA_FILTER_*, SIF_DELTA_KEEP_CIC_WINDOW
 *
 * @return Newly allocated moment set, or NULL on invalid input or failure.
 */
NODISCARD sif_delta_moments_t* sif_delta_moments_grid(const sif_grid_t* grid,
  const real_t* radii, uint32_t n_radii, uint8_t order, uint64_t n_tracers,
  uint64_t seed, sif_option_t opt);

#endif /* __SIF_MEASURE_DELTAMOMENTS_H__ */
