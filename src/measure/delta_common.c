/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "delta_common.h"

#include "core/system_internal.h"
#include "sif/utils/logger.h"

#include <stdbool.h>

int sif__delta_validate_radii(
  const sif_grid_t* grid, const sif_real* radii, uint32_t n_radii) {

  const sif_real min_radius =
    (sif_real)(SIF__DELTA_MIN_CELLS_PER_RADIUS * (double)grid->cell_length);
  const sif_real max_radius = grid->box_length * 0.5f;

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(SIF__DELTA_TAG,
        "radius %u is %g, must be strictly positive", i, (double)radii[i]);
      return SIF_ERR_INVALID;
    }
    if (radii[i] < min_radius) {
      SIF_LOG_ERROR(SIF__DELTA_TAG,
        "radius %u is %g, below the %g resolution limit of a %u^3 grid; "
        "increase n_cells or drop the radius",
        i, (double)radii[i], (double)min_radius, grid->n_cells);
      return SIF_ERR_INVALID;
    }
    if (radii[i] > max_radius) {
      SIF_LOG_ERROR(SIF__DELTA_TAG,
        "radius %u is %g, larger than half the box (%g); the smoothing window "
        "would wrap onto itself",
        i, (double)radii[i], (double)max_radius);
      return SIF_ERR_INVALID;
    }
  }

  return SIF_OK;
}

sif_filter_type_t sif__delta_filter(sif_option opt) {
  return (opt & SIF__DELTA_FILTER_MASK) == SIF_DELTA_FILTER_GAUSSIAN
           ? SIF__FILTER_GAUSSIAN
           : SIF__FILTER_TOP_HAT;
}

int sif__delta_validate_options(sif_option opt) {
  const uint32_t shuffle = opt & SIF__DELTA_SHUFFLE_MASK;

  if (shuffle != SIF_DELTA_SHUFFLE_NONE &&
      shuffle != SIF_DELTA_SHUFFLE_PHASES &&
      shuffle != SIF_DELTA_SHUFFLE_GAUSSIAN) {
    SIF_LOG_ERROR(
      SIF__DELTA_TAG, "unrecognized shuffle mode in the options bitmask");
    return SIF_ERR_INVALID;
  }

  if ((opt & SIF__PBC_MASK) == SIF_PBC_OPEN) {
    SIF_LOG_WARNING(SIF__DELTA_TAG,
      "the Fourier estimator is periodic by construction; SIF_PBC_OPEN has no "
      "effect");
  }

  return SIF_OK;
}

sif_fft_workspace_t* sif__delta_prepare_spectrum(
  const sif_grid_t* grid, uint64_t seed, sif_option opt) {

  /* A grid still holding densities, handed to something that expects a
   * density contrast, produces numbers rather than an error. SIF_GRID_EMPTY is
   * not flagged: that is a grid the caller filled directly, and only the caller
   * knows what is in it. */
  if (grid->content == SIF_GRID_DENSITY) {
    SIF_LOG_WARNING(SIF__DELTA_TAG,
      "this grid holds a density, not a density contrast; call "
      "sif_grid_to_density_contrast first");
  }

  sif_system_state_t* state = sif__system_state();

  sif_fft_workspace_t* ws =
    sif__fft_workspace_alloc(state->fft_mgr, grid->n_cells);
  if (!ws) {
    SIF_LOG_ERROR(SIF__DELTA_TAG, "failed to allocate the FFT workspace");
    return NULL;
  }

  /* Checked, not assumed: planning can fail, and it leaves delta_k holding
   * whatever the allocator returned. Every moment and every PDF downstream is
   * computed from that buffer, so an unchecked failure here does not surface
   * as an error anywhere -- it surfaces as a result. */
  if (sif__fft_grid_forward(ws, grid) != SIF_OK) {
    SIF_LOG_ERROR(SIF__DELTA_TAG, "the forward transform failed");
    sif__fft_workspace_free(ws);
    return NULL;
  }

  if (!(opt & SIF_DELTA_KEEP_CIC_WINDOW)) {
    if (sif__fft_deconvolve_cic(ws) != SIF_OK) {
      SIF_LOG_ERROR(SIF__DELTA_TAG, "failed to deconvolve the CIC window");
      sif__fft_workspace_free(ws);
      return NULL;
    }
  }

  const uint32_t shuffle = opt & SIF__DELTA_SHUFFLE_MASK;
  if (shuffle != SIF_DELTA_SHUFFLE_NONE) {
    const bool resample = (shuffle == SIF_DELTA_SHUFFLE_GAUSSIAN);
    if (sif__fft_randomize_phases(ws, seed, resample) != SIF_OK) {
      SIF_LOG_ERROR(SIF__DELTA_TAG, "failed to randomize the phases");
      sif__fft_workspace_free(ws);
      return NULL;
    }
  }

  return ws;
}
