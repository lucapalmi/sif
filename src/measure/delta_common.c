#include "delta_common.h"

#include "core/get_system.h"
#include "sif/utils/logger.h"

#include <stdbool.h>

int sif_delta_validate_radii(
  const sif_grid_t* grid, const real_t* radii, uint32_t n_radii) {

  const real_t min_radius = (real_t)(__SIF_DELTA_MIN_CELLS_PER_RADIUS *
                                     (double)grid->cell_length);
  const real_t max_radius = grid->box_length * 0.5f;

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(__SIF_DELTA_TAG,
        "radius %u is %g, must be strictly positive", i, (double)radii[i]);
      return SIF_ERR_INVALID;
    }
    if (radii[i] < min_radius) {
      SIF_LOG_ERROR(__SIF_DELTA_TAG,
        "radius %u is %g, below the %g resolution limit of a %u^3 grid; "
        "increase n_cells or drop the radius",
        i, (double)radii[i], (double)min_radius, grid->n_cells);
      return SIF_ERR_INVALID;
    }
    if (radii[i] > max_radius) {
      SIF_LOG_ERROR(__SIF_DELTA_TAG,
        "radius %u is %g, larger than half the box (%g); the smoothing window "
        "would wrap onto itself",
        i, (double)radii[i], (double)max_radius);
      return SIF_ERR_INVALID;
    }
  }

  return SIF_OK;
}

sif_filter_type_t sif_delta_filter(sif_option_t opt) {
  return (opt & __SIF_DELTA_FILTER_MASK) == SIF_DELTA_FILTER_GAUSSIAN
           ? FILTER_GAUSSIAN
           : FILTER_TOP_HAT;
}

int sif_delta_validate_options(sif_option_t opt) {
  const uint32_t shuffle = opt & __SIF_DELTA_SHUFFLE_MASK;

  if (shuffle != SIF_DELTA_SHUFFLE_NONE &&
      shuffle != SIF_DELTA_SHUFFLE_PHASES &&
      shuffle != SIF_DELTA_SHUFFLE_GAUSSIAN) {
    SIF_LOG_ERROR(
      __SIF_DELTA_TAG, "unrecognized shuffle mode in the options bitmask");
    return SIF_ERR_INVALID;
  }

  if ((opt & __SIF_PBC_MASK) == SIF_PBC_OPEN) {
    SIF_LOG_WARNING(__SIF_DELTA_TAG,
      "the Fourier estimator is periodic by construction; SIF_PBC_OPEN has no "
      "effect");
  }

  return SIF_OK;
}

sif_fft_workspace_t* sif_delta_prepare_spectrum(
  const sif_grid_t* grid, uint64_t seed, sif_option_t opt) {

  sif_system_state_t* state = sif_get_system_state();

  sif_fft_workspace_t* ws = sif_fft_workspace_alloc(state->fft_mgr, grid->n_cells);
  if (!ws) {
    SIF_LOG_ERROR(__SIF_DELTA_TAG, "failed to allocate the FFT workspace");
    return NULL;
  }

  sif_fft_grid_forward(ws, grid);

  if (!(opt & SIF_DELTA_KEEP_CIC_WINDOW)) {
    if (sif_fft_deconvolve_cic(ws) != SIF_OK) {
      SIF_LOG_ERROR(__SIF_DELTA_TAG, "failed to deconvolve the CIC window");
      sif_fft_workspace_free(ws);
      return NULL;
    }
  }

  const uint32_t shuffle = opt & __SIF_DELTA_SHUFFLE_MASK;
  if (shuffle != SIF_DELTA_SHUFFLE_NONE) {
    const bool resample = (shuffle == SIF_DELTA_SHUFFLE_GAUSSIAN);
    if (sif_fft_randomize_phases(ws, seed, resample) != SIF_OK) {
      SIF_LOG_ERROR(__SIF_DELTA_TAG, "failed to randomize the phases");
      sif_fft_workspace_free(ws);
      return NULL;
    }
  }

  return ws;
}
