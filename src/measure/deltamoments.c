#include "sif/measure/deltamoments.h"

#include "delta_common.h"
#include "structures/results_internal.h"
#include "math/fft.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define __TAG __SIF_DELTA_TAG

/* high_k_fraction above this is reported: past it the moment is dominated by
 * modes at the edge of the available k range and is not a statement about the
 * field any more. */
#define __HIGH_K_WARN_LEVEL 0.1

/*
 * Stores one radius' column of results. A shot-noise subtraction can push a
 * sigma^2 negative, which is a real outcome -- the moment was noise -- so it
 * is reported as a zero sigma and flagged rather than square-rooted into a
 * NaN.
 */
static void __store(sif_delta_moments_t* m, uint32_t k, const double* sigma_sq,
  const double* high_k) {

  for (uint8_t j = 0; j < m->n_moments; j++) {
    const size_t at = m->offsets[j] + k;

    m->sigma[at] = (real_t)(sigma_sq[j] > 0.0 ? sqrt(sigma_sq[j]) : 0.0);
    m->high_k_fraction[at] = (real_t)high_k[j];

    if (sigma_sq[j] <= 0.0) {
      SIF_LOG_WARNING(__TAG,
        "sigma_%u^2 at radius %g is %g after shot-noise subtraction; the "
        "moment is dominated by Poisson noise",
        j, (double)m->radii[k], sigma_sq[j]);
    }

    if (high_k[j] > __HIGH_K_WARN_LEVEL) {
      SIF_LOG_WARNING(__TAG,
        "at R = %g, %.1f%% of the sigma_%u sum comes from the top half of the "
        "available k range; that moment is resolution-limited, not "
        "field-limited",
        (double)m->radii[k], 100.0 * high_k[j], j);
    }
  }
}

static int __validate_order(uint8_t order) {
  if (order > SIF_MAX_MOMENT_ORDER) {
    SIF_LOG_ERROR(__TAG, "moment order %u exceeds the maximum of %d", order,
      SIF_MAX_MOMENT_ORDER);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/*
 * --- From a gridded field ---
 *
 * A note that matters for anything comparing a field against its surrogate:
 * SIF_DELTA_SHUFFLE_PHASES leaves every moment bit-for-bit unchanged, because
 * the moments depend only on |delta_k| and that shuffle preserves the moduli.
 * SIF_DELTA_SHUFFLE_GAUSSIAN does not -- it resamples the amplitudes, so its
 * moments are a different draw with the same expectation.
 */

sif_delta_moments_t* sif_delta_moments_grid(
  const sif_grid_t* grid, const real_t* radii, uint32_t n_radii, uint8_t order,
  uint64_t n_tracers, uint64_t seed, sif_option_t opt) {

  if (!grid || !grid->delta || !radii || n_radii == 0) {
    SIF_LOG_ERROR(
      __TAG, "invalid arguments to sif_delta_moments_grid");
    return NULL;
  }

  if (__validate_order(order) != SIF_OK)
    return NULL;

  if (__sif_delta_validate_radii(grid, radii, n_radii) != SIF_OK)
    return NULL;

  if (__sif_delta_validate_options(opt) != SIF_OK)
    return NULL;

  const filter_type_t filter = __sif_delta_filter(opt);

  if (filter == FILTER_TOP_HAT && order >= 2) {
    SIF_LOG_INFO(__TAG,
      "the top-hat W^2 decays only as k^-4, so sigma_2 and above are cut off "
      "by the grid rather than by the field; check high_k_fraction or switch "
      "to SIF_DELTA_FILTER_GAUSSIAN");
  }

  sif_delta_moments_t* m = __sif_delta_moments_alloc(n_radii, order);
  if (!m)
    return NULL;

  memcpy(m->radii, radii, (size_t)n_radii * sizeof(real_t));

  SIF_LOG_INFO(__TAG,
    "measuring sigma_0..sigma_%u over %u radii on a %u^3 grid (%s window)",
    order, n_radii, grid->n_cells,
    filter == FILTER_GAUSSIAN ? "Gaussian" : "top-hat");

  fft_workspace_t* ws = __sif_delta_prepare_spectrum(grid, seed, opt);
  if (!ws) {
    sif_delta_moments_free(m);
    return NULL;
  }

  double sigma_sq[SIF_MAX_MOMENT_ORDER + 1];
  double high_k[SIF_MAX_MOMENT_ORDER + 1];

  for (uint32_t k = 0; k < n_radii; k++) {
    /* Every order shares one pass over the spectrum. */
    if (fft_spectral_moments(ws, filter, radii[k], grid->box_length, order,
          n_tracers, sigma_sq, high_k) != SIF_OK) {
      SIF_LOG_ERROR(__TAG, "failed to evaluate the moments at radius %u", k);
      fft_workspace_free(ws);
      sif_delta_moments_free(m);
      return NULL;
    }

    __store(m, k, sigma_sq, high_k);
  }

  fft_workspace_free(ws);

  SIF_LOG_INFO(__TAG, "sigma_0..sigma_%u measured from the field", order);

  return m;
}

