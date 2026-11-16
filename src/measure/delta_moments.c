/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/measure/delta_moments.h"

#include "delta_common.h"
#include "math/fft.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/results_internal.h"
#include "utils/log_tally.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TAG SIF__DELTA_TAG

/* high_k_fraction above this is reported: past it the moment is dominated by
 * modes at the edge of the available k range and is not a statement about the
 * field any more. */
#define HIGH_K_WARN_LEVEL 0.1

/* What store() found wrong, per moment, over every radius. */
typedef struct {
  sif_log_tally_t noise[SIF_MAX_MOMENT_ORDER + 1];
  sif_log_tally_t high_k[SIF_MAX_MOMENT_ORDER + 1];
} findings_t;

/*
 * Stores one radius' column of results. A shot-noise subtraction can push a
 * sigma^2 negative, which is a real outcome -- the moment was noise -- so it
 * is reported as a zero sigma and flagged rather than square-rooted into a
 * NaN. What is flagged is counted, and reported once by report().
 */
static void store(sif_delta_moments_t* m, uint32_t k, const double* sigma_sq,
  const double* high_k, findings_t* found) {

  for (uint8_t j = 0; j < m->n_moments; j++) {
    const size_t at = m->offsets[j] + k;

    m->sigma[at] = (sif_real)(sigma_sq[j] > 0.0 ? sqrt(sigma_sq[j]) : 0.0);
    m->high_k_fraction[at] = (sif_real)high_k[j];

    if (sigma_sq[j] <= 0.0)
      sif__log_tally_add(&found->noise[j], (double)m->radii[k], -sigma_sq[j]);
    if (high_k[j] > HIGH_K_WARN_LEVEL)
      sif__log_tally_add(&found->high_k[j], (double)m->radii[k], high_k[j]);
  }
}

static void report(const sif_delta_moments_t* m, const findings_t* found) {
  for (uint8_t j = 0; j < m->n_moments; j++) {
    const sif_log_tally_t* t = &found->noise[j];
    if (t->n)
      SIF_LOG_WARNING(TAG,
        "sigma_%u: not positive after shot-noise subtraction at %u of %u "
        "radii, R from %g to %g; the moment is dominated by Poisson noise",
        j, t->n, m->n_radii, t->r_min, t->r_max);
    t = &found->high_k[j];
    if (t->n)
      SIF_LOG_WARNING(TAG,
        "sigma_%u: up to %.1f%% of the sum from the top half of the available "
        "k range at %u of %u radii, R from %g to %g; the moment is "
        "resolution-limited, not field-limited",
        j, 100.0 * t->worst, t->n, m->n_radii, t->r_min, t->r_max);
  }
}

static int validate_order(uint8_t order) {
  if (order > SIF_MAX_MOMENT_ORDER) {
    SIF_LOG_ERROR(TAG, "moment order %u exceeds the maximum of %d", order,
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

sif_delta_moments_t* sif_delta_moments_grid(const sif_grid_t* grid,
  const sif_real* radii, uint32_t n_radii, uint8_t order, uint64_t n_tracers,
  uint64_t seed, sif_option opt) {

  if (!grid || !grid->values || !radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "invalid arguments to sif_delta_moments_grid");
    return NULL;
  }

  if (validate_order(order) != SIF_OK)
    return NULL;

  if (sif__delta_validate_radii(grid, radii, n_radii) != SIF_OK)
    return NULL;

  if (sif__delta_validate_options(opt) != SIF_OK)
    return NULL;

  const sif_filter_type_t filter = sif__delta_filter(opt);

  if (filter == SIF__FILTER_TOP_HAT && order >= 2) {
    SIF_LOG_INFO(TAG,
      "the top-hat W^2 decays only as k^-4, so sigma_2 and above are cut off "
      "by the grid rather than by the field; check high_k_fraction or switch "
      "to SIF_DELTA_FILTER_GAUSSIAN");
  }

  sif_delta_moments_t* m = sif__delta_moments_alloc(n_radii, order);
  if (!m)
    return NULL;

  memcpy(m->radii, radii, (size_t)n_radii * sizeof(sif_real));

  SIF_LOG_INFO(TAG, "sigma_0..sigma_%u: %u radii on a %u^3 grid (%s window)",
    order, n_radii, grid->n_cells,
    filter == SIF__FILTER_GAUSSIAN ? "Gaussian" : "top-hat");

  sif_fft_workspace_t* ws = sif__delta_prepare_spectrum(grid, seed, opt);
  if (!ws) {
    sif_delta_moments_free(m);
    return NULL;
  }

  double sigma_sq[SIF_MAX_MOMENT_ORDER + 1];
  double high_k[SIF_MAX_MOMENT_ORDER + 1];
  findings_t found = {0};

  for (uint32_t k = 0; k < n_radii; k++) {
    /* Every order shares one pass over the spectrum. */
    if (sif__fft_spectral_moments(ws, filter, radii[k], grid->box_length, order,
          n_tracers, sigma_sq, high_k) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "failed to evaluate the moments at radius %u", k);
      sif__fft_workspace_free(ws);
      sif_delta_moments_free(m);
      return NULL;
    }

    store(m, k, sigma_sq, high_k, &found);
  }

  sif__fft_workspace_free(ws);
  report(m, &found);

  return m;
}
