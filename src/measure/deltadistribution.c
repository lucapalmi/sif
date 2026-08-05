#include "sif/measure/deltadistribution.h"

#include "core/get_system.h"
#include "delta_common.h"
#include "structures/results_internal.h"
#include "math/fft.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <stdlib.h>
#include <string.h>

#define __TAG __SIF_DELTA_TAG

/*
 * Histograms one filtered field into the row of `out` for this radius.
 * Counters are uint64: a float accumulator saturates at 2^24, which N^3 cells
 * reach for N >= 256 and would silently flat-top the mode of the PDF.
 */
static int __histogram_field(const real_t* data, uint64_t total_cells,
  const real_t delta_bounds[2], uint32_t n_bins, real_t inv_bin_width,
  real_t* out, real_t pdf_norm) {

  const int n_threads =
    system_get_max_threads() > 0 ? system_get_max_threads() : 1;

  uint64_t* scratch =
    sif_calloc_aligned((size_t)n_threads * n_bins, sizeof(uint64_t));
  if (!scratch) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the histogram scratch");
    return SIF_ERR_ALLOC;
  }

  const real_t lo = delta_bounds[0];
  const real_t hi = delta_bounds[1];

#pragma omp parallel num_threads(n_threads)
  {
    uint64_t* local = scratch + (size_t)system_get_thread_num() * n_bins;

#pragma omp for schedule(static)
    for (uint64_t i = 0; i < total_cells; i++) {
      const real_t d = data[i];

      /* NaN-safe: a NaN fails both comparisons and is dropped. */
      if (d >= lo && d <= hi) {
        uint32_t bin = (uint32_t)((d - lo) * inv_bin_width);
        if (bin >= n_bins)
          bin = n_bins - 1;
        local[bin]++;
      }
    }
  }

  for (uint32_t b = 0; b < n_bins; b++) {
    uint64_t total = 0;
    for (int t = 0; t < n_threads; t++) {
      total += scratch[(size_t)t * n_bins + b];
    }
    out[b] = (real_t)((double)total * (double)pdf_norm);
  }

  sif_free_aligned(scratch);
  return SIF_OK;
}

sif_delta_distribution_t* sif_delta_distribution_grid(
  const sif_grid_t* grid, const real_t* radii, uint32_t n_radii,
  uint32_t n_bins, const real_t delta_bounds[2], uint64_t seed,
  sif_option_t opt) {

  if (!grid || !grid->delta || !radii || !delta_bounds || n_radii == 0 ||
      n_bins == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to sif_delta_distribution_grid");
    return NULL;
  }

  if (!(delta_bounds[0] < delta_bounds[1])) {
    SIF_LOG_ERROR(
      __TAG, "delta_bounds[0] must be strictly less than delta_bounds[1]");
    return NULL;
  }

  if (__sif_delta_validate_radii(grid, radii, n_radii) != SIF_OK)
    return NULL;

  if (__sif_delta_validate_options(opt) != SIF_OK)
    return NULL;

  const filter_type_t filter = __sif_delta_filter(opt);

  sif_delta_distribution_t* dist =
    __sif_delta_distribution_alloc(n_radii, n_bins);
  if (!dist)
    return NULL;

  memcpy(dist->radii, radii, (size_t)n_radii * sizeof(real_t));
  dist->n_samples = grid->total_cells;

  const real_t bin_width =
    (real_t)(((double)delta_bounds[1] - (double)delta_bounds[0]) / n_bins);
  const real_t inv_bin_width = (real_t)(1.0 / (double)bin_width);

  for (uint32_t i = 0; i < n_bins; i++) {
    dist->delta_edges[i] = delta_bounds[0] + (real_t)i * bin_width;
  }
  /* Set explicitly rather than let the accumulated bin_width drift land it a
   * few ulp away from the bound the histogram actually clamps against. */
  dist->delta_edges[n_bins] = delta_bounds[1];

  SIF_LOG_INFO(__TAG,
    "measuring the delta PDF over %u radii on a %u^3 grid (%s window, %s)",
    n_radii, grid->n_cells,
    filter == FILTER_GAUSSIAN ? "Gaussian" : "top-hat",
    (opt & __SIF_DELTA_SHUFFLE_MASK) == SIF_DELTA_SHUFFLE_NONE
      ? "field as measured"
    : (opt & __SIF_DELTA_SHUFFLE_MASK) == SIF_DELTA_SHUFFLE_PHASES
      ? "phase-randomized surrogate"
      : "Gaussian surrogate");

  fft_workspace_t* ws = __sif_delta_prepare_spectrum(grid, seed, opt);
  if (!ws) {
    sif_delta_distribution_free(dist);
    return NULL;
  }

  system_state_t* state = get_system_state();
  if (fft_workspace_init_backward(ws, state->fft_mgr) != SIF_OK) {
    SIF_LOG_ERROR(__TAG, "failed to initialize the backward FFT");
    goto fail;
  }

  const real_t pdf_norm =
    (real_t)(1.0 / ((double)grid->total_cells * (double)bin_width));

  for (uint32_t k = 0; k < n_radii; k++) {
    if (fft_apply_filter(ws, filter, radii[k], grid->box_length) != SIF_OK) {
      SIF_LOG_ERROR(__TAG, "failed to apply the filter at radius %u", k);
      goto fail;
    }

    /* Runs in place on delta_k_cpy, which the next iteration's filter fully
     * overwrites from the untouched delta_k. */
    real_t* filtered = fft_grid_backward(ws);
    if (!filtered) {
      SIF_LOG_ERROR(__TAG, "backward transform failed at radius %u", k);
      goto fail;
    }

    if (__histogram_field(filtered, grid->total_cells, delta_bounds, n_bins,
          inv_bin_width, dist->distributions + (size_t)k * n_bins,
          pdf_norm) != SIF_OK) {
      goto fail;
    }
  }

  fft_workspace_free(ws);

  SIF_LOG_INFO(__TAG, "delta distribution computation complete");

  return dist;

fail:
  fft_workspace_free(ws);
  sif_delta_distribution_free(dist);
  return NULL;
}
