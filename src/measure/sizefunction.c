#include "sif/measure/sizefunction.h"

#include "structures/results_internal.h"

#include "sif/utils/logger.h"
#include "sif/utils/sort.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

SIF_DEFINE_QUICKSORT(sif_sort_real_array, real_t, a < b)

/*
 * Bins every void whose radius lies in [true_r_min, true_r_max] and normalizes
 * to a number density.
 *
 * Both branches clamp the top edge into the last bin, so a void sitting
 * exactly at true_r_max is counted rather than silently dropped.
 */
static void __sif_compute_histogram(sif_size_function_t* vsf,
  const sif_catalog_t* cat, real_t box_length, real_t true_r_min,
  real_t true_r_max, bool use_ln_bins) {

  const uint32_t n_bins = vsf->n_bins;
  uint64_t* out_counts = vsf->counts;

  const real_t lo = use_ln_bins ? REAL_LOG(vsf->r_min) : vsf->r_min;
  const real_t hi = use_ln_bins ? REAL_LOG(vsf->r_max) : vsf->r_max;
  const real_t width = (hi - lo) / (real_t)n_bins;

  if (!(width > 0.0f)) {
    SIF_LOG_ERROR("size_function",
      "degenerate radius range [%g, %g], cannot bin", (double)vsf->r_min,
      (double)vsf->r_max);
    return;
  }

  const real_t inv_width = 1.0f / width;

/* The array-section reduction gives each thread a private copy of the bins,
 * so the data-dependent index below is safe. */
#pragma omp parallel for schedule(static) reduction(+ : out_counts[ : n_bins])
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    const real_t r = cat->radii[i];
    if (r < true_r_min || r > true_r_max)
      continue;

    const real_t v = use_ln_bins ? REAL_LOG(r) : r;
    int64_t b = (int64_t)((v - lo) * inv_width);

    if (b < 0)
      b = 0;
    else if (b >= (int64_t)n_bins)
      b = (int64_t)n_bins - 1;

    out_counts[b]++;
  }

  /* Density = counts / (volume * bin width). With log binning the bin width is
   * in ln r, so vsf is dn/dlnr rather than dn/dr. */
  const double vol = (double)box_length * box_length * box_length;
  const double norm = vol * (double)width;

  for (uint32_t b = 0; b < n_bins; b++)
    vsf->vsf[b] = (real_t)((double)out_counts[b] / norm);
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

sif_size_function_t* sif_size_function_catalog(const sif_catalog_t* cat,
  real_t box_length, uint32_t n_bins, sif_option_t options, real_t r_min_in,
  real_t r_max_in) {

  const bool use_ln_bins = (options & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LN;

  if (!cat || cat->n_voids == 0 || n_bins == 0) {
    SIF_LOG_ERROR("size_function", "invalid catalog or bin count");
    return NULL;
  }

  if (!(box_length > 0.0f)) {
    SIF_LOG_ERROR("size_function", "box_length must be positive");
    return NULL;
  }

  /* 1. Physical boundaries: honour explicit bounds, otherwise scan. */
  real_t true_r_min = (r_min_in > 0.0f) ? r_min_in : cat->radii[0];
  real_t true_r_max = (r_max_in > 0.0f) ? r_max_in : cat->radii[0];

  if (r_min_in <= 0.0f || r_max_in <= 0.0f) {
    for (uint64_t i = 0; i < cat->n_voids; i++) {
      if (r_min_in <= 0.0f && cat->radii[i] < true_r_min)
        true_r_min = cat->radii[i];
      if (r_max_in <= 0.0f && cat->radii[i] > true_r_max)
        true_r_max = cat->radii[i];
    }
  }

  if (!(true_r_max > true_r_min)) {
    SIF_LOG_ERROR("size_function",
      "degenerate radius range [%g, %g]: need at least two distinct radii",
      (double)true_r_min, (double)true_r_max);
    return NULL;
  }

  if (use_ln_bins && !(true_r_min > 0.0f)) {
    SIF_LOG_ERROR("size_function",
      "log binning needs a strictly positive minimum radius (got %g)",
      (double)true_r_min);
    return NULL;
  }

  /* 2. Allocate the VSF struct */
  sif_size_function_t* vsf = __sif_size_function_alloc(n_bins);
  if (!vsf)
    return NULL;

  vsf->options = options;
  vsf->r_min = true_r_min;
  vsf->r_max = true_r_max;

  /* 3. Bin edges and centers */
  if (use_ln_bins) {
    const real_t log_rmin = REAL_LOG(true_r_min);
    const real_t log_rmax = REAL_LOG(true_r_max);
    const real_t dlogr = (log_rmax - log_rmin) / (real_t)n_bins;

    for (uint32_t i = 0; i <= n_bins; i++)
      vsf->r_edges[i] = REAL_EXP(log_rmin + i * dlogr);
    for (uint32_t i = 0; i < n_bins; i++)
      vsf->r_centers[i] = REAL_EXP(log_rmin + (i + 0.5f) * dlogr);
  } else {
    const real_t dr = (true_r_max - true_r_min) / (real_t)n_bins;

    for (uint32_t i = 0; i <= n_bins; i++)
      vsf->r_edges[i] = true_r_min + i * dr;
    for (uint32_t i = 0; i < n_bins; i++)
      vsf->r_centers[i] = true_r_min + (i + 0.5f) * dr;
  }

  /* 4. Bin and normalize */
  __sif_compute_histogram(
    vsf, cat, box_length, true_r_min, true_r_max, use_ln_bins);

  /* 5. Poisson error: the relative error on a bin is 1/sqrt(N). */
  for (uint32_t b = 0; b < n_bins; b++) {
    vsf->err[b] = (vsf->counts[b] > 0)
                    ? vsf->vsf[b] / REAL_SQRT((real_t)vsf->counts[b])
                    : 0.0f;
  }

  SIF_LOG_INFO("size_function",
    "size function computed over [%g, %g] in %u %s bins",
    (double)true_r_min, (double)true_r_max, n_bins,
    use_ln_bins ? "log" : "linear");

  return vsf;
}

/* * Internal helper to safely interpolate a VSF value at a specific radius
 */
static void __sif_interpolate_vsf(const sif_size_function_t* vsf,
  real_t r_target, real_t* out_val, real_t* out_err) {
  /* If we are completely outside the centers, return 0 */
  if (r_target <= vsf->r_centers[0]) {
    *out_val = vsf->vsf[0];
    *out_err = vsf->err[0];
    return;
  }
  if (r_target >= vsf->r_centers[vsf->n_bins - 1]) {
    *out_val = vsf->vsf[vsf->n_bins - 1];
    *out_err = vsf->err[vsf->n_bins - 1];
    return;
  }

  /* Find bounding bins */
  uint32_t idx = 0;
  for (uint32_t i = 0; i < vsf->n_bins - 1; i++) {
    if (r_target >= vsf->r_centers[i] && r_target <= vsf->r_centers[i + 1]) {
      idx = i;
      break;
    }
  }

  real_t r0 = vsf->r_centers[idx], r1 = vsf->r_centers[idx + 1];
  real_t v0 = vsf->vsf[idx], v1 = vsf->vsf[idx + 1];
  real_t e0 = vsf->err[idx], e1 = vsf->err[idx + 1];

  /* Use Log-Log interpolation for heavily skewed VSF data, fallback to linear
   * if zeros exist */
  if (v0 > 0.0f && v1 > 0.0f) {
    real_t t =
      (REAL_LOG(r_target) - REAL_LOG(r0)) / (REAL_LOG(r1) - REAL_LOG(r0));
    *out_val = REAL_EXP(REAL_LOG(v0) + t * (REAL_LOG(v1) - REAL_LOG(v0)));
    *out_err = e0 + t * (e1 - e0); /* Linear interp for errors is sufficient */
  } else {
    real_t t = (r_target - r0) / (r1 - r0);
    *out_val = v0 + t * (v1 - v0);
    *out_err = e0 + t * (e1 - e0);
  }
}

sif_size_function_t* sif_size_function_combine(
  const sif_size_function_t** vsfs, uint32_t n_vsfs, uint32_t master_bins,
  const sif_interval_t* domains, sif_option_t options) {

  if (!vsfs || n_vsfs == 0 || master_bins == 0)
    return NULL;

  uint32_t merge_strategy = (options & __SIF_VSF_MERGE_MASK);
  bool use_ln_bins = (options & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LN;

  /* 1. Determine absolute bounds across all domains */
  real_t abs_min = REAL_MAX_VAL;
  real_t abs_max = -REAL_MAX_VAL;

  for (uint32_t i = 0; i < n_vsfs; i++) {
    real_t current_min = domains ? domains[i].min : vsfs[i]->r_min;
    real_t current_max = domains ? domains[i].max : vsfs[i]->r_max;
    if (current_min < abs_min)
      abs_min = current_min;
    if (current_max > abs_max)
      abs_max = current_max;
  }

  if (abs_min >= abs_max)
    return NULL;

  /* 2. Allocate the Master VSF. counts stay zero: raw void counts are not
   * meaningful once several catalogs have been interpolated onto a shared
   * grid. */
  sif_size_function_t* master = __sif_size_function_alloc(master_bins);
  if (!master)
    return NULL;

  master->options = options;
  master->r_min = abs_min;
  master->r_max = abs_max;

  /* 3. Build the Master Grid */
  if (use_ln_bins) {
    real_t log_rmin = REAL_LOG(abs_min);
    real_t log_rmax = REAL_LOG(abs_max);
    real_t dlogr = (log_rmax - log_rmin) / (real_t)master_bins;
    for (uint32_t i = 0; i <= master_bins; i++)
      master->r_edges[i] = REAL_EXP(log_rmin + i * dlogr);
    for (uint32_t i = 0; i < master_bins; i++)
      master->r_centers[i] = REAL_EXP(log_rmin + (i + 0.5f) * dlogr);
  } else {
    real_t dr = (abs_max - abs_min) / (real_t)master_bins;
    for (uint32_t i = 0; i <= master_bins; i++)
      master->r_edges[i] = abs_min + i * dr;
    for (uint32_t i = 0; i < master_bins; i++)
      master->r_centers[i] = abs_min + (i + 0.5f) * dr;
  }

  /* 4. Interpolate and Merge */
  real_t* temp_vals = malloc((size_t)n_vsfs * sizeof(real_t));
  real_t* temp_errs = malloc((size_t)n_vsfs * sizeof(real_t));

  if (!temp_vals || !temp_errs) {
    free(temp_vals);
    free(temp_errs);
    sif_size_function_free(master);
    return NULL;
  }

  for (uint32_t b = 0; b < master_bins; b++) {
    real_t r_target = master->r_centers[b];
    uint32_t valid_count = 0;

    /* Gather data from all catalogs that are valid at this radius */
    for (uint32_t i = 0; i < n_vsfs; i++) {
      real_t current_min = domains ? domains[i].min : vsfs[i]->r_min;
      real_t current_max = domains ? domains[i].max : vsfs[i]->r_max;

      if (r_target >= current_min && r_target <= current_max) {
        __sif_interpolate_vsf(
          vsfs[i], r_target, &temp_vals[valid_count], &temp_errs[valid_count]);
        valid_count++;

        /* If Stitch mode, the first valid catalog takes priority and we break
         */
        if (merge_strategy == SIF_VSF_MERGE_STITCH)
          break;
      }
    }

    if (valid_count == 0) {
      master->vsf[b] = 0.0f;
      master->err[b] = 0.0f;
      continue;
    }

    /* Apply requested statistical operation */
    if (merge_strategy == SIF_VSF_MERGE_STITCH || valid_count == 1) {
      master->vsf[b] = temp_vals[0];
      master->err[b] = temp_errs[0];
    } else if (merge_strategy == SIF_VSF_MERGE_MEAN) {
      double sum_v = 0.0, sum_e2 = 0.0;
      for (uint32_t k = 0; k < valid_count; k++) {
        sum_v += temp_vals[k];
        sum_e2 += temp_errs[k] * temp_errs[k];
      }
      master->vsf[b] = (real_t)(sum_v / valid_count);
      master->err[b] = (real_t)(REAL_SQRT(sum_e2) / valid_count);
    } else if (merge_strategy == SIF_VSF_MERGE_MEDIAN) {
      sif_sort_real_array(temp_vals, valid_count);
      master->vsf[b] = temp_vals[valid_count / 2];
      /* For median error, taking the mean of errors is a stable proxy without
       * heavy bootstrapping */
      double sum_e = 0.0;
      for (uint32_t k = 0; k < valid_count; k++)
        sum_e += temp_errs[k];
      master->err[b] = (real_t)(sum_e / valid_count);
    }
  }

  free(temp_vals);
  free(temp_errs);

  SIF_LOG_INFO("size_function",
    "successfully combined %u VSFs into master grid (Bins: %u)", n_vsfs,
    master_bins);
  return master;
}
