/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/measure/size_function.h"

#include "structures/results_internal.h"

#include "sif/utils/logger.h"
#include "sif/utils/sort.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

SIF_DEFINE_QUICKSORT(sif_sort_real_array, sif_real, a < b)

/*
 * Bins every void whose radius lies in [true_r_min, true_r_max] and normalizes
 * to a number density.
 *
 * Both branches clamp the top edge into the last bin, so a void sitting
 * exactly at true_r_max is counted rather than silently dropped.
 */
static void fill_histogram(sif_size_function_t* vsf, const sif_catalog_t* cat,
  sif_real box_length, sif_real true_r_min, sif_real true_r_max,
  bool use_ln_bins) {

  const uint32_t n_bins = vsf->n_bins;
  uint64_t* out_counts = vsf->counts;

  const sif_real lo = use_ln_bins ? SIF_REAL_LOG(vsf->r_min) : vsf->r_min;
  const sif_real hi = use_ln_bins ? SIF_REAL_LOG(vsf->r_max) : vsf->r_max;
  const sif_real width = (hi - lo) / (sif_real)n_bins;

  if (!(width > 0.0f)) {
    SIF_LOG_ERROR("size_function",
      "degenerate radius range [%g, %g], cannot bin", (double)vsf->r_min,
      (double)vsf->r_max);
    return;
  }

  const sif_real inv_width = 1.0f / width;

/* The array-section reduction gives each thread a private copy of the bins, so
 * the data-dependent index below is a private write rather than a scatter into
 * shared memory. It is an OpenMP 4.5 feature, which CMakeLists requires
 * explicitly; the alternative is the hand-rolled per-thread scratch that
 * sif__fft_spectral_moments uses, and here the loop is over voids rather than
 * over cells, so the reduction's clarity is worth more than the control. */
#pragma omp parallel for schedule(static) reduction(+ : out_counts[ : n_bins])
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    const sif_real r = cat->radii[i];
    if (r < true_r_min || r > true_r_max)
      continue;

    const sif_real v = use_ln_bins ? SIF_REAL_LOG(r) : r;
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
    vsf->vsf[b] = (sif_real)((double)out_counts[b] / norm);
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

sif_size_function_t* sif_size_function_catalog(const sif_catalog_t* cat,
  sif_real box_length, uint32_t n_bins, sif_option options, sif_real r_min_in,
  sif_real r_max_in) {

  const bool use_ln_bins = (options & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LN;

  if (!cat || cat->n_voids == 0 || n_bins == 0) {
    SIF_LOG_ERROR("size_function", "invalid catalog or bin count");
    return NULL;
  }

  if (!(box_length > 0.0f)) {
    SIF_LOG_ERROR("size_function", "box_length must be positive");
    return NULL;
  }

  /* 1. Physical boundaries: honour explicit bounds, otherwise scan. */
  sif_real true_r_min = (r_min_in > 0.0f) ? r_min_in : cat->radii[0];
  sif_real true_r_max = (r_max_in > 0.0f) ? r_max_in : cat->radii[0];

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
  sif_size_function_t* vsf = sif__size_function_alloc(n_bins);
  if (!vsf)
    return NULL;

  vsf->options = options;
  vsf->source_id = sif_catalog_id(cat);
  vsf->r_min = true_r_min;
  vsf->r_max = true_r_max;

  /* 3. Bin edges and centers */
  if (use_ln_bins) {
    const sif_real log_rmin = SIF_REAL_LOG(true_r_min);
    const sif_real log_rmax = SIF_REAL_LOG(true_r_max);
    const sif_real dlogr = (log_rmax - log_rmin) / (sif_real)n_bins;

    for (uint32_t i = 0; i <= n_bins; i++)
      vsf->r_edges[i] = SIF_REAL_EXP(log_rmin + i * dlogr);
    for (uint32_t i = 0; i < n_bins; i++)
      vsf->r_centers[i] = SIF_REAL_EXP(log_rmin + (i + 0.5f) * dlogr);
  } else {
    const sif_real dr = (true_r_max - true_r_min) / (sif_real)n_bins;

    for (uint32_t i = 0; i <= n_bins; i++)
      vsf->r_edges[i] = true_r_min + i * dr;
    for (uint32_t i = 0; i < n_bins; i++)
      vsf->r_centers[i] = true_r_min + (i + 0.5f) * dr;
  }

  /* 4. Bin and normalize */
  fill_histogram(vsf, cat, box_length, true_r_min, true_r_max, use_ln_bins);

  /* 5. Poisson error: the relative error on a bin is 1/sqrt(N). */
  for (uint32_t b = 0; b < n_bins; b++) {
    vsf->err[b] = (vsf->counts[b] > 0)
                    ? vsf->vsf[b] / SIF_REAL_SQRT((sif_real)vsf->counts[b])
                    : 0.0f;
  }

  SIF_LOG_INFO("size_function",
    "size function computed over [%g, %g] in %u %s bins", (double)true_r_min,
    (double)true_r_max, n_bins, use_ln_bins ? "log" : "linear");

  return vsf;
}

/*
 * Reads one size function at an arbitrary radius.
 *
 * Outside the bin centres the nearest one is held flat rather than
 * extrapolated: a size function falls steeply with radius, and a log-log line
 * continued past the last populated bin produces a confident number where the
 * catalogue has no voids at all.
 */
static void interpolate_vsf(const sif_size_function_t* vsf, sif_real r_target,
  sif_real* out_val, sif_real* out_err) {
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

  sif_real r0 = vsf->r_centers[idx], r1 = vsf->r_centers[idx + 1];
  sif_real v0 = vsf->vsf[idx], v1 = vsf->vsf[idx + 1];
  sif_real e0 = vsf->err[idx], e1 = vsf->err[idx + 1];

  /* Use Log-Log interpolation for heavily skewed VSF data, fallback to linear
   * if zeros exist */
  if (v0 > 0.0f && v1 > 0.0f) {
    sif_real t = (SIF_REAL_LOG(r_target) - SIF_REAL_LOG(r0)) /
                 (SIF_REAL_LOG(r1) - SIF_REAL_LOG(r0));
    *out_val = SIF_REAL_EXP(
      SIF_REAL_LOG(v0) + t * (SIF_REAL_LOG(v1) - SIF_REAL_LOG(v0)));
    *out_err = e0 + t * (e1 - e0); /* Linear interp for errors is sufficient */
  } else {
    sif_real t = (r_target - r0) / (r1 - r0);
    *out_val = v0 + t * (v1 - v0);
    *out_err = e0 + t * (e1 - e0);
  }
}

sif_size_function_t* sif_size_function_combine(const sif_size_function_t** vsfs,
  uint32_t n_vsfs, uint32_t master_bins, const sif_interval_t* domains,
  sif_option options) {

  if (!vsfs || n_vsfs == 0 || master_bins == 0) {
    SIF_LOG_ERROR("size_function", "invalid arguments to combine");
    return NULL;
  }

  for (uint32_t i = 0; i < n_vsfs; i++) {
    if (!vsfs[i] || vsfs[i]->n_bins == 0) {
      SIF_LOG_ERROR("size_function", "size function %u is missing or empty", i);
      return NULL;
    }
  }

  const uint32_t merge_strategy = (options & SIF__VSF_MERGE_MASK);
  const bool use_ln_bins = (options & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LN;

  /* The mask admits four values and only three are defined. An unrecognized
   * one used to match none of the branches below, leaving every master bin at
   * whatever the allocator returned -- a result-shaped object full of nothing.
   */
  if (merge_strategy != SIF_VSF_MERGE_MEAN &&
      merge_strategy != SIF_VSF_MERGE_MEDIAN &&
      merge_strategy != SIF_VSF_MERGE_STITCH) {
    SIF_LOG_ERROR(
      "size_function", "unrecognized merge strategy in the options bitmask");
    return NULL;
  }

  /* 1. Determine absolute bounds across all domains */
  sif_real abs_min = SIF_REAL_MAX_VAL;
  sif_real abs_max = -SIF_REAL_MAX_VAL;

  for (uint32_t i = 0; i < n_vsfs; i++) {
    sif_real current_min = domains ? domains[i].min : vsfs[i]->r_min;
    sif_real current_max = domains ? domains[i].max : vsfs[i]->r_max;
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
  sif_size_function_t* master = sif__size_function_alloc(master_bins);
  if (!master)
    return NULL;

  master->options = options;
  master->r_min = abs_min;
  master->r_max = abs_max;

  /* 3. Build the Master Grid */
  if (use_ln_bins) {
    sif_real log_rmin = SIF_REAL_LOG(abs_min);
    sif_real log_rmax = SIF_REAL_LOG(abs_max);
    sif_real dlogr = (log_rmax - log_rmin) / (sif_real)master_bins;
    for (uint32_t i = 0; i <= master_bins; i++)
      master->r_edges[i] = SIF_REAL_EXP(log_rmin + i * dlogr);
    for (uint32_t i = 0; i < master_bins; i++)
      master->r_centers[i] = SIF_REAL_EXP(log_rmin + (i + 0.5f) * dlogr);
  } else {
    sif_real dr = (abs_max - abs_min) / (sif_real)master_bins;
    for (uint32_t i = 0; i <= master_bins; i++)
      master->r_edges[i] = abs_min + i * dr;
    for (uint32_t i = 0; i < master_bins; i++)
      master->r_centers[i] = abs_min + (i + 0.5f) * dr;
  }

  /* 4. Interpolate and Merge */
  sif_real* temp_vals = malloc((size_t)n_vsfs * sizeof(sif_real));
  sif_real* temp_errs = malloc((size_t)n_vsfs * sizeof(sif_real));

  if (!temp_vals || !temp_errs) {
    free(temp_vals);
    free(temp_errs);
    sif_size_function_free(master);
    return NULL;
  }

  for (uint32_t b = 0; b < master_bins; b++) {
    sif_real r_target = master->r_centers[b];
    uint32_t valid_count = 0;

    /* Gather data from all catalogs that are valid at this radius */
    for (uint32_t i = 0; i < n_vsfs; i++) {
      sif_real current_min = domains ? domains[i].min : vsfs[i]->r_min;
      sif_real current_max = domains ? domains[i].max : vsfs[i]->r_max;

      if (r_target >= current_min && r_target <= current_max) {
        interpolate_vsf(
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
      master->vsf[b] = (sif_real)(sum_v / valid_count);
      master->err[b] = (sif_real)(SIF_REAL_SQRT(sum_e2) / valid_count);
    } else if (merge_strategy == SIF_VSF_MERGE_MEDIAN) {
      sif_sort_real_array(temp_vals, valid_count);

      /* An even count has no single middle element, and taking the upper one
       * biases the merge high -- consistently, in the same direction, for
       * every bin. Averaging the two is the definition every other tool uses.
       */
      if (valid_count % 2 == 0) {
        const double a = (double)temp_vals[valid_count / 2 - 1];
        const double b_val = (double)temp_vals[valid_count / 2];
        master->vsf[b] = (sif_real)(0.5 * (a + b_val));
      } else {
        master->vsf[b] = temp_vals[valid_count / 2];
      }

      /* The errors are not re-sorted alongside the values, which does not
       * matter because only their mean is taken: a stable proxy for the
       * spread of a median without bootstrapping it. */
      double sum_e = 0.0;
      for (uint32_t k = 0; k < valid_count; k++)
        sum_e += temp_errs[k];
      master->err[b] = (sif_real)(sum_e / valid_count);
    }
  }

  free(temp_vals);
  free(temp_errs);

  SIF_LOG_INFO("size_function",
    "successfully combined %u VSFs into master grid (Bins: %u)", n_vsfs,
    master_bins);
  return master;
}
