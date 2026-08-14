/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/size_function.h"

#include "results_internal.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>

/*
 * One container for two quite different things: a histogram of a catalogue,
 * and a model evaluated pointwise.
 *
 * They share it because everything downstream -- merging, interpolation,
 * plotting, the Python view -- wants one shape. What differs is which fields
 * carry meaning: a measurement fills counts and the Poisson err, a model
 * leaves both zero because it counted nothing and has no sampling
 * uncertainty. Zero is the honest value there, which is why every array is
 * cleared here rather than left to the producer.
 */
sif_size_function_t* sif__size_function_alloc(uint32_t n_bins) {

  sif_size_function_t* vsf = calloc(1, sizeof(sif_size_function_t));
  if (!vsf) {
    SIF_LOG_ERROR("size_function", "failed to allocate the VSF struct");
    return NULL;
  }

  vsf->n_bins = n_bins;

  /* Plain malloc/calloc, not the aligned allocator the bulk arrays use: these
   * are a handful of values per bin, read once at the end of a run and never
   * in a vectorized loop, so a cache-line-aligned arena would buy nothing and
   * cost a second free convention in the public API. */
  vsf->r_edges = malloc(((size_t)n_bins + 1) * sizeof(sif_real));
  vsf->r_centers = malloc((size_t)n_bins * sizeof(sif_real));
  vsf->counts = calloc(n_bins, sizeof(uint64_t));
  vsf->vsf = calloc(n_bins, sizeof(sif_real));
  vsf->err = calloc(n_bins, sizeof(sif_real));

  if (!vsf->r_edges || !vsf->r_centers || !vsf->counts || !vsf->vsf ||
      !vsf->err) {
    SIF_LOG_ERROR("size_function", "failed to allocate the VSF arrays");
    sif_size_function_free(vsf);
    return NULL;
  }

  return vsf;
}

void sif_size_function_free(sif_size_function_t* vsf) {
  if (!vsf)
    return;

  free(vsf->r_edges);
  free(vsf->r_centers);
  free(vsf->counts);
  free(vsf->vsf);
  free(vsf->err);
  free(vsf);
}

void sif__edges_from_centers(
  const sif_real* centers, uint32_t n, sif_real* edges) {

  if (n == 0)
    return;

  /* A single centre has no neighbour to take a midpoint against, and the
   * extrapolation below would read the edge it is about to write. There is no
   * spacing to infer, so the bin is given an arbitrary but harmless decade
   * around the point -- enough for the container to be complete and plottable,
   * which is all these edges are for. */
  if (n == 1) {
    edges[0] = (sif_real)((double)centers[0] / 1.1);
    edges[1] = (sif_real)((double)centers[0] * 1.1);
    return;
  }

  for (uint32_t i = 1; i < n; i++)
    edges[i] = (sif_real)sqrt((double)centers[i - 1] * (double)centers[i]);

  /* Extrapolate the outer two, which reproduces the exact midpoints when the
   * radii are geometrically spaced. */
  edges[0] = (sif_real)((double)centers[0] * centers[0] / (double)edges[1]);
  edges[n] =
    (sif_real)((double)centers[n - 1] * centers[n - 1] / (double)edges[n - 1]);
}
