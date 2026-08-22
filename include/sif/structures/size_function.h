/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file size_function.h
 * @brief The void size function: number density of voids per radius bin.
 *
 * One container serves both a measured size function, binned from a catalogue,
 * and a modelled one evaluated pointwise. That is what lets the two be
 * compared, plotted and combined without converting between representations --
 * at the cost of a few fields being meaningful in only one of the two cases,
 * noted individually below.
 */

#ifndef SIF_STRUCTURES_SIZE_FUNCTION_H
#define SIF_STRUCTURES_SIZE_FUNCTION_H

#include <stdint.h>

#include "sif/core/macros.h"

/**
 * @brief A void size function, measured or modelled.
 *
 * Produced either by binning a catalogue (sif_size_function_catalog()) or by a
 * model; released with sif_size_function_free().
 */
typedef struct {
  uint32_t n_bins;
  /** Options the size function was produced with. Records in particular
   * whether #vsf is per unit ln R (SIF_VSF_BIN_LN) or per unit R
   * (SIF_VSF_BIN_LINEAR), which nothing else in the struct reveals.
   */
  sif_option options;

  /** Lower edge of the first bin. */
  sif_real r_min;
  /** Upper edge of the last bin. */
  sif_real r_max;

  /** n_bins + 1 bin edges. */
  sif_real* r_edges;
  /** n_bins bin centres. */
  sif_real* r_centers;

  /** Raw void count per bin. Zero for a model, which is evaluated pointwise
   * and never counts anything.
   */
  uint64_t* counts;
  /** Normalized number density per bin: the size function itself. */
  sif_real* vsf;
  /** Poisson error on #vsf, from #counts. Zero for a model. */
  sif_real* err;

  /**
   * @brief Which catalogue this was binned from: that catalogue's
   * sif_catalog_t::id, or 0 when it came from none.
   *
   * Zero for a model, which is evaluated from parameters and counted nothing,
   * and zero for a stitched size function, which came from several catalogues
   * and so belongs to no single one. An `.sdf` file stores only the ones that
   * name a catalogue it holds.
   */
  uint64_t source_id;
} sif_size_function_t;

/**
 * @brief A closed interval of radius.
 */
typedef struct {
  sif_real min;
  sif_real max;
} sif_interval_t;

/**
 * @brief Release a size function.
 * @param vsf Size function to free. NULL is accepted and ignored.
 */
void sif_size_function_free(sif_size_function_t* vsf);

/**
 * @brief The normalized number density in one bin.
 *
 * @param vsf The size function.
 * @param bin_idx Bin to read; must be below sif_size_function_t::n_bins.
 * @return The value of #vsf in that bin.
 *
 * @warning The index is not checked. Reading past the last bin is undefined.
 */
static inline sif_real sif_size_function_get(
  const sif_size_function_t* vsf, uint32_t bin_idx) {
  SIF_ASSERT(bin_idx < vsf->n_bins);
  return vsf->vsf[bin_idx];
}

#endif /* SIF_STRUCTURES_SIZE_FUNCTION_H */
