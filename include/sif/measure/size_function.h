/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file size_function.h
 * @brief Measuring the void size function from a catalogue, and merging
 * several into one.
 */

#ifndef SIF_MEASURE_SIZE_FUNCTION_H
#define SIF_MEASURE_SIZE_FUNCTION_H

#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/size_function.h"

/**
 * @brief Bin a catalogue into a void size function.
 *
 * Counts voids per radius bin and divides by the bin width and the box volume,
 * giving a number density rather than a histogram, so catalogues from
 * different volumes are comparable. The Poisson error follows from the raw
 * counts, which are kept.
 *
 * @param cat Catalogue to bin.
 * @param box_length Physical side length of the box, which sets the volume the
 * density is per.
 * @param n_bins Radial bins, at least 1.
 * @param opt SIF_VSF_BIN_LN (default, bins uniform in ln R) or
 * SIF_VSF_BIN_LINEAR (uniform in R). Recorded in the result.
 * @param r_min_in Lower radius bound; pass 0 or less to take the catalogue's
 * smallest radius.
 * @param r_max_in Upper radius bound; pass 0 or less to take the catalogue's
 * largest.
 * @return Newly allocated size function, released with
 * sif_size_function_free(), or NULL on invalid input or failure.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_catalog(
  const sif_catalog_t* cat, sif_real box_length, uint32_t n_bins,
  sif_option opt, sif_real r_min_in, sif_real r_max_in);

/**
 * @brief Merge several size functions into one.
 *
 * Each input is resampled onto a common radius grid and the overlapping values
 * combined. This is how measurements from boxes of different resolution are
 * joined: each resolves a different range of radii well, and @p domains says
 * where each one should be trusted.
 *
 * @param vsfs The size functions to merge.
 * @param n_vsfs How many.
 * @param master_bins Radial bins in the merged result.
 * @param domains Optional, one interval per input giving the radius range it
 * contributes over. NULL uses each input's own full range.
 * @param opt SIF_VSF_MERGE_MEAN (default), SIF_VSF_MERGE_MEDIAN or
 * SIF_VSF_MERGE_STITCH.
 * @return Newly allocated size function, released with
 * sif_size_function_free(), or NULL on invalid input or failure.
 *
 * @note The inputs must agree on their binning convention: mixing a
 * SIF_VSF_BIN_LN measurement with a SIF_VSF_BIN_LINEAR one merges two
 * different quantities.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_combine(
  const sif_size_function_t** vsfs, uint32_t n_vsfs, uint32_t master_bins,
  const sif_interval_t* domains, sif_option opt);

#endif /* SIF_MEASURE_SIZE_FUNCTION_H */
