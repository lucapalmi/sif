/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF__MEASURE_DELTA_COMMON_H
#define SIF__MEASURE_DELTA_COMMON_H

/**
 * @file delta_common.h
 * @brief Internals shared by the delta distribution and delta moment
 * estimators. Private to the library.
 *
 * Both estimators start from the same prepared spectrum, and duplicating that
 * preparation is how the two would drift apart -- at which point a PDF and a
 * set of moments that are supposed to describe one field quietly describe two.
 */

#include "math/fft.h"
#include "sif/core/macros.h"
#include "sif/structures/grid.h"

/** @brief Log tag shared by both estimators. */
#define SIF__DELTA_TAG "delta"

/** @brief Below this many cells per radius the k-space window is aliased badly
 *  enough that the result is grid artefacts rather than field. */
#define SIF__DELTA_MIN_CELLS_PER_RADIUS 2.0

/**
 * @brief Validates radii against the grid resolution and the box size.
 *
 * A radius below a couple of cells is not a smaller measurement, it is a
 * different (wrong) one, so this fails rather than returning a plausible
 * looking answer.
 *
 * @return SIF_OK or SIF_ERR_INVALID
 */
int sif__delta_validate_radii(
  const sif_grid_t* grid, const sif_real* radii, uint32_t n_radii);

/**
 * @brief Resolves the smoothing window selected in the options bitmask.
 */
sif_filter_type_t sif__delta_filter(sif_option opt);

/**
 * @brief Validates the parts of the options bitmask both estimators share.
 *
 * @return SIF_OK or SIF_ERR_INVALID
 */
int sif__delta_validate_options(sif_option opt);

/**
 * @brief Builds the spectrum both estimators work from.
 *
 * Allocates a workspace, runs the forward transform, deconvolves the CIC
 * assignment window unless asked not to, and applies the phase shuffle. The
 * shuffle is a pure function of (seed, n_cells), so calling this twice with
 * the same seed -- once from each estimator -- yields the identical surrogate,
 * which is what lets the PDF and the moments describe the same field.
 *
 * The backward stage is *not* initialized: a caller that only needs the
 * moments never pays for it.
 *
 * @return The workspace, owned by the caller, or NULL on failure.
 */
SIF_NODISCARD sif_fft_workspace_t* sif__delta_prepare_spectrum(
  const sif_grid_t* grid, uint64_t seed, sif_option opt);

#endif /* SIF__MEASURE_DELTA_COMMON_H */
