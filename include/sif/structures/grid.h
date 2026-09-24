/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file grid.h
 * @brief Cubic mesh holding a scalar field sampled on cells.
 *
 * The usual sequence is to allocate a grid, deposit a particle field onto it
 * with sif_grid_assign_cic(), then convert the result in place to the density
 * contrast with sif_grid_to_density_contrast().
 */

#ifndef SIF_STRUCTURES_GRID_H
#define SIF_STRUCTURES_GRID_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

/**
 * @brief What a grid's cells currently hold.
 *
 * The same array means different things at different points in the pipeline,
 * and mixing them up is a physics error that produces numbers rather than a
 * diagnostic -- normalizing an already-normalized field, or measuring moments
 * of a grid that still holds densities. The grid therefore records which it is.
 */
typedef enum {
  /** Allocated and zeroed; nothing deposited. */
  SIF_GRID_EMPTY = 0,
  /**
   * Density: accumulated weight **per unit volume**.
   *
   * Per unit volume and not per cell -- sif_grid_assign_cic() divides by the
   * cell volume on its way out. Summing the cells therefore gives the total
   * weight divided by the cell volume, not the total weight; multiply by
   * `cell_length^3` to recover it. The distinction cancels in
   * sif_grid_to_density_contrast(), which normalizes by the mean, and matters
   * only to code reading these values directly.
   */
  SIF_GRID_DENSITY,
  /** delta = rho / rho_mean - 1. */
  SIF_GRID_DENSITY_CONTRAST
} sif_grid_content_t;

/**
 * @brief A cubic grid of `n_cells^3` cells over a periodic box.
 */
typedef struct {
  /**
   * Cell values, flat and row-major with z varying fastest.
   *
   * What they mean is recorded in #content, and changes as the pipeline
   * proceeds: a density after sif_grid_assign_cic(), the density contrast
   * after sif_grid_to_density_contrast() has overwritten them in place.
   */
  sif_real* values;

  /** What #values currently holds. */
  sif_grid_content_t content;

  /** Cells per side. */
  uint32_t n_cells;
  /** n_cells^3. */
  uint64_t total_cells;
  /** Physical side length of the box. */
  sif_real box_length;
  /** Physical side length of one cell. */
  sif_real cell_length;

  /**
   * Fast path for the flat index when #n_cells is a power of two: an index can
   * then be masked and shifted instead of multiplied and divided. Both are 0
   * when #n_cells is not a power of two, which is the signal to take the
   * general path.
   */
  /** n_cells - 1, or 0. */
  uint32_t p2_mask;
  /** log2(n_cells), or 0. */
  uint32_t p2_shift;
} sif_grid_t;

/**
 * @brief Allocate a zero-initialized cubic grid.
 *
 * @param n_cells Cells per side; the grid holds n_cells^3 of them. Must be
 * non-zero.
 * @param box_length Physical side length of the box. Must be positive.
 * @return The grid, owned by the caller and released with sif_grid_free().
 * NULL on allocation failure or invalid geometry.
 */
SIF_NODISCARD sif_grid_t* sif_grid_alloc(uint32_t n_cells, sif_real box_length);

/**
 * @brief Release a grid and its cell array.
 * @param grid Grid to free. NULL is accepted and ignored.
 */
void sif_grid_free(sif_grid_t* grid);

/**
 * @brief Deposit a particle field onto the grid by Cloud-In-Cell assignment.
 *
 * Each particle contributes to the eight cells surrounding it, weighted by the
 * overlap of a cell-sized cube centred on the particle. Weights are used when
 * the field carries them, otherwise every particle counts as one.
 *
 * Every particle coordinate must lie in [0, box_length); the call is rejected
 * otherwise. sif_field_wrap_periodic() is the way to satisfy that for a
 * periodic snapshot whose coordinates have drifted onto the boundary.
 *
 * @param grid Destination grid, overwritten.
 * @param field Particle field to deposit. Read only.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument or a field with no
 * particles (one a consuming mesh has emptied, say); SIF_ERR_RANGE when any
 * particle lies outside the box, with the count in the log; SIF_ERR_ALLOC.
 * On any failure the grid is left exactly as it was -- it is not zeroed, and
 * its sif_grid_t::content does not change -- so check the status rather than
 * the cells.
 *
 * @note **This function can read and write the filesystem.** When the
 * `grid_cache_enabled` setting is on it looks for a previously computed grid
 * under the `cache_directory` setting (default `$HOME/.sif/grid_cache`) and
 * returns that instead of recomputing, writing its own result there otherwise.
 * The cache is off by default.
 *
 * @warning The cache key is not a content hash. It combines the grid and field
 * shape with a spot check of **five** particle positions, so two distinct
 * fields that agree on those would collide and the second would silently
 * receive the first's grid. It is meant for reuse within a run over known data,
 * not as a general-purpose content-addressed cache.
 */
SIF_NODISCARD int sif_grid_assign_cic(
  sif_grid_t* grid, const sif_field_t* field);

/**
 * @brief Convert cell weights in place to the density contrast.
 *
 * Replaces each cell value with delta = rho / rho_mean - 1, where rho_mean is
 * the mean over all cells. After this the field averages to zero and is bounded
 * below by -1.
 *
 * The mean is accumulated in double regardless of how sif_real is configured:
 * summing millions of cells in single precision loses the small contributions
 * to rounding, and the mean is a divisor for every cell that follows.
 *
 * @param grid Grid to convert, modified in place.
 * @return SIF_OK; SIF_ERR_INVALID for a missing or empty grid, or one that
 * already holds a density contrast (converting again would divide by a mean
 * of about zero); SIF_ERR_RANGE when the mean density is not positive -- a
 * grid nothing was deposited on, most often, or one holding a NaN. On any
 * failure the cells and sif_grid_t::content are left exactly as they were.
 */
SIF_NODISCARD int sif_grid_to_density_contrast(sif_grid_t* grid);

#endif /* SIF_STRUCTURES_GRID_H */
