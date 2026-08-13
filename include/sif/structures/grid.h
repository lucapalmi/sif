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
 * of a mass grid. The grid therefore records which it is.
 */
typedef enum {
  SIF_GRID_EMPTY = 0,       /**< Allocated and zeroed; nothing deposited. */
  SIF_GRID_MASS,            /**< Mass, or tracer count, per cell. */
  SIF_GRID_DENSITY_CONTRAST /**< delta = rho / rho_mean - 1. */
} sif_grid_content_t;

/**
 * @brief A cubic grid of `n_cells`^3 cells over a periodic box.
 */
typedef struct {
  /**
   * Cell values, flat and row-major with z varying fastest.
   *
   * What they mean is recorded in #content, and changes as the pipeline
   * proceeds: mass per cell after sif_grid_assign_cic(), the density contrast
   * after sif_grid_to_density_contrast() has overwritten them in place.
   */
  sif_real* values;

  /** What #values currently holds. */
  sif_grid_content_t content;

  uint32_t n_cells;     /**< Cells per side. */
  uint64_t total_cells; /**< n_cells^3. */
  sif_real box_length;  /**< Physical side length of the box. */
  sif_real cell_length; /**< Physical side length of one cell. */

  /**
   * Fast path for the flat index when #n_cells is a power of two: an index can
   * then be masked and shifted instead of multiplied and divided. Both are 0
   * when #n_cells is not a power of two, which is the signal to take the
   * general path.
   */
  uint32_t p2_mask;  /**< n_cells - 1, or 0. */
  uint32_t p2_shift; /**< log2(n_cells), or 0. */
} sif_grid_t;

/**
 * @brief Allocate a zero-initialized cubic grid.
 *
 * @param n_cells Cells per side; the grid holds n_cells^3 of them.
 * @param box_length Physical side length of the box.
 * @return The grid, owned by the caller and released with sif_grid_free().
 * NULL on allocation failure.
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
 * overlap of a cell-sized cube centred on the particle. Masses are used when
 * the field carries them, otherwise every particle counts as one.
 *
 * Every particle coordinate must lie in [0, box_length); the call is rejected
 * otherwise. sif_field_wrap_periodic() is the way to satisfy that for a
 * periodic snapshot whose coordinates have drifted onto the boundary.
 *
 * @param grid Destination grid, overwritten.
 * @param field Particle field to deposit. Read only.
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
void sif_grid_assign_cic(sif_grid_t* grid, const sif_field_t* field);

/**
 * @brief Convert cell masses in place to the density contrast.
 *
 * Replaces each cell value with delta = rho / rho_mean - 1, where rho_mean is
 * the mean over all cells. After this the field averages to zero and is bounded
 * below by -1.
 *
 * The mean is accumulated in double regardless of how sif_real is configured:
 * summing millions of cells in single precision loses the small contributions
 * to rounding, and the mean is a divisor for every cell that follows.
 *
 * @param grid Grid to convert, modified in place. Logs and returns without
 * touching the data if the mean density is not positive.
 */
void sif_grid_to_density_contrast(sif_grid_t* grid);

#endif /* SIF_STRUCTURES_GRID_H */
