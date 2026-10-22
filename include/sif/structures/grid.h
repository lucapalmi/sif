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
 * with sif_grid_assign() (or sif_grid_assign_cic()), then convert the result
 * in place to the density contrast with sif_grid_to_density_contrast().
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
 * diagnostic -- normalising an already-normalised field, or measuring moments
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
   * sif_grid_to_density_contrast(), which normalises by the mean, and matters
   * only to code reading these values directly.
   */
  SIF_GRID_DENSITY,
  /** delta = rho / rho_mean - 1. */
  SIF_GRID_DENSITY_CONTRAST
} sif_grid_content_t;

/**
 * @brief How particles were deposited onto a grid: the mass-assignment window.
 *
 * Each scheme spreads a particle over p cells per axis with a kernel whose
 * transform is sinc^p(k_i l / 2) per axis, and the enumerator's value is that
 * p. Higher orders are smoother in real space and alias less -- the images
 * above the grid's Nyquist frequency fold back weighted by sinc^p -- at the
 * price of p^3 cells touched per particle.
 *
 * A filter applied later has to divide this window back out, and the finders
 * and the delta statistics do, reading which window from
 * sif_grid_t::assignment. UNKNOWN is a grid filled some other way -- by hand,
 * or read from a file written before the scheme was recorded -- and is treated
 * as CIC, which is what every grid was before the choice existed.
 */
typedef enum {
  SIF_ASSIGNMENT_UNKNOWN = 0,
  SIF_ASSIGNMENT_NGP = 1, /**< Nearest grid point: 1 cell. */
  SIF_ASSIGNMENT_CIC = 2, /**< Cloud in cell: 2 cells, trilinear. */
  SIF_ASSIGNMENT_TSC = 3, /**< Triangular-shaped cloud: 3 cells. */
  SIF_ASSIGNMENT_PCS = 4  /**< Piecewise cubic spline: 4 cells. */
} sif_assignment_t;

/**
 * Deposit an interlaced grid: a second grid with every particle moved by half
 * a cell along each axis, combined with the first in Fourier space with the
 * phase that undoes the shift. The aliased images whose index sum is odd
 * arrive in the two grids with opposite signs and cancel, which removes the
 * leading aliasing (Hockney & Eastwood 1981; Sefusatti et al. 2016). Costs a
 * second deposit and two FFTs, once.
 */
#define SIF_GRID_INTERLACE (1u << 8)

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

  /**
   * The window the cells were deposited with, set by sif_grid_assign() and
   * read by everything that deconvolves it. #SIF_ASSIGNMENT_UNKNOWN for a grid
   * filled any other way.
   */
  sif_assignment_t assignment;

  /** 1 if deposited with #SIF_GRID_INTERLACE. */
  uint32_t interlaced;

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
 * @brief Allocate a zero-initialised cubic grid.
 *
 * @param n_cells Cells per side; the grid holds n_cells^3 of them. Must be
 * non-zero.
 * @param box_length Physical side length of the box. Must be positive.
 * @return The grid, owned by the caller and released with sif_grid_free().
 * NULL on invalid geometry or allocation failure, with sif_error_status() at
 * SIF_ERR_INVALID or SIF_ERR_ALLOC.
 */
SIF_NODISCARD sif_grid_t* sif_grid_alloc(uint32_t n_cells, sif_real box_length);

/**
 * @brief Release a grid and its cell array.
 * @param grid Grid to free. NULL is accepted and ignored.
 */
void sif_grid_free(sif_grid_t* grid);

/**
 * @brief Deposit a particle field onto the grid with a chosen assignment
 * scheme, optionally interlaced.
 *
 * As sif_grid_assign_cic() in everything but the window: the same box check,
 * the same density per unit volume on the way out, the same independence of
 * the thread count, and the same grid cache, keyed on the scheme too. The
 * scheme and the interlacing are recorded in the grid, so that a finder or a
 * delta statistic divides out the window the grid actually has. CIC without
 * interlacing is sif_grid_assign_cic(), bit for bit.
 *
 * #SIF_GRID_INTERLACE needs sif_system_init(), for the FFTs. It works with
 * three buffers the size of the grid's spectrum rather than one, so at the
 * largest grids it is the step that sets the memory needed.
 *
 * @param grid Destination grid, overwritten. At least 4 cells per side.
 * @param field Particle field to deposit. Read only.
 * @param scheme One of NGP, CIC, TSC, PCS.
 * @param opt #SIF_GRID_INTERLACE, or 0.
 * @return As sif_grid_assign_cic(), plus SIF_ERR_INVALID for an unknown scheme
 * or a grid smaller than 4 cells, and SIF_ERR_UNINITIALISED for an interlaced
 * deposit without sif_system_init(). On failure the grid is left as it was --
 * except that an interlaced deposit failing after its first transform leaves
 * it zeroed and empty, since its cells were already overwritten by then.
 */
SIF_NODISCARD int sif_grid_assign(sif_grid_t* grid, const sif_field_t* field,
  sif_assignment_t scheme, sif_option opt);

/**
 * @brief Deposit a particle field onto the grid by Cloud-In-Cell assignment.
 *
 * Each particle contributes to the eight cells surrounding it, weighted by the
 * overlap of a cell-sized cube centred on the particle. Weights are used when
 * the field carries them, otherwise every particle counts as one.
 *
 * Every particle coordinate must lie in [0, box_length); the call is rejected
 * otherwise. A coordinate within a few ULPs of a face -- stored as exactly
 * box_length by single-precision rounding, say -- is accepted and deposited
 * as folded periodically; the field itself is not changed. For a periodic
 * snapshot that was never wrapped into the box, sif_field_wrap_periodic().
 *
 * @param grid Destination grid, overwritten.
 * @param field Particle field to deposit. Read only.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, a grid with no cell
 * values, or a field with no particles (one a consuming mesh has emptied,
 * say), no positions, or sky coordinates; SIF_ERR_RANGE when any particle lies
 * outside the box, with the count and the first one in the message;
 * SIF_ERR_ALLOC.
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
