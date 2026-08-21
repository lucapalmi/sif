/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file utils.h
 * @brief Machinery shared by the void finders. Private to the library.
 *
 * The finders differ in how they fix a void's radius, not in how they find
 * candidates, mask accepted volume or test overlap. Those parts live here so
 * that the two do not drift: a change to what counts as an overlap has to
 * apply to both, or the catalogues stop being comparable.
 */

#ifndef SIF__FINDER_UTILS_H
#define SIF__FINDER_UTILS_H

#include "sif/structures/bitmask.h"
#include "sif/structures/catalog.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/grid.h"

#include "math/fft.h"

/* --- index helpers --- */

/**
 * @brief Wrap a possibly out-of-range cell index into [0, n).
 * @param mask n - 1 when n is a power of two, else 0 for the general path.
 */
static inline uint32_t sif__wrap_pbc(int32_t val, uint32_t n, uint32_t mask) {
  if (mask)
    return (uint32_t)val & mask;
  int32_t w = val % (int32_t)n;
  return (uint32_t)(w < 0 ? w + n : w);
}

/** @brief val % n, using the mask when n is a power of two. */
static inline uint32_t sif__fast_mod(uint64_t val, uint32_t n, uint32_t mask) {
  return mask ? (uint32_t)(val & mask) : (uint32_t)(val % n);
}

/** @brief val / n, using the shift when n is a power of two. */
static inline uint32_t sif__fast_div(uint64_t val, uint32_t n, uint32_t shift) {
  return shift ? (uint32_t)(val >> shift) : (uint32_t)(val / n);
}

/** @brief Flat row-major index of cell (ix, iy, iz) in an n^3 grid. */
uint64_t sif__flat_index(uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz);

/**
 * @brief Split a flat grid index back into its three cell coordinates.
 */
static inline void sif__unflatten_index(const sif_grid_t* grid, uint64_t flat,
  uint32_t* ix, uint32_t* iy, uint32_t* iz) {

  const uint32_t n = grid->n_cells;
  const uint32_t p2_mask = grid->p2_mask;
  const uint32_t p2_shift = grid->p2_shift;

  *iz = sif__fast_mod(flat, n, p2_mask);
  *iy = sif__fast_mod(sif__fast_div(flat, n, p2_shift), n, p2_mask);
  *ix = sif__fast_div(flat, (uint64_t)n * n, p2_shift ? 2 * p2_shift : 0);
}

/* --- field preparation, shared by every finder --- */

/**
 * @brief Arrange for the CIC assignment window to come back out of every
 * smoothed field, so the top-hat is the only window a finder ever sees.
 *
 * A finder that skipped this would be smoothing an already-smoothed field: the
 * effective window is wider than the radius it names, with an edge about a
 * cell thick, and in a void that reads as a contrast biased high -- fewer
 * candidates at every rung, and for exodus a rescaled radius biased large.
 * Both finders call this, so a catalogue from one stays comparable with a
 * catalogue from the other.
 *
 * The correction rides on the filter rather than on the stored spectrum, which
 * is what lets a finder still hand the caller's grid back untouched at the
 * end.
 *
 * The window's reciprocal grows without bound towards the Nyquist corner and
 * is held in check only by the top-hat carrying it. A radius of at least two
 * cells does that; below it the smallest scales are amplified instead, so this
 * warns rather than silently handing back a noisier field than it was given.
 *
 * @param tag Subsystem name for the log messages.
 * @param ws Workspace holding the forward transform.
 * @param grid The grid the transform came from, read for its cell size.
 * @param min_radius Smallest smoothing radius the run will use.
 * @param opt Honours #SIF_FINDER_KEEP_CIC_WINDOW, under which this is a no-op.
 *
 * @return SIF_OK, including when the flag skipped the work, or SIF_ERR_* if
 * the correction could not be set up.
 */
int sif__finder_deconvolve_cic(const char* tag, sif_fft_workspace_t* ws,
  const sif_grid_t* grid, sif_real min_radius, sif_option opt);

/* --- candidate scanning, shared by every finder --- */

/** @brief One candidate cell: its depth and where it sits in the grid. */
typedef struct {
  sif_real delta;
  uint64_t flat_idx;
} sif_candidate_t;

/**
 * @brief A reusable, self-growing list of candidate cells.
 *
 * Zero-initialize before first use, then hand the same buffer to every scan so
 * the allocation is amortized across radii.
 */
typedef struct {
  sif_candidate_t* items;
  uint64_t count;
  uint64_t capacity;
} sif_candidate_buffer_t;

/** @brief Release a candidate buffer's storage. */
void sif__candidate_buffer_free(sif_candidate_buffer_t* buf);

/**
 * @brief Collect every unmasked cell at or below @p threshold, sorted by
 * increasing delta so the deepest underdensity comes first.
 *
 * @return SIF_OK on success, where a count of 0 is a legitimate result, or
 * SIF_ERR_ALLOC if the buffer could not grow.
 */
int sif__finder_scan_candidates(const sif_grid_t* grid,
  const sif_bitmask_t* mask, sif_real threshold, sif_candidate_buffer_t* buf);

/* --- per-radius reporting --- */

/** @brief What happened to the candidates at one radius. */
typedef struct {
  uint64_t n_candidates;
  uint64_t rejected_masked;  /**< Already covered by an accepted void. */
  uint64_t rejected_proxy;   /**< Failed the cheap axis-pole overlap probe. */
  uint64_t rejected_overlap; /**< Failed the exact pairwise overlap test. */
  uint64_t rejected_rescale; /**< Radius rescaling did not converge. */

  /* The same two tests again, at the rescaled radius and against a catalog
   * that has moved on since. Only a finder that rescales fills these. */
  uint64_t rejected_proxy_recheck;
  uint64_t rejected_exact;

  uint64_t accepted;
} sif_finder_radius_stats_t;

/**
 * @brief Emit the per-radius TRACE breakdown and the INFO summary line.
 */
void sif__finder_log_radius(const char* tag, sif_real radius,
  const sif_finder_radius_stats_t* stats, uint64_t total_voids,
  double elapsed_s);

/* --- geometry --- */

/**
 * @brief Cheap overlap probe: are any of the sphere's axis poles already
 * masked?
 *
 * A one-sided pre-filter for sif__overlap_exact(): it reads the six bits
 * at the sphere's axis poles instead of walking the neighbouring voids. A hit
 * is conclusive and short-circuits the exact test; a miss is not, so a
 * candidate that passes here must still go through
 * sif__overlap_exact().
 *
 * @return Non-zero if the candidate certainly overlaps an accepted void.
 */
uint8_t sif__overlap_quick(const sif_bitmask_t* mask, uint32_t n_cells,
  uint32_t p2_mask, uint32_t ix, uint32_t iy, uint32_t iz, uint32_t r);

/**
 * @brief Exact overlap test against every void near the candidate.
 *
 * Two voids collide when their separation falls below
 * r1 + r2 - overlap_fraction * min(r1, r2), so @p overlap_fraction is measured
 * against the smaller of the pair.
 *
 * @return Non-zero if the candidate overlaps an accepted void.
 */
uint8_t sif__overlap_exact(const sif_catalog_t* cat, sif_real cx, sif_real cy,
  sif_real cz, sif_real r, sif_real max_r, sif_real box_length,
  const sif_cell_linked_list_t* cll, uint32_t p2_mask,
  sif_real overlap_fraction);

/**
 * @brief Mark every grid cell whose centre lies inside an accepted void.
 *
 * This is what stops a later, smaller radius from re-finding the same
 * underdensity.
 */
void sif__mark_sphere(sif_bitmask_t* mask, sif_real cx, sif_real cy,
  sif_real cz, sif_real r_true, uint32_t n_cells, uint32_t p2_mask,
  sif_real cell_length);

#endif /* SIF__FINDER_UTILS_H */
