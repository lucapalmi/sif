/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file profiles.h
 * @brief Stacked radial profiles of voids: density and radial velocity.
 *
 * The estimator fills one row per void, binned in radius scaled by that
 * void's own radius, so profiles of different-sized voids are directly
 * stackable.
 *
 * The containers are only obtainable from the estimator: they mean nothing
 * until it has filled them, so there is no public allocator, only the frees.
 */

#ifndef SIF_MEASURE_PROFILES_H
#define SIF_MEASURE_PROFILES_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"

/**
 * @brief Stacked radial density profiles, one row per void.
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  /** Outer edge of the profile, in units of each void's own radius. */
  sif_real ext;
  /**
   * True when a bin holds the contrast of that shell alone, false when it
   * holds the contrast enclosed within the bin's outer edge. Set by
   * #SIF_PROFILES_DIFFERENTIAL, and recorded because the two are not
   * distinguishable from the values.
   */
  bool differential;
  /** n_bins + 1 bin edges, shared by every row. */
  sif_real* r_edges;
  /** n_voids * n_bins values, row-major. */
  sif_real* profiles;
} sif_density_profiles_t;

/**
 * @brief Radius a density bin's value belongs at, in units of the void radius.
 *
 * Not the same edge for the two binnings, and plotting one at the other's
 * radius shifts the whole curve by half a bin: a cumulative bin holds
 * everything enclosed within its *outer* edge, while a differential bin is a
 * shell mean and belongs at its centre. The difference is easy to miss because
 * both are smooth and only the position of a feature moves -- a void found at
 * a fixed enclosed contrast puts that contrast exactly at r = R_v, and half a
 * bin is enough to hide it.
 *
 * @param profs The profile set.
 * @param bin Bin to place, in [0, n_bins).
 * @return The radius, in units of each void's own radius; multiply by a void's
 * radius for physical units.
 */
static inline sif_real sif_density_profiles_bin_radius(
  const sif_density_profiles_t* profs, uint32_t bin) {
  SIF_ASSERT(bin < profs->n_bins);

  return profs->differential
           ? (profs->r_edges[bin] + profs->r_edges[bin + 1]) * (sif_real)0.5
           : profs->r_edges[bin + 1];
}

/**
 * @brief Stacked radial velocity profiles, one row per void.
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  /** Outer edge of the profile, in units of each void's own radius. */
  sif_real ext;
  /** n_bins + 1 bin edges, shared by every row. */
  sif_real* r_edges;
  /** n_voids * n_bins values, row-major. */
  sif_real* v_rad;
} sif_velocity_profiles_t;

/**
 * @brief Release a density profile set.
 * @param profs Set to free. NULL is accepted and ignored.
 */
void sif_density_profiles_free(sif_density_profiles_t* profs);

/**
 * @brief Release a velocity profile set.
 * @param profs Set to free. NULL is accepted and ignored.
 */
void sif_velocity_profiles_free(sif_velocity_profiles_t* profs);

/**
 * @brief One void's density profile.
 * @param profs The profile set.
 * @param void_idx Void to read, indexing the catalogue the set was built from.
 * @return Borrowed pointer to that void's n_bins values.
 */
static inline const sif_real* sif_density_profiles_get(
  const sif_density_profiles_t* profs, uint64_t void_idx) {
  SIF_ASSERT(void_idx < profs->n_voids);
  return &profs->profiles[void_idx * profs->n_bins];
}

/**
 * @brief One void's radial velocity profile.
 * @param profs The profile set.
 * @param void_idx Void to read, indexing the catalogue the set was built from.
 * @return Borrowed pointer to that void's n_bins values.
 */
static inline const sif_real* sif_velocity_profiles_get(
  const sif_velocity_profiles_t* profs, uint64_t void_idx) {
  SIF_ASSERT(void_idx < profs->n_voids);
  return &profs->v_rad[void_idx * profs->n_bins];
}

/**
 * @brief How far a profile reaches when the caller does not say, in units of
 * each void's own radius.
 */
#define SIF_PROFILES_DEFAULT_EXT ((sif_real)5.0)

/**
 * @brief Stack radial profiles by binning the mesh's tracers around each void.
 *
 * Densities come out normalized to the box mean, so a profile approaches 1
 * far from the void centre.
 *
 * Either output may be omitted, and only what is asked for is computed --
 * velocities in particular are only available from a mesh that carries them.
 *
 * Densities are cumulative by default -- each bin is the contrast enclosed
 * within its outer edge, which is what the spherical-evolution mapping expects
 * -- and #SIF_PROFILES_DIFFERENTIAL makes each bin the contrast of its own
 * shell instead. Velocities are the mean radial velocity of a shell either
 * way.
 *
 * Which edge a bin's value belongs at follows from that, and differs between
 * the two: see sif_density_profiles_bin_radius(), which is what anything
 * plotting or fitting these should ask.
 *
 * A void with a non-positive radius is skipped and leaves a row of zeros;
 * there is no profile to measure around it.
 *
 * @param cat Voids to profile.
 * @param mesh Tracers to bin, and the box they live in: the mesh is where the
 * box length, the tracer count and the mean density all come from, so it has
 * to hold every tracer of the sample rather than a subset. A mesh of a
 * tessellation's samples is equally valid and is what makes the result
 * volume-weighted rather than tracer-weighted -- see
 * sif_chain_mesh_alloc_tessellation(), which needs no change here because the
 * samples carry the tracer weight between them. Built with
 * sif_profiles_suggest_mesh_cells() unless the caller has a mesh already --
 * one built for a finder does just as well, and reusing it is the point of
 * taking a mesh here rather than a field. #SIF_MESH_DROP_INDICES is fine:
 * this estimator walks cells and never names a tracer in field order.
 * @param ext Outer edge of the profile, in units of each void's radius.
 * Anything not positive selects #SIF_PROFILES_DEFAULT_EXT.
 * @param n_bins Radial bins per profile. Must be non-zero.
 * @param opt Honours SIF_PBC_PERIODIC / SIF_PBC_OPEN and
 * SIF_PROFILES_CUMULATIVE / SIF_PROFILES_DIFFERENTIAL.
 * @param out_dens Address of a density set pointer, or NULL to skip. If it
 * points at NULL a set is allocated; otherwise the existing one is filled.
 * @param out_vel Address of a velocity set pointer, or NULL to skip. Same
 * convention.
 * @return SIF_OK, SIF_ERR_INVALID for a bad argument or a request for
 * velocities from a mesh that has none, or SIF_ERR_ALLOC.
 *
 * @note On any failure both outputs are left NULL, including a set this call
 * allocated before a later step failed. A caller may therefore check either
 * the status or the pointers.
 */
SIF_NODISCARD int sif_profiles(const sif_catalog_t* cat,
  const sif_chain_mesh_t* mesh, sif_real ext, uint32_t n_bins, sif_option opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel);

/** @brief Tracers per mesh cell that sif_profiles() runs fastest at.
 *
 * The walk pays per cell it visits and per tracer it reads, and the two pull
 * opposite ways: coarser cells overshoot the sphere and read tracers that were
 * never going to be inside it, finer ones spend the saving on cell overhead --
 * which is mostly the cache miss on the cell table, not arithmetic. The
 * balance was measured across a factor of eight in tracer count and six in
 * void radius, and it sits here in every one of them; notably it does not move
 * with the void size, which is why this rule does not ask about it.
 *
 * Close enough to #SIF_FINDER_MESH_PARTICLES_PER_CELL that one mesh serves
 * both calls: either constant lands the other within a percent of its own
 * minimum.
 */
#define SIF_PROFILES_MESH_PARTICLES_PER_CELL 36.0

/** @brief Cap on the suggested resolution. The mesh's cell_offsets array alone
 * is 8 * n_cells^3 bytes, which is already ~130 MiB here.
 */
#define SIF_PROFILES_MESH_MAX_CELLS 256u

/**
 * @brief Suggested chain-mesh resolution for sif_profiles().
 *
 * Resolution changes only speed and memory -- the profiles are identical at
 * any of them -- so it is safe to tune, and a mesh built for something else is
 * always a valid input. It is worth tuning: the rule this replaces, one cell
 * per search radius, measured 2.5x slower than the minimum on an ordinary
 * catalogue and 5x on one of large voids.
 *
 * Unlike sif_finder_suggest_mesh_cells() there is no floor for the search
 * sphere having to fit inside the mesh. A profile sphere wider than the box is
 * capped at one full row of cells inside sif_profiles() instead, so a mesh too
 * coarse to hold it is slow, never wrong.
 *
 * The minimum is broad -- a factor of two either way costs on the order of
 * 10% -- so being approximate here is the point rather than a shortcoming.
 *
 * @param n_particles Tracers the mesh will hold.
 * @return n_cells to hand to sif_chain_mesh_alloc(), never zero.
 */
static inline uint32_t sif_profiles_suggest_mesh_cells(uint64_t n_particles) {
  if (n_particles == 0)
    return 1;

  const double n =
    cbrt((double)n_particles / SIF_PROFILES_MESH_PARTICLES_PER_CELL);

  if (n >= (double)SIF_PROFILES_MESH_MAX_CELLS)
    return SIF_PROFILES_MESH_MAX_CELLS;
  if (n >= 1.0)
    return (uint32_t)n;

  return 1;
}

#endif /* SIF_MEASURE_PROFILES_H */
