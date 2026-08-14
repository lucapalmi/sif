/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file profiles.h
 * @brief Stacked radial profiles of voids: density and radial velocity.
 *
 * Both estimators fill one row per void, binned in radius scaled by that
 * void's own radius, so profiles of different-sized voids are directly
 * stackable.
 *
 * The containers are only obtainable from an estimator: they mean nothing
 * until one has filled them, so there is no public allocator, only the frees.
 */

#ifndef SIF_MEASURE_PROFILES_H
#define SIF_MEASURE_PROFILES_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/field.h"
#include "sif/structures/tessellation.h"

/**
 * @brief Stacked radial density profiles, one row per void.
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  /** Outer edge of the profile, in units of each void's own radius. */
  sif_real ext;
  sif_real* r_edges;  /**< n_bins + 1 bin edges, shared by every row. */
  sif_real* profiles; /**< n_voids * n_bins values, row-major. */
} sif_density_profiles_t;

/**
 * @brief Stacked radial velocity profiles, one row per void.
 */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  /** Outer edge of the profile, in units of each void's own radius. */
  sif_real ext;
  sif_real* r_edges; /**< n_bins + 1 bin edges, shared by every row. */
  sif_real* v_rad;   /**< n_voids * n_bins values, row-major. */
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
 * @brief Stack radial profiles by binning the particles around each void.
 *
 * Bins tracers into spherical shells using a chain mesh sized from the mean
 * void radius. Densities come out normalized to the box mean, so a profile
 * approaches 1 far from the void centre.
 *
 * Either output may be omitted, and only what is asked for is computed --
 * velocities in particular cost a second payload in the mesh.
 *
 * Densities are cumulative -- each bin is the contrast enclosed within its
 * outer edge, which is what the spherical-evolution mapping expects -- while
 * velocities are differential, the mean radial velocity of that shell alone.
 *
 * A void with a non-positive radius is skipped and leaves a row of zeros;
 * there is no profile to measure around it.
 *
 * @param cat Voids to profile.
 * @param field Particle field. Must carry velocities if @p out_vel is wanted.
 * @param box_length Physical side length of the box. Must be positive.
 * @param ext Outer edge of the profile, in units of each void's radius. Must
 * be positive.
 * @param n_bins Radial bins per profile. Must be non-zero.
 * @param opt Honours SIF_PBC_PERIODIC / SIF_PBC_OPEN.
 * @param out_dens Address of a density set pointer, or NULL to skip. If it
 * points at NULL a set is allocated; otherwise the existing one is filled.
 * @param out_vel Address of a velocity set pointer, or NULL to skip. Same
 * convention.
 * @return SIF_OK, SIF_ERR_INVALID for a bad argument or a request for
 * velocities from a field that has none, or SIF_ERR_ALLOC.
 *
 * @note On any failure both outputs are left NULL, including a set this call
 * allocated before a later step failed. A caller may therefore check either
 * the status or the pointers.
 */
SIF_NODISCARD int sif_profiles_mesh(const sif_catalog_t* cat,
  const sif_field_t* field, sif_real box_length, sif_real ext, uint32_t n_bins,
  sif_option opt, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel);

/**
 * @brief Stack radial profiles from a Voronoi tessellation.
 *
 * @warning EXPERIMENTAL. This path sweeps a 100^3 voxel grid per void and runs
 * a nearest-neighbour query at every voxel, so it costs ~1e6 queries per void
 * and does not scale to production catalogues. It is kept for future work and
 * is not exercised by the test suite. Use sif_profiles_mesh() instead.
 *
 * Arguments, return value and failure behaviour are as sif_profiles_mesh(),
 * with the addition of @p tess, whose per-cell volumes weight each tracer's
 * contribution and which must cover the same field.
 */
SIF_NODISCARD int sif_profiles_voronoi(const sif_catalog_t* cat,
  const sif_field_t* field, const sif_tessellation_t* tess, sif_real box_length,
  sif_real ext, uint32_t n_bins, sif_option opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel);

#endif /* SIF_MEASURE_PROFILES_H */
