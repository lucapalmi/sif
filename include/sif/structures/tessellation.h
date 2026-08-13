/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file tessellation.h
 * @brief Per-particle volumes and adjacency, as a sparse graph.
 *
 * A Voronoi tessellation gives every tracer a volume, and so a density
 * estimate that adapts to the local sampling instead of to a fixed grid --
 * which is what a void finder wants, since voids are exactly where a fixed
 * grid has the fewest tracers per cell.
 *
 * The adjacency is stored in compressed sparse row form: the neighbours of
 * particle i occupy `neighbor_indices[neighbor_offsets[i] ..
 * neighbor_offsets[i + 1])`. One allocation for the whole graph, and a
 * particle's neighbours are contiguous.
 */

#ifndef SIF_STRUCTURES_TESSELLATION_H
#define SIF_STRUCTURES_TESSELLATION_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

/**
 * @brief Volumes and adjacency of a tessellation, in CSR form.
 */
typedef struct {
  uint64_t n_particles;
  uint64_t n_edges; /**< Entries in #neighbor_indices. */

  /** Volume assigned to each particle, n_particles entries. */
  sif_real* volumes;
  /** Where each particle's neighbour list starts, n_particles + 1 entries. */
  uint64_t* neighbor_offsets;
  /** Neighbour particle indices, #n_edges entries, grouped by owner. */
  uint64_t* neighbor_indices;
} sif_tessellation_t;

/**
 * @brief Build an approximate tessellation of a field.
 *
 * Approximate rather than exact: the volumes come from sampling instead of
 * from constructing the Voronoi cells, which trades a controllable error for a
 * cost that stays linear in the particle count. @p supersample_factor buys
 * accuracy with time and memory.
 *
 * @param field Particle field to tessellate. **Modified**: the extent is
 * read off the field, so its bounds are established if they are not
 * already valid.
 * @param supersample_factor Sample points generated per real particle. Higher
 * values reduce the volume error and raise the cost proportionally.
 * @param opt SIF_TESS_METHOD_RANDOM (default) or SIF_TESS_METHOD_VOXEL for how
 * the sample points are placed.
 * @return The tessellation, owned by the caller and released with
 * sif_tessellation_free(). NULL on invalid input or allocation failure.
 */
SIF_NODISCARD sif_tessellation_t* sif_tessellation_approx(
  sif_field_t* field, uint32_t supersample_factor, sif_option opt);

/**
 * @brief Release a tessellation and its graph.
 * @param tess Tessellation to free. NULL is accepted and ignored.
 */
void sif_tessellation_free(sif_tessellation_t* tess);

#endif /* SIF_STRUCTURES_TESSELLATION_H */
