/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file chain_mesh.h
 * @brief Uniform spatial bin over a periodic box, for neighbour queries.
 *
 * Particles are copied into cell order at construction, so the members of a
 * cell occupy one contiguous run of the payload arrays and a query that visits
 * a cell reads straight through memory. That copy is the whole point: an index
 * into the original field would cost a random access per particle, which at
 * these sizes dominates the distance arithmetic.
 *
 * Cell membership is found by truncation, so the mesh is only useful when the
 * cells are comparable to the search radius -- too few and every query scans
 * the box, too many and the shell walk visits mostly empty cells.
 */

#ifndef SIF_STRUCTURES_CHAIN_MESH_H
#define SIF_STRUCTURES_CHAIN_MESH_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief A 3D cubic chain mesh.
 *
 * @note The mesh is anchored at the origin: all particle coordinates, and all
 * query coordinates, must lie in [0, box_length) on every axis. This is
 * enforced at construction time. Callers working in a shifted frame have to
 * translate into box-local coordinates first.
 */
typedef struct {
  uint32_t n_cells;     /**< Cells per side. */
  uint64_t total_cells; /**< n_cells^3. */
  sif_real box_length;  /**< Physical side length of the box. */
  sif_real cell_length; /**< Physical side length of one cell. */

  /** Backing store for x/y/z. Owned; not for callers. */
  sif_real* _position_block;
  /** Backing store for vx/vy/vz. Owned; not for callers. */
  sif_real* _velocity_block;

  sif_real* x; /**< Positions, reordered so each cell is contiguous. */
  sif_real* y;
  sif_real* z;

  /** Velocities in the same order, or NULL if the field carries none. */
  sif_real* vx;
  sif_real* vy;
  sif_real* vz;

  /** Weights in the same order, or NULL if the field carries none. */
  sif_real* weights;
  /** Index of each particle in the source field. Always present: the mesh
   *  reorders particles, so this is the only way to relate a query result
   *  back to the field it came from. */
  uint64_t* original_indices;

  uint64_t n_particles;
  /**
   * Where each cell's run begins, with #total_cells + 1 entries: cell `c`
   * occupies `[cell_offsets[c], cell_offsets[c + 1])`, so an empty cell is one
   * whose two offsets are equal and no separate count is needed.
   */
  uint64_t* cell_offsets;
} sif_chain_mesh_t;

/**
 * @brief Allocate a chain mesh and bin a field into it.
 *
 * The mesh mirrors the field: velocities and weights are copied if the field
 * carries them and not otherwise, since there is nothing else a caller could
 * ask for. The map back to field indices is always built -- a mesh that cannot
 * say which particle an answer refers to cannot answer the queries this
 * structure exists for -- which costs 8 bytes per particle on top of the 12 or
 * 24 the positions take, or 25 GiB at 3.4e9 tracers.
 *
 * @param n_cells Cells per side.
 * @param box_length Physical side length of the box.
 * @param field Particle field to bin. Every coordinate must be in
 * [0, box_length); the call fails if any particle lies outside.
 * @return The mesh, owned by the caller and released with
 * sif_chain_mesh_free(). NULL on invalid input or allocation failure.
 */
SIF_NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc(
  uint32_t n_cells, sif_real box_length, const sif_field_t* field);

/**
 * @brief Release a chain mesh and everything it owns.
 * @param mesh Mesh to free. NULL is accepted and ignored.
 */
void sif_chain_mesh_free(sif_chain_mesh_t* mesh);

/**
 * @brief Index of the particle nearest a point, with open boundaries.
 *
 * Walks cell shells outward from the query point and stops once the nearest
 * candidate found is closer than the nearest possible point of the next shell,
 * so the answer is exact rather than restricted to the starting cell.
 *
 * @param mesh The mesh.
 * @param px,py,pz Query point, in [0, box_length) on every axis.
 * @return Index into the original field, or UINT64_MAX if the mesh is empty.
 */
uint64_t sif_chain_mesh_find_nearest_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

/**
 * @brief Index of the particle nearest a point, with periodic boundaries.
 *
 * As sif_chain_mesh_find_nearest_open(), except that separations are taken
 * through the nearest periodic image and the shell walk wraps at the faces.
 *
 * @param mesh The mesh.
 * @param px,py,pz Query point, in [0, box_length) on every axis.
 * @return Index into the original field, or UINT64_MAX if the mesh is empty.
 */
uint64_t sif_chain_mesh_find_nearest_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

#endif /* SIF_STRUCTURES_CHAIN_MESH_H */
