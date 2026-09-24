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
  /** Cells per side. */
  uint32_t n_cells;
  /** n_cells^3. */
  uint64_t total_cells;
  /** Physical side length of the box. */
  sif_real box_length;
  /** Physical side length of one cell. */
  sif_real cell_length;

  /** Backing store for x/y/z. Owned; not for callers. */
  sif_real* _position_block;
  /** Backing store for vx/vy/vz. Owned; not for callers. */
  sif_real* _velocity_block;

  /** Positions, reordered so each cell is contiguous. */
  sif_real* x;
  sif_real* y;
  sif_real* z;

  /** Velocities in the same order, or NULL if the field carries none. */
  sif_real* vx;
  sif_real* vy;
  sif_real* vz;

  /** Weights in the same order, or NULL if the field carries none. */
  sif_real* weights;
  /**
   * Sum of #weights, or #n_particles when the mesh carries none -- either way,
   * what the box's mean density divides by. Summed once at construction, in
   * double whatever sif_real is, so a measurement normalized to the box mean
   * does not pay for a pass over the weights per call.
   */
  double total_weight;
  /**
   * Sum of #weights over each cell, #total_cells entries in cell order, or
   * NULL when the mesh carries no weights -- an unweighted mesh has its counts
   * in #cell_offsets already and pays nothing for this.
   *
   * What a weighted count of whole cells costs without it is a pass over every
   * tracer they hold, where the unweighted count is two offsets subtracted. A
   * sphere query touches mostly whole cells and reads tracers only in the ones
   * its surface cuts, so without this table weighting it would turn an
   * O(surface) query into an O(volume) one.
   *
   * Each cell is summed in double and stored as sif_real: a cell holds tens of
   * tracers, which a float sum represents well, and anything summing many
   * cells should accumulate them in double again.
   */
  sif_real* cell_weights;
  /**
   * Index of each particle in the source field: the only way to relate a
   * query result back to the field it came from, since the mesh reorders
   * particles.
   *
   * Always built, because the construction sorts on it (see
   * sif_chain_mesh_alloc()), but NULL afterwards when the mesh was built with
   * #SIF_MESH_DROP_INDICES.
   */
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
 * ask for.
 *
 * The map back to field indices is always *built*, and not because a caller
 * might want it: the scatter claims slots with an atomic, so which particle
 * lands where inside a cell is a race, and sorting each cell's run by source
 * index is what replaces that with an order determined by the input alone.
 * Without it two identical runs produce different meshes. It costs 8 bytes per
 * particle on top of the 12 or 24 the positions take, or 25 GiB at 3.4e9
 * tracers, so #SIF_MESH_DROP_INDICES releases it again once the sort is done --
 * same mesh, 8 bytes per particle lighter, at the price of the two
 * sif_chain_mesh_find_nearest_*() entry points.
 *
 * @param n_cells Cells per side.
 * @param box_length Physical side length of the box.
 * @param field Particle field to bin. Every coordinate must be in
 * [0, box_length); the call fails if any particle lies outside.
 * @param opt Honours #SIF_MESH_DROP_INDICES. #SIF_DEFAULT keeps everything.
 * @return The mesh, owned by the caller and released with
 * sif_chain_mesh_free(). NULL on invalid input or allocation failure.
 */
SIF_NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc(uint32_t n_cells,
  sif_real box_length, const sif_field_t* field, sif_option opt);

/**
 * @brief Build a mesh out of a field's own storage, emptying the field.
 *
 * Identical to sif_chain_mesh_alloc() in every observable way -- same binning,
 * same canonical order, same mesh -- except that it takes the field's payload
 * blocks over instead of copying them. sif_chain_mesh_alloc() has both the
 * field's positions and the mesh's alive at once, which at 3.4e9 tracers is
 * 81 GB to describe 40 GB of particles, and that duplication is the largest
 * single allocation in a typical run.
 *
 * Reordering in place still needs somewhere to shuffle through, but only one
 * array at a time rather than all of them: the cost falls from a full copy of
 * every payload to a single column, 4 bytes per particle instead of 12 or more.
 *
 * @param n_cells Cells per side.
 * @param box_length Physical side length of the box.
 * @param field Particle field to bin, **consumed**. On return it is a valid
 * but empty field -- no particles, no payloads -- which the caller still owns
 * and must still release with sif_field_free().
 * @param opt As sif_chain_mesh_alloc().
 * @return The mesh, owned by the caller and released with
 * sif_chain_mesh_free(). NULL on invalid input or allocation failure.
 *
 * @warning The field is emptied whether or not this succeeds. A failure part
 * way through has already taken the storage, and handing back a field pointing
 * at memory the mesh now owns would be worse than handing back an empty one.
 * Use sif_chain_mesh_alloc() where the field has to survive the call.
 */
SIF_NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc_consume(
  uint32_t n_cells, sif_real box_length, sif_field_t* field, sif_option opt);

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
 * @param mesh The mesh. Must carry sif_chain_mesh_t::original_indices, so not
 * one built with #SIF_MESH_DROP_INDICES.
 * @param px,py,pz Query point, in [0, box_length) on every axis.
 * @return Index into the original field, or UINT64_MAX if the mesh is empty or
 * cannot name its particles.
 */
uint64_t sif_chain_mesh_find_nearest_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

/**
 * @brief Index of the particle nearest a point, with periodic boundaries.
 *
 * As sif_chain_mesh_find_nearest_open(), except that separations are taken
 * through the nearest periodic image and the shell walk wraps at the faces.
 *
 * @param mesh The mesh. Must carry sif_chain_mesh_t::original_indices, so not
 * one built with #SIF_MESH_DROP_INDICES.
 * @param px,py,pz Query point, in [0, box_length) on every axis.
 * @return Index into the original field, or UINT64_MAX if the mesh is empty or
 * cannot name its particles.
 */
uint64_t sif_chain_mesh_find_nearest_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

#endif /* SIF_STRUCTURES_CHAIN_MESH_H */
