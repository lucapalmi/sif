/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file octree.h
 * @brief Adaptive octree over a particle field, for neighbour queries on
 * clustered data.
 *
 * Where the chain mesh divides space uniformly, this subdivides only where
 * particles are, so a query cost tracks the local density instead of the box
 * volume. That is what makes it the right structure for a field with voids and
 * filaments and the wrong one for a nearly uniform field, where the mesh is
 * cheaper and simpler.
 *
 * Nodes carry a contiguous particle range rather than a list of members, which
 * is possible only because the field is Morton-sorted: Morton order *is*
 * depth-first octree order, so every subtree occupies one slice of the arrays.
 * The tree is therefore an index into the field, and both must be passed to
 * every query.
 */

#ifndef SIF_STRUCTURES_OCTREE_H
#define SIF_STRUCTURES_OCTREE_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/field.h"

/**
 * @brief One node: either a leaf, or an internal node with eight children.
 *
 * The eight children of a node are consecutive, so one index locates all of
 * them and the whole tree is a flat array rather than a web of pointers.
 */
typedef struct {
  /** Index of the first of eight consecutive children, or UINT32_MAX for a
   *  leaf. */
  uint32_t first_child;
  /** First particle of this node's range, indexing the Morton-sorted field. */
  uint32_t p_start;
  /** Particles in the range. */
  uint32_t p_counting;
  /** Unused; keeps the node at 16 bytes so four fit in a cache line. */
  uint32_t padding;
} sif_octree_node_t;

/** @brief An octree over one field. */
typedef struct {
  sif_octree_node_t* nodes; /**< Flat node array; the root is nodes[0]. */
  uint32_t capacity;        /**< Nodes allocated. */
  uint32_t count;           /**< Nodes in use. */

  sif_real root_center[3]; /**< Centre of the root cube, from the field. */
  sif_real root_half_span; /**< Half-side of the root cube. */
} sif_octree_t;

/**
 * @brief Build an octree over a field.
 *
 * Subdivides until every leaf holds at most @p max_per_leaf particles, or until
 * the Morton resolution runs out -- coincident particles cannot be separated
 * and stop the recursion early, so a leaf may exceed the threshold.
 *
 * @param field Field to index. **Modified**: the tree requires Morton order and
 * valid bounds, so both are established on the field if they are not already
 * present, which physically reorders its arrays.
 * @param max_per_leaf Particles a leaf may hold before it splits. Larger values
 * make a shallower tree that scans more particles per leaf; the useful range is
 * tens.
 * @return The tree, owned by the caller and released with sif_octree_free().
 * NULL on invalid input or allocation failure.
 *
 * @warning The tree indexes the field by position and stays valid only as long
 * as the field is not reordered or resized under it. Anything that re-sorts or
 * re-assigns the field invalidates the tree.
 */
SIF_NODISCARD sif_octree_t* sif_octree_alloc(
  sif_field_t* field, uint32_t max_per_leaf);

/**
 * @brief Release an octree.
 * @param tree Tree to free. NULL is accepted and ignored. Does not free the
 * field.
 */
void sif_octree_free(sif_octree_t* tree);

/**
 * @brief Index of the particle nearest a point.
 *
 * Descends to the point's own leaf first, then unwinds, skipping any subtree
 * whose cube is farther than the best candidate so far.
 *
 * @param tree The tree.
 * @param field The field it was built over.
 * @param px Query point, x axis.
 * @param py Query point, y axis.
 * @param pz Query point, z axis.
 * @return Index into the Morton-sorted field. Use
 * sif_field_t::original_indices to map it back to the caller's input order.
 *
 * @note Open boundaries: separations are taken directly, not through periodic
 * images. Use sif_chain_mesh_find_nearest_pbc() where the box wraps.
 */
uint64_t sif_octree_find_nearest(const sif_octree_t* tree,
  const sif_field_t* field, sif_real px, sif_real py, sif_real pz);

/**
 * @brief Every particle within a radius of a point.
 *
 * @param tree The tree.
 * @param field The field it was built over.
 * @param px Centre, x axis.
 * @param py Centre, y axis.
 * @param pz Centre, z axis.
 * @param radius Search radius.
 * @param out_indices Written with the indices found, into the Morton-sorted
 * field.
 * @param max_capacity Entries @p out_indices can hold.
 * @return The number of particles found, which may exceed @p max_capacity when
 * the output was truncated -- compare the two to detect it, and note that only
 * the first @p max_capacity indices were written.
 */
uint64_t sif_octree_search_radius(const sif_octree_t* tree,
  const sif_field_t* field, sif_real px, sif_real py, sif_real pz,
  sif_real radius, uint64_t* out_indices, uint64_t max_capacity);

/**
 * @brief Every particle inside an axis-aligned box.
 *
 * @param tree The tree.
 * @param field The field it was built over.
 * @param min_x Lower bound, x axis.
 * @param min_y Lower bound, y axis.
 * @param min_z Lower bound, z axis.
 * @param max_x Upper bound, x axis.
 * @param max_y Upper bound, y axis.
 * @param max_z Upper bound, z axis.
 * @param out_indices Written with the indices found, into the Morton-sorted
 * field.
 * @param max_capacity Entries @p out_indices can hold.
 * @return The number of particles found; see sif_octree_search_radius() on
 * truncation.
 */
uint64_t sif_octree_search_box(const sif_octree_t* tree,
  const sif_field_t* field, sif_real min_x, sif_real min_y, sif_real min_z,
  sif_real max_x, sif_real max_y, sif_real max_z, uint64_t* out_indices,
  uint64_t max_capacity);

#endif /* SIF_STRUCTURES_OCTREE_H */
