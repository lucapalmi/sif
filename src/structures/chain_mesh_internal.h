/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file chain_mesh_internal.h
 * @brief Nearest-neighbour queries in mesh order, for the library's own use.
 *
 * The public queries answer with a field index, which costs the caller a mesh
 * built with its index map intact and, at the read site, a random access per
 * answer. Code inside the library that only wants to reach the mesh's own
 * payload arrays -- the tessellation sampler is the case this exists for --
 * wants the slot instead: it indexes every payload directly, it is what the
 * volumes are ordered by, and it is available on a mesh built with
 * #SIF_MESH_DROP_INDICES.
 */

#ifndef SIF_STRUCTURES_CHAIN_MESH_INTERNAL_H
#define SIF_STRUCTURES_CHAIN_MESH_INTERNAL_H

#include "sif/structures/chain_mesh.h"

/**
 * @brief Slot of the particle nearest a point, with open boundaries.
 *
 * As sif_chain_mesh_find_nearest_open(), except that the answer indexes the
 * mesh's own arrays rather than the field the mesh was built from, so it needs
 * no index map.
 *
 * @return Slot in [0, n_particles), or UINT64_MAX if the mesh is empty.
 */
uint64_t sif__chain_mesh_find_nearest_slot_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

/**
 * @brief Slot of the particle nearest a point, with periodic boundaries.
 *
 * As sif__chain_mesh_find_nearest_slot_open(), with separations taken through
 * the nearest periodic image.
 */
uint64_t sif__chain_mesh_find_nearest_slot_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz);

#endif /* SIF_STRUCTURES_CHAIN_MESH_INTERNAL_H */
