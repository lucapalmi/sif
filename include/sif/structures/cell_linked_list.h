/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file cell_linked_list.h
 * @brief Spatial bin built from intrusive linked lists, one per cell.
 *
 * Where the chain mesh sorts a fixed set of particles into contiguous runs,
 * this holds items that arrive one at a time and whose count is not known in
 * advance -- voids as a finder accepts them, for instance. Insertion is O(1)
 * and never moves an existing entry, at the cost of a cell's members being
 * scattered through memory rather than contiguous.
 *
 * The lists are intrusive and array-based: #head gives the first item of a
 * cell and #next chains to the following one, both indexing the same item
 * space, so there are two allocations for the whole structure and no per-item
 * node.
 *
 * Walking one cell:
 *
 * @code
 * for (int32_t i = cll->head[cell]; i >= 0; i = cll->next[i]) {
 *   ...
 * }
 * @endcode
 */

#ifndef SIF_STRUCTURES_CELL_LINKED_LIST_H
#define SIF_STRUCTURES_CELL_LINKED_LIST_H

#include "sif/core/macros.h"
#include <stdint.h>

/**
 * @brief An array-based spatial linked list for fast neighbourhood lookups.
 *
 * @note Item indices are stored as int32_t, so the capacity is capped at
 * INT32_MAX entries. The sign is what makes the empty marker possible.
 */
typedef struct {
  uint32_t n_cells;         /**< Cells per side. */
  uint64_t total_cells;     /**< n_cells^3. */
  sif_real box_length;      /**< Physical side length of the box. */
  sif_real inv_cell_length; /**< n_cells / box_length, kept to avoid a divide
                             *   in the hot path. */

  /** First item of each cell, or -1 where the cell is empty. */
  int32_t* head;
  /** Next item in the same cell, or -1 at the end of the chain. Indexed by
   *  item, not by cell. */
  int32_t* next;
  /** Items #next has room for. */
  uint64_t capacity;

  /** SIF_PBC_PERIODIC wraps out-of-box coordinates, SIF_PBC_OPEN clamps them
   *  to the boundary cell. */
  uint8_t periodic;
} sif_cell_linked_list_t;

/**
 * @brief Allocate an empty cell linked list.
 *
 * @param n_cells Dimension of the coarse grid.
 * @param box_length Physical size of the simulation box.
 * @param initial_capacity Starting size for the item array, clamped up to 1.
 * @param opt Options. Honours SIF_PBC_PERIODIC (default) / SIF_PBC_OPEN, which
 * must match the convention used by whoever queries the list.
 * @return The list, owned by the caller and released with
 * sif_cell_linked_list_free(). NULL on invalid geometry, a capacity above
 * INT32_MAX, or allocation failure.
 */
SIF_NODISCARD sif_cell_linked_list_t* sif_cell_linked_list_alloc(
  uint32_t n_cells, sif_real box_length, uint64_t initial_capacity,
  sif_option opt);

/**
 * @brief Release a cell linked list.
 * @param cll List to free. NULL is accepted and ignored.
 */
void sif_cell_linked_list_free(sif_cell_linked_list_t* cll);

/**
 * @brief Grow the item array so it can hold at least @p required_capacity.
 *
 * Only #next is sized by the item count; the cell heads depend on the geometry
 * and never move. Existing chains are preserved, since they are indices rather
 * than pointers.
 *
 * @param cll The list.
 * @param required_capacity New minimum capacity.
 * @return SIF_OK on success, including when no growth was needed.
 * SIF_ERR_ALLOC if the larger buffer could not be allocated, in which case the
 * list keeps its current capacity and later inserts past it are rejected.
 * SIF_ERR_INVALID on a NULL list, SIF_ERR_RANGE if the request exceeds
 * INT32_MAX.
 */
int sif_cell_linked_list_ensure_capacity(
  sif_cell_linked_list_t* cll, uint64_t required_capacity);

/**
 * @brief Insert an item into the cell containing a point.
 *
 * The item is pushed onto the front of its cell's chain, so a walk of that cell
 * yields its members in reverse insertion order. Anything whose result depends
 * on the order in which a cell is visited has to sort, or be written not to
 * care.
 *
 * Coordinates outside the box are wrapped or clamped according to the boundary
 * convention the list was built with, rather than rejected.
 *
 * @param cll The list.
 * @param item_idx Identifier of the item, which must be below the current
 * capacity. This is the caller's own index; the list only stores it.
 * @param cx Point, x axis.
 * @param cy Point, y axis.
 * @param cz Point, z axis.
 * @return SIF_OK on success. SIF_ERR_RANGE if @p item_idx is past the current
 * capacity, in which case nothing is written. SIF_ERR_INVALID on a NULL list.
 *
 * @warning Not thread-safe: two inserts touching the same cell race on its
 * head.
 */
int sif_cell_linked_list_insert(sif_cell_linked_list_t* cll, uint64_t item_idx,
  sif_real cx, sif_real cy, sif_real cz);

#endif /* SIF_STRUCTURES_CELL_LINKED_LIST_H */
