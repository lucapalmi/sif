/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/cell_linked_list.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Item ids are stored as int32_t (-1 marks "empty"), so this is the hard cap.
 *
 * Signed and 32-bit on purpose. The structure is two arrays of one id each per
 * cell and per item, so the width is half the memory of the whole index; and
 * a signed sentinel means "empty" costs no extra field and no reserved
 * positive value that a real id might collide with. The price is this cap,
 * which a field of more than 2 G items has to reach for the chain mesh
 * instead.
 */
#define CLL_MAX_CAPACITY ((uint64_t)INT32_MAX)

/*
 * Maps one physical coordinate onto a cell index, honoring the boundary
 * convention the list was created with.
 *
 * The two conventions differ in what they do with a coordinate outside the
 * box, and both answers are deliberate: periodic wraps it, because the box has
 * no outside; open clamps it into the edge cell, because a particle just past
 * the boundary is still nearest to what is just inside. Neither rejects,
 * since a caller inserting a stray coordinate wants it indexed somewhere
 * rather than dropped without trace.
 */
static inline uint32_t cll_axis_index(
  const sif_cell_linked_list_t* cll, sif_real c) {

  int32_t i = (int32_t)SIF_REAL_FLOOR(c * cll->inv_cell_length);
  const int32_t n = (int32_t)cll->n_cells;

  if (cll->periodic) {
    i %= n;
    if (i < 0)
      i += n;
    return (uint32_t)i;
  }

  if (i < 0)
    return 0;
  if (i >= n)
    return (uint32_t)(n - 1);
  return (uint32_t)i;
}

sif_cell_linked_list_t* sif_cell_linked_list_alloc(uint32_t n_cells,
  sif_real box_length, uint64_t initial_capacity, sif_option opt) {

  if (n_cells == 0 || !(box_length > 0.0f)) {
    SIF_LOG_ERROR("cell_linked_list",
      "invalid geometry (n_cells=%u, box_length=%g)", n_cells,
      (double)box_length);
    return NULL;
  }

  if (initial_capacity == 0)
    initial_capacity = 1;

  if (initial_capacity > CLL_MAX_CAPACITY) {
    SIF_LOG_ERROR("cell_linked_list",
      "requested capacity %" PRIu64 " exceeds the int32 item limit %" PRIu64,
      initial_capacity, CLL_MAX_CAPACITY);
    return NULL;
  }

  sif_cell_linked_list_t* cll = malloc(sizeof(sif_cell_linked_list_t));
  if (!cll) {
    SIF_LOG_ERROR("cell_linked_list",
      "failed to allocate linked list (%zu bytes)",
      sizeof(sif_cell_linked_list_t));
    return NULL;
  }

  cll->n_cells = n_cells;
  cll->total_cells = (uint64_t)n_cells * n_cells * n_cells;
  cll->box_length = box_length;
  cll->inv_cell_length = (sif_real)n_cells / box_length;
  cll->capacity = initial_capacity;
  cll->periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC) ? 1u : 0u;

  cll->head = sif_malloc_aligned(cll->total_cells * sizeof(int32_t));
  cll->next = sif_malloc_aligned(cll->capacity * sizeof(int32_t));

  if (!cll->head || !cll->next) {
    SIF_LOG_ERROR("cell_linked_list",
      "failed to allocate head/next arrays (total: %" PRIu64 " bytes)",
      (cll->total_cells + cll->capacity) * sizeof(int32_t));
    sif_free_aligned(cll->head);
    sif_free_aligned(cll->next);
    free(cll);
    return NULL;
  }

  /* 0xFF bytes is -1 in two's complement, which is the empty marker: memset
   * fills the whole head array in one pass where a loop would write one cell
   * at a time, and total_cells is n^3.
   *
   * Only `head` needs it. `next` is written by insert before anything can
   * follow the chain into it, so pre-filling it would be work whose result is
   * always overwritten. */
  memset(cll->head, 0xFF, cll->total_cells * sizeof(int32_t));

  SIF_LOG_TRACE("cell_linked_list",
    "initialized with capacity %" PRIu64 " (%s boundaries)", cll->capacity,
    cll->periodic ? "periodic" : "open");

  return cll;
}

void sif_cell_linked_list_free(sif_cell_linked_list_t* cll) {
  if (!cll)
    return;

  sif_free_aligned(cll->head);
  sif_free_aligned(cll->next);
  free(cll);
}

int sif_cell_linked_list_ensure_capacity(
  sif_cell_linked_list_t* cll, uint64_t required_capacity) {

  if (!cll)
    return SIF_ERR_INVALID;

  if (required_capacity <= cll->capacity)
    return SIF_OK;

  if (required_capacity > CLL_MAX_CAPACITY) {
    SIF_LOG_ERROR("cell_linked_list",
      "required capacity %" PRIu64 " exceeds the int32 item limit %" PRIu64,
      required_capacity, CLL_MAX_CAPACITY);
    return SIF_ERR_RANGE;
  }

  /* Doubling first, then the request, so a caller inserting one item at a time
   * pays a copy per doubling rather than per insert. The clamp is reachable
   * only from the doubling, since the request was bounded above. */
  uint64_t new_capacity = cll->capacity << 1;
  if (new_capacity < required_capacity)
    new_capacity = required_capacity;
  if (new_capacity > CLL_MAX_CAPACITY)
    new_capacity = CLL_MAX_CAPACITY;

  /* Only `next` grows. `head` is sized by the cell count, which is geometry
   * and does not change with the number of items in the list. */
  int32_t* new_next = sif_malloc_aligned(new_capacity * sizeof(int32_t));
  if (!new_next) {
    SIF_LOG_ERROR("cell_linked_list",
      "failed to reallocate next array (%" PRIu64 " bytes)",
      new_capacity * sizeof(int32_t));
    return SIF_ERR_ALLOC;
  }

  memcpy(new_next, cll->next, cll->capacity * sizeof(int32_t));
  sif_free_aligned(cll->next);

  cll->next = new_next;
  cll->capacity = new_capacity;

  SIF_LOG_TRACE(
    "cell_linked_list", "capacity expanded to %" PRIu64, cll->capacity);

  return SIF_OK;
}

int sif_cell_linked_list_insert(sif_cell_linked_list_t* cll, uint64_t item_idx,
  sif_real cx, sif_real cy, sif_real cz) {

  if (!cll)
    return SIF_ERR_INVALID;

  if (item_idx >= cll->capacity) {
    SIF_LOG_ERROR("cell_linked_list",
      "item index %" PRIu64 " is past capacity %" PRIu64, item_idx,
      cll->capacity);
    return SIF_ERR_RANGE;
  }

  const uint32_t ix = cll_axis_index(cll, cx);
  const uint32_t iy = cll_axis_index(cll, cy);
  const uint32_t iz = cll_axis_index(cll, cz);

  const uint64_t flat_idx = (uint64_t)ix * cll->n_cells * cll->n_cells +
                            (uint64_t)iy * cll->n_cells + (uint64_t)iz;

  /* Push onto the front of the cell's chain: the new item inherits whatever
   * head pointed at, and head points at it. O(1) with no tail pointer, at the
   * price of a chain in reverse insertion order -- which nothing depends on,
   * since a cell's contents are a set. These two lines are also why the
   * header calls insertion thread-unsafe. */
  cll->next[item_idx] = cll->head[flat_idx];
  cll->head[flat_idx] = (int32_t)item_idx;

  return SIF_OK;
}
