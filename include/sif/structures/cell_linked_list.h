#ifndef __SIF_CELL_LINKED_LIST_H__
#define __SIF_CELL_LINKED_LIST_H__

#include "sif/core/macros.h"
#include <stdint.h>

/*
 * @brief An array-based spatial linked list for fast neighborhood lookups
 *
 * @note Item indices are stored as int32_t, so the capacity is capped at
 * INT32_MAX entries.
 */
typedef struct {
  uint32_t n_cells;
  uint64_t total_cells;
  real_t box_length;
  real_t inv_cell_length;

  int32_t* head;
  int32_t* next;
  uint64_t capacity;

  /* SIF_PBC_PERIODIC wraps out-of-box coordinates, SIF_PBC_OPEN clamps them
   * to the boundary cell. */
  uint8_t periodic;
} sif_cell_linked_list_t;

/*
 * @brief Allocates a new cell linked list
 *
 * @param n_cells Dimension of the coarse grid
 * @param box_length Physical size of the simulation box
 * @param initial_capacity Starting size for the item array (clamped up to 1)
 * @param opt Options. Honors SIF_PBC_PERIODIC (default) / SIF_PBC_OPEN, which
 * must match the convention used by whoever queries the list.
 *
 * @return Pointer to the list, NULL on failure
 */
NODISCARD sif_cell_linked_list_t* sif_cell_linked_list_alloc(uint32_t n_cells,
  real_t box_length, uint64_t initial_capacity, sif_option_t opt);

/*
 * @brief Frees a cell linked list (NULL is a no-op)
 */
void sif_cell_linked_list_free(sif_cell_linked_list_t* cll);

/*
 * @brief Dynamically resizes the item array if needed
 *
 * @param required_capacity The new minimum capacity required
 *
 * @return SIF_OK on success, including when no growth was needed.
 * SIF_ERR_ALLOC if the larger buffer could not be allocated, in which case the
 * list keeps its current capacity and later inserts past it are rejected.
 * SIF_ERR_INVALID on a NULL list, SIF_ERR_RANGE if the request exceeds
 * INT32_MAX.
 */
int sif_cell_linked_list_ensure_capacity(
  sif_cell_linked_list_t* cll, uint64_t required_capacity);

/*
 * @brief Inserts an item into the correct spatial cell
 *
 * @param item_idx The ID of the item, must be < capacity
 * @param cx X-axis coordinate
 * @param cy Y-axis coordinate
 * @param cz Z-axis coordinate
 *
 * @return SIF_OK on success. SIF_ERR_RANGE if item_idx is past the current
 * capacity, in which case nothing is written. SIF_ERR_INVALID on a NULL list.
 */
int sif_cell_linked_list_insert(sif_cell_linked_list_t* cll, uint64_t item_idx,
  real_t cx, real_t cy, real_t cz);

#endif /* __SIF_CELL_LINKED_LIST_H__ */
