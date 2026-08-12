#ifndef __SIF_FINDER_UTILS_H
#define __SIF_FINDER_UTILS_H

#include "sif/structures/bitmask.h"
#include "sif/structures/catalog.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/grid.h"

/* --- Index helpers --- */

static inline uint32_t sif_wrap_pbc(int32_t val, uint32_t n, uint32_t mask) {
  if (mask) return (uint32_t)val & mask;
  int32_t w = val % (int32_t)n;
  return (uint32_t)(w < 0 ? w + n : w);
}

static inline uint32_t sif_fast_mod(uint64_t val, uint32_t n, uint32_t mask) {
  return mask ? (uint32_t)(val & mask) : (uint32_t)(val % n);
}

static inline uint32_t sif_fast_div(uint64_t val, uint32_t n, uint32_t shift) {
  return shift ? (uint32_t)(val >> shift) : (uint32_t)(val / n);
}

uint64_t sif_get_flat_index(uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz);

/*
 * @brief Splits a flat grid index back into its three cell coordinates.
 */
static inline void sif_unflatten_index(const sif_grid_t* grid, uint64_t flat,
  uint32_t* ix, uint32_t* iy, uint32_t* iz) {

  const uint32_t n = grid->n_cells;
  const uint32_t p2_mask = grid->p2_mask;
  const uint32_t p2_shift = grid->p2_shift;

  *iz = sif_fast_mod(flat, n, p2_mask);
  *iy = sif_fast_mod(sif_fast_div(flat, n, p2_shift), n, p2_mask);
  *ix = sif_fast_div(flat, (uint64_t)n * n, p2_shift ? 2 * p2_shift : 0);
}

/* --- Candidate scanning (shared by every finder) --- */

typedef struct {
  real_t delta;
  uint64_t flat_idx;
} sif_candidate_t;

/*
 * @brief A reusable, self-growing list of candidate cells.
 *
 * Zero-initialize before first use, then hand the same buffer to every scan so
 * the allocation is amortized across radii.
 */
typedef struct {
  sif_candidate_t* items;
  uint64_t count;
  uint64_t capacity;
} sif_candidate_buffer_t;

void sif_candidate_buffer_free(sif_candidate_buffer_t* buf);

/*
 * @brief Collects every unmasked cell at or below `threshold`, sorted by
 * increasing delta (deepest underdensity first).
 *
 * @return SIF_OK on success (buf->count may legitimately be 0),
 * SIF_ERR_ALLOC if the buffer could not grow
 */
int sif_finder_scan_candidates(const sif_grid_t* grid,
  const sif_bitmask_t* mask, real_t threshold, sif_candidate_buffer_t* buf);

/* --- Per-radius reporting --- */

typedef struct {
  uint64_t n_candidates;
  uint64_t rejected_masked;  /* already covered by an accepted void */
  uint64_t rejected_proxy;   /* failed the cheap axis-pole overlap probe */
  uint64_t rejected_mesh;    /* failed the exact pairwise overlap test */
  uint64_t rejected_rescale; /* radius rescaling did not converge */
  uint64_t rejected_exact;   /* failed a re-check after rescaling */
  uint64_t accepted;
} sif_finder_radius_stats_t;

/*
 * @brief Emits the per-radius TRACE breakdown and the INFO summary line.
 */
void sif_finder_log_radius(const char* tag, real_t radius,
  const sif_finder_radius_stats_t* stats, uint64_t total_voids, double elapsed_s);

/* --- Geometry --- */

uint8_t sif_check_overlap_cells(const sif_bitmask_t* mask, uint32_t n_cells,
  uint32_t p2_mask, uint32_t ix, uint32_t iy, uint32_t iz, uint32_t r);

uint8_t sif_check_overlap_mesh(const sif_catalog_t* cat, real_t cx, real_t cy,
  real_t cz, real_t r, real_t max_r, real_t box_length,
  const sif_cell_linked_list_t* cll, uint32_t p2_mask, real_t overlap_fraction);

/*
 * @brief Marks every grid cell whose center lies inside the void sphere.
 */
void sif_mark_sphere(sif_bitmask_t* mask, real_t cx, real_t cy, real_t cz,
  real_t r_true, uint32_t n_cells, uint32_t p2_mask, real_t cell_length);

#endif // __SIF_FINDER_UTILS_H
