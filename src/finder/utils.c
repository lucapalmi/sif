/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "utils.h"

#include "sif/core/macros.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"

#include <math.h>
#include <stdlib.h>

/* Logical partitioning of the candidate scan. Decoupled from the thread count
 * so the two passes agree on chunk boundaries regardless of the team size. */
#define FINDER_SCAN_CHUNKS 256

SIF_DEFINE_QUICKSORT(sort_candidates, sif_candidate_t, a.delta < b.delta)

/* --- 1. Base Utilities --- */

SIF_PURE_FUNCTION inline uint64_t sif__flat_index(
  uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz) {
  return (uint64_t)ix * n * n + (uint64_t)iy * n + (uint64_t)iz;
}

/* --- 2. Candidate scanning --- */

void sif__candidate_buffer_free(sif_candidate_buffer_t* buf) {
  if (!buf)
    return;
  sif_free_aligned(buf->items);
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

/* Shared predicate for both passes, so they can never disagree. */
static inline uint8_t cell_is_candidate(const sif_real* delta,
  const sif_bitmask_t* mask, uint64_t i, sif_real threshold) {

  return delta[i] <= threshold && !sif_bitmask_get(mask, i);
}

int sif__finder_scan_candidates(const sif_grid_t* grid,
  const sif_bitmask_t* mask, sif_real threshold, sif_candidate_buffer_t* buf) {

  const uint64_t total_cells = grid->total_cells;
  const sif_real* g_delta = SIF_ASSUME_ALIGNED(grid->values);

  buf->count = 0;

  uint64_t chunk_offsets[FINDER_SCAN_CHUNKS] = {0};
  const uint64_t chunk = total_cells / FINDER_SCAN_CHUNKS;

  /* pass 1: count per chunk */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t start = (uint64_t)c * chunk;
    const uint64_t end =
      (c == FINDER_SCAN_CHUNKS - 1) ? total_cells : start + chunk;

    uint64_t local = 0;
    for (uint64_t i = start; i < end; i++)
      local += cell_is_candidate(g_delta, mask, i, threshold);
    chunk_offsets[c] = local;
  }

  /* exclusive prefix sum: chunk_offsets[c] becomes that chunk's write cursor */
  uint64_t n_candidates = 0;
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t count = chunk_offsets[c];
    chunk_offsets[c] = n_candidates;
    n_candidates += count;
  }

  if (n_candidates == 0)
    return SIF_OK;

  if (n_candidates > buf->capacity) {
    const uint64_t new_capacity = (uint64_t)((double)n_candidates * 1.2) + 1;

    /* Pass 2 rewrites every entry, so nothing in the old buffer is worth
     * carrying over. Releasing it first keeps the high-water mark at one
     * buffer instead of two, which on a 2250^3 grid is the difference between
     * ~10 and ~19 GiB at a 5% candidate rate. The cost is that a failure here
     * leaves the buffer empty rather than stale, so count and capacity are
     * cleared to match: the caller aborts the run either way. */
    sif_free_aligned(buf->items);
    buf->items = NULL;
    buf->capacity = 0;

    sif_candidate_t* items =
      sif_malloc_aligned(new_capacity * sizeof(sif_candidate_t));
    if (!items) {
      SIF_LOG_ERROR(
        "finder", "failed to allocate %" PRIu64 " candidates", new_capacity);
      return SIF_ERR_ALLOC;
    }
    buf->items = items;
    buf->capacity = new_capacity;
  }

  sif_candidate_t* candidates = buf->items;

  /* pass 2: write, each chunk into its own reserved span */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t start = (uint64_t)c * chunk;
    const uint64_t end =
      (c == FINDER_SCAN_CHUNKS - 1) ? total_cells : start + chunk;

    uint64_t w = chunk_offsets[c];
    for (uint64_t i = start; i < end; i++) {
      if (cell_is_candidate(g_delta, mask, i, threshold)) {
        candidates[w].flat_idx = i;
        candidates[w].delta = g_delta[i];
        w++;
      }
    }
  }

  sort_candidates(candidates, n_candidates);
  buf->count = n_candidates;

  return SIF_OK;
}

void sif__finder_log_radius(const char* tag, sif_real radius,
  const sif_finder_radius_stats_t* stats, uint64_t total_voids,
  double elapsed_s) {

  SIF_LOG_TRACE(
    tag, "initial candidates:        %7" PRIu64, stats->n_candidates);
  SIF_LOG_TRACE(
    tag, "already masked:            %7" PRIu64, stats->rejected_masked);
  SIF_LOG_TRACE(
    tag, "proxy overlap exclusions:  %7" PRIu64, stats->rejected_proxy);
  SIF_LOG_TRACE(
    tag, "mesh overlap exclusions:   %7" PRIu64, stats->rejected_mesh);
  SIF_LOG_TRACE(
    tag, "rescaling failures:        %7" PRIu64, stats->rejected_rescale);
  SIF_LOG_TRACE(
    tag, "exact re-check exclusions: %7" PRIu64, stats->rejected_exact);

  SIF_LOG_INFO(tag,
    "r = %6.2f | candidates: %7" PRIu64 " | accepted: %7" PRIu64
    " | total voids: %7" PRIu64 " | time: %6.3fs",
    (double)radius, stats->n_candidates, stats->accepted, total_voids,
    elapsed_s);

  SIF_LOG_FLUSH();
}

/* --- 3. Geometry --- */

/*
 * Cheap rejection probe: samples the mask at the six axis poles of the shrunk
 * sphere. A hit means this candidate reaches into an accepted void, so it can
 * be dropped without the full pairwise test. It is a heuristic, not exact: it
 * can miss overlaps that do not touch a pole, which is why the exact mesh
 * check still runs on everything that survives.
 */
SIF_HOT_LOOP SIF_PURE_FUNCTION uint8_t sif__overlap_quick(
  const sif_bitmask_t* mask, uint32_t n_cells, uint32_t p2_mask, uint32_t ix,
  uint32_t iy, uint32_t iz, uint32_t r) {

  uint32_t x_nord = sif__wrap_pbc(ix + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, x_nord, iy, iz)))
    return 1;

  uint32_t x_south = sif__wrap_pbc((int32_t)ix - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, x_south, iy, iz)))
    return 1;

  uint32_t y_nord = sif__wrap_pbc(iy + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, ix, y_nord, iz)))
    return 1;

  uint32_t y_south = sif__wrap_pbc((int32_t)iy - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, ix, y_south, iz)))
    return 1;

  uint32_t z_nord = sif__wrap_pbc(iz + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, ix, iy, z_nord)))
    return 1;

  uint32_t z_south = sif__wrap_pbc((int32_t)iz - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif__flat_index(n_cells, ix, iy, z_south)))
    return 1;

  return 0;
}

SIF_HOT_LOOP uint8_t sif__overlap_exact(const sif_catalog_t* cat, sif_real cx,
  sif_real cy, sif_real cz, sif_real r, sif_real max_r, sif_real box_length,
  const sif_cell_linked_list_t* cll, uint32_t p2_mask,
  sif_real overlap_fraction) {

  (void)p2_mask; /* the list has its own grid, see cll_mask below */

  const sif_real* cat_cx = SIF_ASSUME_ALIGNED(cat->cx);
  const sif_real* cat_cy = SIF_ASSUME_ALIGNED(cat->cy);
  const sif_real* cat_cz = SIF_ASSUME_ALIGNED(cat->cz);
  const sif_real* cat_r = SIF_ASSUME_ALIGNED(cat->radii);

  sif_real half_box = box_length * 0.5f;
  sif_real search_rad = max_r + r * (1.0f - overlap_fraction);

  int32_t ix_min =
    (int32_t)SIF_REAL_FLOOR((cx - search_rad) * cll->inv_cell_length);
  int32_t ix_max =
    (int32_t)SIF_REAL_FLOOR((cx + search_rad) * cll->inv_cell_length);
  int32_t iy_min =
    (int32_t)SIF_REAL_FLOOR((cy - search_rad) * cll->inv_cell_length);
  int32_t iy_max =
    (int32_t)SIF_REAL_FLOOR((cy + search_rad) * cll->inv_cell_length);
  int32_t iz_min =
    (int32_t)SIF_REAL_FLOOR((cz - search_rad) * cll->inv_cell_length);
  int32_t iz_max =
    (int32_t)SIF_REAL_FLOOR((cz + search_rad) * cll->inv_cell_length);

  const int32_t n_cells = (int32_t)cll->n_cells;

  /* The CLL has its OWN grid (coarse_n cells), which is different from the
   * main grid, so it needs a mask derived from its own size. */
  uint32_t cll_mask = 0;
  if (n_cells > 0 && (n_cells & (n_cells - 1)) == 0)
    cll_mask = (uint32_t)n_cells - 1;

  uint8_t is_interior =
    (cx - search_rad >= 0.0f) && (cx + search_rad < box_length) &&
    (cy - search_rad >= 0.0f) && (cy + search_rad < box_length) &&
    (cz - search_rad >= 0.0f) && (cz + search_rad < box_length);

  for (int32_t ix = ix_min; ix <= ix_max; ix++) {
    int32_t wrap_x = sif__wrap_pbc(ix, n_cells, cll_mask);
    for (int32_t iy = iy_min; iy <= iy_max; iy++) {
      int32_t wrap_y = sif__wrap_pbc(iy, n_cells, cll_mask);
      for (int32_t iz = iz_min; iz <= iz_max; iz++) {
        int32_t wrap_z = sif__wrap_pbc(iz, n_cells, cll_mask);

        uint32_t c_flat =
          wrap_x * n_cells * n_cells + wrap_y * n_cells + wrap_z;
        int32_t curr = cll->head[c_flat];

        if (is_interior) {
          while (curr != -1) {
            /* Shrink the collision radius by the smaller of the two, so
             * overlap_fraction means "fraction of the smaller void". */
            sif_real min_r = (r < cat_r[curr]) ? r : cat_r[curr];
            sif_real r_tot = r + cat_r[curr] - overlap_fraction * min_r;

            sif_real r2 = r_tot * r_tot;
            sif_real dx = cx - cat_cx[curr];
            sif_real dy = cy - cat_cy[curr];
            sif_real dz = cz - cat_cz[curr];
            if (dx * dx + dy * dy + dz * dz < r2)
              return 1;
            curr = cll->next[curr];
          }
        } else {
          while (curr != -1) {
            sif_real min_r = (r < cat_r[curr]) ? r : cat_r[curr];
            sif_real r_tot = r + cat_r[curr] - overlap_fraction * min_r;
            sif_real r_tot2 = r_tot * r_tot;

            sif_real dx = SIF_REAL_ABS(cx - cat_cx[curr]);
            dx = (dx > half_box) ? box_length - dx : dx;
            sif_real dx2 = dx * dx;
            if (dx2 >= r_tot2) {
              curr = cll->next[curr];
              continue;
            }

            sif_real dy = SIF_REAL_ABS(cy - cat_cy[curr]);
            dy = (dy > half_box) ? box_length - dy : dy;
            sif_real dy2 = dy * dy;
            if (dx2 + dy2 >= r_tot2) {
              curr = cll->next[curr];
              continue;
            }

            sif_real dz = SIF_REAL_ABS(cz - cat_cz[curr]);
            dz = (dz > half_box) ? box_length - dz : dz;
            if (dx2 + dy2 + dz * dz < r_tot2)
              return 1;

            curr = cll->next[curr];
          }
        }
      }
    }
  }
  return 0;
}

/* --- 4. Marking --- */

SIF_HOT_LOOP void sif__mark_sphere(sif_bitmask_t* mask, sif_real cx,
  sif_real cy, sif_real cz, sif_real r_true, uint32_t n_cells, uint32_t p2_mask,
  sif_real cell_length) {

  const sif_real r2 = r_true * r_true;
  const sif_real inv_cell_length = 1.0f / cell_length;
  const sif_real box_length = (sif_real)n_cells * cell_length;
  const sif_real half_box = box_length * 0.5f;

  const int32_t cell_radius = (int32_t)(r_true * inv_cell_length) + 1;
  const int32_t center_ix = (int32_t)(cx * inv_cell_length);
  const int32_t center_iy = (int32_t)(cy * inv_cell_length);
  const int32_t center_iz = (int32_t)(cz * inv_cell_length);

  const int32_t span = 2 * cell_radius + 1;

  /* Threads write bits that share 64-bit words, so the atomic setter is
   * mandatory here. Setting a bit is idempotent, so the result does not depend
   * on the schedule. */
#pragma omp parallel for schedule(static) if (span > 24)
  for (int32_t dx = -cell_radius; dx <= cell_radius; dx++) {
    const int32_t global_x = center_ix + dx;
    const sif_real px = (sif_real)global_x * cell_length;
    sif_real dist_x = SIF_REAL_ABS(px - cx);
    dist_x = (dist_x > half_box) ? box_length - dist_x : dist_x;
    const sif_real dx2 = dist_x * dist_x;

    if (dx2 > r2)
      continue;

    const uint32_t wrap_x = sif__wrap_pbc(global_x, n_cells, p2_mask);

    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      const int32_t global_y = center_iy + dy;
      const sif_real py = (sif_real)global_y * cell_length;
      sif_real dist_y = SIF_REAL_ABS(py - cy);
      dist_y = (dist_y > half_box) ? box_length - dist_y : dist_y;
      const sif_real dxy2 = dx2 + dist_y * dist_y;

      if (dxy2 > r2)
        continue;

      const uint32_t wrap_y = sif__wrap_pbc(global_y, n_cells, p2_mask);

      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        const int32_t global_z = center_iz + dz;
        const sif_real pz = (sif_real)global_z * cell_length;
        sif_real dist_z = SIF_REAL_ABS(pz - cz);
        dist_z = (dist_z > half_box) ? box_length - dist_z : dist_z;

        if (dxy2 + dist_z * dist_z <= r2) {
          const uint32_t wrap_z = sif__wrap_pbc(global_z, n_cells, p2_mask);
          sif_bitmask_set_atomic(
            mask, sif__flat_index(n_cells, wrap_x, wrap_y, wrap_z));
        }
      }
    }
  }
}
