#include "utils.h"

#include "sif/core/macros.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"

#include <math.h>
#include <stdlib.h>

/* Logical partitioning of the candidate scan. Decoupled from the thread count
 * so the two passes agree on chunk boundaries regardless of the team size. */
#define __FINDER_SCAN_CHUNKS 256

SIF_DEFINE_QUICKSORT(__sort_candidates, sif_candidate_t, a.delta < b.delta)

/* --- 1. Base Utilities --- */

PURE_FUNCTION inline uint64_t sif_get_flat_index(
  uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz) {
  return (uint64_t)ix * n * n + (uint64_t)iy * n + (uint64_t)iz;
}

/* --- 2. Candidate scanning --- */

void sif_candidate_buffer_free(sif_candidate_buffer_t* buf) {
  if (!buf)
    return;
  sif_free_aligned(buf->items);
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

/*
 * True when cell `i` is strictly lower than all 26 of its neighbours.
 *
 * The interior fast path walks precomputed flat offsets; boundary cells fall
 * back to per-axis periodic wrapping.
 */
static inline uint8_t __is_local_minimum(const real_t* delta, uint64_t i,
  real_t center_val, uint32_t N, uint32_t p2_mask, uint32_t p2_shift,
  const int64_t* neighbor_offsets) {

  uint32_t iz = sif_fast_mod(i, N, p2_mask);
  uint32_t iy = sif_fast_mod(sif_fast_div(i, N, p2_shift), N, p2_mask);
  uint32_t ix = sif_fast_div(i, (uint64_t)N * N, p2_shift ? 2 * p2_shift : 0);

  if (ix > 0 && ix < N - 1 && iy > 0 && iy < N - 1 && iz > 0 && iz < N - 1) {
    for (int n = 0; n < 26; n++) {
      if (delta[i + neighbor_offsets[n]] <= center_val)
        return 0;
    }
    return 1;
  }

  for (int32_t dx = -1; dx <= 1; dx++) {
    int32_t nx = sif_wrap_pbc((int32_t)ix + dx, N, p2_mask);
    for (int32_t dy = -1; dy <= 1; dy++) {
      int32_t ny = sif_wrap_pbc((int32_t)iy + dy, N, p2_mask);
      for (int32_t dz = -1; dz <= 1; dz++) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        int32_t nz = sif_wrap_pbc((int32_t)iz + dz, N, p2_mask);
        if (delta[sif_get_flat_index(N, nx, ny, nz)] <= center_val)
          return 0;
      }
    }
  }
  return 1;
}

/* Shared predicate for both passes, so they can never disagree. */
static inline uint8_t __cell_is_candidate(const real_t* delta,
  const sif_bitmask_t* mask, uint64_t i, real_t threshold,
  uint8_t require_minimum, uint32_t N, uint32_t p2_mask, uint32_t p2_shift,
  const int64_t* neighbor_offsets) {

  const real_t center_val = delta[i];

  if (center_val > threshold || sif_bitmask_get(mask, i))
    return 0;

  if (!require_minimum)
    return 1;

  return __is_local_minimum(
    delta, i, center_val, N, p2_mask, p2_shift, neighbor_offsets);
}

int sif_finder_scan_candidates(const sif_grid_t* grid,
  const sif_bitmask_t* mask, real_t threshold, uint8_t require_minimum,
  sif_candidate_buffer_t* buf) {

  const uint64_t total_cells = grid->total_cells;
  const uint32_t N = grid->n_cells;
  const uint32_t p2_mask = grid->p2_mask;
  const uint32_t p2_shift = grid->p2_shift;
  const real_t* g_delta = SIF_ASSUME_ALIGNED(grid->delta);

  buf->count = 0;

  int64_t neighbor_offsets[26];
  if (require_minimum) {
    int n_idx = 0;
    for (int32_t dx = -1; dx <= 1; dx++) {
      for (int32_t dy = -1; dy <= 1; dy++) {
        for (int32_t dz = -1; dz <= 1; dz++) {
          if (dx == 0 && dy == 0 && dz == 0)
            continue;
          neighbor_offsets[n_idx++] =
            (int64_t)dx * N * N + (int64_t)dy * N + dz;
        }
      }
    }
  }

  uint64_t chunk_offsets[__FINDER_SCAN_CHUNKS] = {0};
  const uint64_t chunk = total_cells / __FINDER_SCAN_CHUNKS;

  /* pass 1: count per chunk */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < __FINDER_SCAN_CHUNKS; c++) {
    const uint64_t start = (uint64_t)c * chunk;
    const uint64_t end =
      (c == __FINDER_SCAN_CHUNKS - 1) ? total_cells : start + chunk;

    uint64_t local = 0;
    for (uint64_t i = start; i < end; i++) {
      local += __cell_is_candidate(g_delta, mask, i, threshold, require_minimum,
        N, p2_mask, p2_shift, neighbor_offsets);
    }
    chunk_offsets[c] = local;
  }

  /* exclusive prefix sum: chunk_offsets[c] becomes that chunk's write cursor */
  uint64_t n_candidates = 0;
  for (int c = 0; c < __FINDER_SCAN_CHUNKS; c++) {
    const uint64_t count = chunk_offsets[c];
    chunk_offsets[c] = n_candidates;
    n_candidates += count;
  }

  if (n_candidates == 0)
    return SIF_OK;

  if (n_candidates > buf->capacity) {
    const uint64_t new_capacity = (uint64_t)((double)n_candidates * 1.2) + 1;
    sif_candidate_t* items =
      sif_malloc_aligned(new_capacity * sizeof(sif_candidate_t));
    if (!items) {
      SIF_LOG_ERROR("finder",
        "failed to allocate %" PRIu64 " candidates", new_capacity);
      return SIF_ERR_ALLOC;
    }
    sif_free_aligned(buf->items);
    buf->items = items;
    buf->capacity = new_capacity;
  }

  sif_candidate_t* candidates = buf->items;

  /* pass 2: write, each chunk into its own reserved span */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < __FINDER_SCAN_CHUNKS; c++) {
    const uint64_t start = (uint64_t)c * chunk;
    const uint64_t end =
      (c == __FINDER_SCAN_CHUNKS - 1) ? total_cells : start + chunk;

    uint64_t w = chunk_offsets[c];
    for (uint64_t i = start; i < end; i++) {
      if (__cell_is_candidate(g_delta, mask, i, threshold, require_minimum, N,
            p2_mask, p2_shift, neighbor_offsets)) {
        candidates[w].flat_idx = i;
        candidates[w].delta = g_delta[i];
        w++;
      }
    }
  }

  __sort_candidates(candidates, n_candidates);
  buf->count = n_candidates;

  return SIF_OK;
}

void sif_finder_log_radius(const char* tag, real_t radius,
  const sif_finder_radius_stats_t* stats, uint64_t total_voids, double elapsed_s) {

  SIF_LOG_TRACE(tag, "initial candidates:        %7" PRIu64,
    stats->n_candidates);
  SIF_LOG_TRACE(tag, "already masked:            %7" PRIu64,
    stats->rejected_masked);
  SIF_LOG_TRACE(tag, "proxy overlap exclusions:  %7" PRIu64,
    stats->rejected_proxy);
  SIF_LOG_TRACE(tag, "mesh overlap exclusions:   %7" PRIu64,
    stats->rejected_mesh);
  SIF_LOG_TRACE(tag, "rescaling failures:        %7" PRIu64,
    stats->rejected_rescale);
  SIF_LOG_TRACE(tag, "exact re-check exclusions: %7" PRIu64,
    stats->rejected_exact);

  SIF_LOG_INFO(tag,
    "r = %6.2f | candidates: %7" PRIu64 " | accepted: %7" PRIu64
    " | total voids: %7" PRIu64 " | time: %6.3fs",
    (double)radius, stats->n_candidates, stats->accepted, total_voids,
    elapsed_s);

  SIF_LOG_FLUSH();
}

/* --- 3. Geometry --- */

HOT_LOOP void sif_refine_center_hessian(const sif_grid_t* grid, uint32_t ix,
  uint32_t iy, uint32_t iz, real_t* cx, real_t* cy, real_t* cz) {

  const uint32_t N = grid->n_cells;
  const uint32_t mask = grid->p2_mask;
  const uint32_t shift = grid->p2_shift;
  const real_t* d = grid->delta;

  real_t dx = 0.5f * (d[sif_fast_get_flat_index(N, shift, sif_wrap_pbc(ix + 1, N, mask), iy, iz)] -
                       d[sif_fast_get_flat_index(N, shift, sif_wrap_pbc(ix - 1, N, mask), iy, iz)]);
  real_t dy = 0.5f * (d[sif_fast_get_flat_index(N, shift, ix, sif_wrap_pbc(iy + 1, N, mask), iz)] -
                       d[sif_fast_get_flat_index(N, shift, ix, sif_wrap_pbc(iy - 1, N, mask), iz)]);
  real_t dz = 0.5f * (d[sif_fast_get_flat_index(N, shift, ix, iy, sif_wrap_pbc(iz + 1, N, mask))] -
                       d[sif_fast_get_flat_index(N, shift, ix, iy, sif_wrap_pbc(iz - 1, N, mask))]);

  real_t dxx = d[sif_fast_get_flat_index(N, shift, sif_wrap_pbc(ix + 1, N, mask), iy, iz)] -
               2.0f * d[sif_fast_get_flat_index(N, shift, ix, iy, iz)] +
               d[sif_fast_get_flat_index(N, shift, sif_wrap_pbc(ix - 1, N, mask), iy, iz)];
  real_t dyy = d[sif_fast_get_flat_index(N, shift, ix, sif_wrap_pbc(iy + 1, N, mask), iz)] -
               2.0f * d[sif_fast_get_flat_index(N, shift, ix, iy, iz)] +
               d[sif_fast_get_flat_index(N, shift, ix, sif_wrap_pbc(iy - 1, N, mask), iz)];
  real_t dzz = d[sif_fast_get_flat_index(N, shift, ix, iy, sif_wrap_pbc(iz + 1, N, mask))] -
               2.0f * d[sif_fast_get_flat_index(N, shift, ix, iy, iz)] +
               d[sif_fast_get_flat_index(N, shift, ix, iy, sif_wrap_pbc(iz - 1, N, mask))];

  if (dxx > 1e-6) {
    real_t shift_x = dx / dxx;
    shift_x = (shift_x > 0.5f) ? 0.5f : ((shift_x < -0.5f) ? -0.5f : shift_x);
    *cx -= shift_x * grid->cell_length;
  }
  if (dyy > 1e-6) {
    real_t shift_y = dy / dyy;
    shift_y = (shift_y > 0.5f) ? 0.5f : ((shift_y < -0.5f) ? -0.5f : shift_y);
    *cy -= shift_y * grid->cell_length;
  }
  if (dzz > 1e-6) {
    real_t shift_z = dz / dzz;
    shift_z = (shift_z > 0.5f) ? 0.5f : ((shift_z < -0.5f) ? -0.5f : shift_z);
    *cz -= shift_z * grid->cell_length;
  }

  real_t box_L = grid->box_length;
  if (*cx < 0.0f)
    *cx += box_L;
  if (*cx >= box_L)
    *cx -= box_L;
  if (*cy < 0.0f)
    *cy += box_L;
  if (*cy >= box_L)
    *cy -= box_L;
  if (*cz < 0.0f)
    *cz += box_L;
  if (*cz >= box_L)
    *cz -= box_L;
}

/*
 * Cheap rejection probe: samples the mask at the six axis poles of the shrunk
 * sphere. A hit means this candidate reaches into an accepted void, so it can
 * be dropped without the full pairwise test. It is a heuristic, not exact: it
 * can miss overlaps that do not touch a pole, which is why the exact mesh
 * check still runs on everything that survives.
 */
HOT_LOOP PURE_FUNCTION uint8_t sif_check_overlap_cells(const sif_bitmask_t* mask,
  uint32_t n_cells, uint32_t p2_mask, uint32_t ix, uint32_t iy, uint32_t iz, uint32_t r) {

  uint32_t x_nord = sif_wrap_pbc(ix + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, x_nord, iy, iz)))
    return 1;

  uint32_t x_south = sif_wrap_pbc((int32_t)ix - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, x_south, iy, iz)))
    return 1;

  uint32_t y_nord = sif_wrap_pbc(iy + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, ix, y_nord, iz)))
    return 1;

  uint32_t y_south = sif_wrap_pbc((int32_t)iy - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, ix, y_south, iz)))
    return 1;

  uint32_t z_nord = sif_wrap_pbc(iz + r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, ix, iy, z_nord)))
    return 1;

  uint32_t z_south = sif_wrap_pbc((int32_t)iz - r, n_cells, p2_mask);
  if (sif_bitmask_get(mask, sif_get_flat_index(n_cells, ix, iy, z_south)))
    return 1;

  return 0;
}

HOT_LOOP uint8_t sif_check_overlap_mesh(const sif_catalog_t* cat, real_t cx,
  real_t cy, real_t cz, real_t r, real_t max_r, real_t box_length,
  const sif_cell_linked_list_t* cll, uint32_t p2_mask, real_t overlap_fraction) {

  (void)p2_mask; /* the list has its own grid, see cll_mask below */

  const real_t* cat_cx = SIF_ASSUME_ALIGNED(cat->cx);
  const real_t* cat_cy = SIF_ASSUME_ALIGNED(cat->cy);
  const real_t* cat_cz = SIF_ASSUME_ALIGNED(cat->cz);
  const real_t* cat_r = SIF_ASSUME_ALIGNED(cat->radii);

  real_t half_box = box_length * 0.5f;
  real_t search_rad = max_r + r * (1.0f - overlap_fraction);

  int32_t ix_min =
    (int32_t)REAL_FLOOR((cx - search_rad) * cll->inv_cell_length);
  int32_t ix_max =
    (int32_t)REAL_FLOOR((cx + search_rad) * cll->inv_cell_length);
  int32_t iy_min =
    (int32_t)REAL_FLOOR((cy - search_rad) * cll->inv_cell_length);
  int32_t iy_max =
    (int32_t)REAL_FLOOR((cy + search_rad) * cll->inv_cell_length);
  int32_t iz_min =
    (int32_t)REAL_FLOOR((cz - search_rad) * cll->inv_cell_length);
  int32_t iz_max =
    (int32_t)REAL_FLOOR((cz + search_rad) * cll->inv_cell_length);

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
    int32_t wrap_x = sif_wrap_pbc(ix, n_cells, cll_mask);
    for (int32_t iy = iy_min; iy <= iy_max; iy++) {
      int32_t wrap_y = sif_wrap_pbc(iy, n_cells, cll_mask);
      for (int32_t iz = iz_min; iz <= iz_max; iz++) {
        int32_t wrap_z = sif_wrap_pbc(iz, n_cells, cll_mask);

        uint32_t c_flat =
          wrap_x * n_cells * n_cells + wrap_y * n_cells + wrap_z;
        int32_t curr = cll->head[c_flat];

        if (is_interior) {
          while (curr != -1) {
            /* Shrink the collision radius by the smaller of the two, so
             * overlap_fraction means "fraction of the smaller void". */
            real_t min_r = (r < cat_r[curr]) ? r : cat_r[curr];
            real_t r_tot = r + cat_r[curr] - overlap_fraction * min_r;

            real_t r2 = r_tot * r_tot;
            real_t dx = cx - cat_cx[curr];
            real_t dy = cy - cat_cy[curr];
            real_t dz = cz - cat_cz[curr];
            if (dx * dx + dy * dy + dz * dz < r2)
              return 1;
            curr = cll->next[curr];
          }
        } else {
          while (curr != -1) {
            real_t min_r = (r < cat_r[curr]) ? r : cat_r[curr];
            real_t r_tot = r + cat_r[curr] - overlap_fraction * min_r;
            real_t r_tot2 = r_tot * r_tot;

            real_t dx = REAL_ABS(cx - cat_cx[curr]);
            dx = (dx > half_box) ? box_length - dx : dx;
            real_t dx2 = dx * dx;
            if (dx2 >= r_tot2) {
              curr = cll->next[curr];
              continue;
            }

            real_t dy = REAL_ABS(cy - cat_cy[curr]);
            dy = (dy > half_box) ? box_length - dy : dy;
            real_t dy2 = dy * dy;
            if (dx2 + dy2 >= r_tot2) {
              curr = cll->next[curr];
              continue;
            }

            real_t dz = REAL_ABS(cz - cat_cz[curr]);
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

HOT_LOOP void sif_mark_sphere(sif_bitmask_t* mask, real_t cx, real_t cy,
  real_t cz, real_t r_true, uint32_t n_cells, uint32_t p2_mask,
  real_t cell_length) {

  const real_t r2 = r_true * r_true;
  const real_t inv_cell_length = 1.0f / cell_length;
  const real_t box_length = (real_t)n_cells * cell_length;
  const real_t half_box = box_length * 0.5f;

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
    const real_t px = (global_x + 0.5f) * cell_length;
    real_t dist_x = REAL_ABS(px - cx);
    dist_x = (dist_x > half_box) ? box_length - dist_x : dist_x;
    const real_t dx2 = dist_x * dist_x;

    if (dx2 > r2)
      continue;

    const uint32_t wrap_x = sif_wrap_pbc(global_x, n_cells, p2_mask);

    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      const int32_t global_y = center_iy + dy;
      const real_t py = (global_y + 0.5f) * cell_length;
      real_t dist_y = REAL_ABS(py - cy);
      dist_y = (dist_y > half_box) ? box_length - dist_y : dist_y;
      const real_t dxy2 = dx2 + dist_y * dist_y;

      if (dxy2 > r2)
        continue;

      const uint32_t wrap_y = sif_wrap_pbc(global_y, n_cells, p2_mask);

      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        const int32_t global_z = center_iz + dz;
        const real_t pz = (global_z + 0.5f) * cell_length;
        real_t dist_z = REAL_ABS(pz - cz);
        dist_z = (dist_z > half_box) ? box_length - dist_z : dist_z;

        if (dxy2 + dist_z * dist_z <= r2) {
          const uint32_t wrap_z = sif_wrap_pbc(global_z, n_cells, p2_mask);
          sif_bitmask_set_atomic(
            mask, sif_get_flat_index(n_cells, wrap_x, wrap_y, wrap_z));
        }
      }
    }
  }
}
