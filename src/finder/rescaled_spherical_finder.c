#include "sif/finder/rescaled_spherical_finder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/chain_mesh.h"

#include "core/get_system.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"
#include "sif/utils/timer.h"

#include "utils.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846f
#endif

SIF_DEFINE_QUICKSORT(__sort_radii_desc, real_t, a > b)
SIF_DEFINE_QUICKSORT(__sort_real_asc, real_t, a < b)

#define __TAG "rescaled_spherical_finder"

/* Candidates evaluated speculatively in parallel before the sequential
 * commit. */
#define __BATCH_SIZE 1024

/* Coarse grid for the void-vs-void overlap index. */
#define __VOID_CLL_CELLS 32

#define __CATALOG_INITIAL_CAPACITY 250000

/*
 * Rescaling constants.
 *
 * These were previously selectable through SIF_FINDER_RMIN_FACTOR_* and
 * SIF_FINDER_NOISE_TOLERANCE_*. Both option families encoded their documented
 * default as bit pattern 0, which never reached the dispatch, so the values
 * below are the ones every run has actually used. The options were removed
 * rather than fixed, to keep results reproducible.
 *
 * (For the record, the API had advertised rmin_factor = 0.75 and a noise
 * tolerance of 20, which would have meant N_MIN ~ 25.)
 */
#define __RMIN_FACTOR 0.50f
#define __N_MIN       0.0f

/* Initial per-thread shell capacity. The buffer grows on demand, so this only
 * has to be a reasonable starting point rather than a worst case. */
#define __SHELL_INITIAL_CAPACITY 65536u

/* --- Growable per-thread scratch for the radial shell --- */

typedef struct {
  real_t* data;
  uint32_t count;
  uint32_t capacity;
} shell_buffer_t;

static int __shell_reserve(shell_buffer_t* b, uint32_t needed) {
  if (needed <= b->capacity)
    return SIF_OK;

  uint32_t new_capacity = b->capacity ? b->capacity : __SHELL_INITIAL_CAPACITY;
  while (new_capacity < needed) {
    if (new_capacity > UINT32_MAX / 2)
      return SIF_ERR_ALLOC;
    new_capacity *= 2;
  }

  real_t* grown = sif_malloc_aligned((size_t)new_capacity * sizeof(real_t));
  if (!grown)
    return SIF_ERR_ALLOC;

  if (b->count > 0)
    memcpy(grown, b->data, (size_t)b->count * sizeof(real_t));

  sif_free_aligned(b->data);
  b->data = grown;
  b->capacity = new_capacity;

  return SIF_OK;
}

/* --- Mesh traversal template --- */

/* Cell classification relative to the [rmin, r_search] annulus. */
#define __CELL_FULLY_CORE   1 /* entirely inside rmin: count, never store */
#define __CELL_STRADDLES    2 /* spans the rmin boundary: test each particle */
#define __CELL_PARTIAL_EDGE 3 /* spans the r_search boundary: test each        */
#define __CELL_FULLY_SHELL  4 /* entirely inside the annulus: store all        */

typedef struct {
  int32_t* dx;
  int32_t* dy;
  int32_t* dz;
  uint8_t* type;
  uint32_t count;
} mesh_template_t;

static void __template_free(mesh_template_t* tpl) {
  sif_free_aligned(tpl->dx);
  sif_free_aligned(tpl->dy);
  sif_free_aligned(tpl->dz);
  sif_free_aligned(tpl->type);
  memset(tpl, 0, sizeof(*tpl));
}

static int __template_build(mesh_template_t* tpl, real_t cell_length,
  real_t rmin, real_t r_search) {

  memset(tpl, 0, sizeof(*tpl));

  const int32_t cell_radius = (int32_t)(r_search / cell_length) + 1;
  const uint64_t max_cells =
    (uint64_t)(2 * cell_radius + 1) * (2 * cell_radius + 1) *
    (2 * cell_radius + 1);

  tpl->dx = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->dy = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->dz = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->type = sif_malloc_aligned(max_cells * sizeof(uint8_t));

  if (!tpl->dx || !tpl->dy || !tpl->dz || !tpl->type) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the mesh template");
    __template_free(tpl);
    return SIF_ERR_ALLOC;
  }

  const real_t r_search2 = r_search * r_search;
  const real_t r_core2 = rmin * rmin;

  for (int32_t dx = -cell_radius; dx <= cell_radius; dx++) {
    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        /* Nearest and farthest a point in this cell can be from the center. */
        const real_t min_x =
          (dx == 0) ? 0.0f : (REAL_ABS((real_t)dx) - 1.0f) * cell_length;
        const real_t min_y =
          (dy == 0) ? 0.0f : (REAL_ABS((real_t)dy) - 1.0f) * cell_length;
        const real_t min_z =
          (dz == 0) ? 0.0f : (REAL_ABS((real_t)dz) - 1.0f) * cell_length;
        const real_t min_dist2 = min_x * min_x + min_y * min_y + min_z * min_z;

        if (min_dist2 > r_search2)
          continue;

        const real_t max_x = (REAL_ABS((real_t)dx) + 1.0f) * cell_length;
        const real_t max_y = (REAL_ABS((real_t)dy) + 1.0f) * cell_length;
        const real_t max_z = (REAL_ABS((real_t)dz) + 1.0f) * cell_length;
        const real_t max_dist2 = max_x * max_x + max_y * max_y + max_z * max_z;

        tpl->dx[tpl->count] = dx;
        tpl->dy[tpl->count] = dy;
        tpl->dz[tpl->count] = dz;

        if (max_dist2 <= r_core2) {
          tpl->type[tpl->count] = __CELL_FULLY_CORE;
        } else if (min_dist2 > r_core2) {
          tpl->type[tpl->count] = (max_dist2 <= r_search2)
                                    ? __CELL_FULLY_SHELL
                                    : __CELL_PARTIAL_EDGE;
        } else {
          tpl->type[tpl->count] = __CELL_STRADDLES;
        }
        tpl->count++;
      }
    }
  }

  return SIF_OK;
}

/* --- Radius rescaling --- */

/*
 * Grows the void outwards from `center` until the enclosed number density
 * first rises above (1 + threshold) times the mean.
 *
 * @return the rescaled radius, or -1 if no acceptable radius exists
 */
HOT_LOOP static real_t __find_exact_radius(const sif_chain_mesh_t* mesh,
  real_t cx, real_t cy, real_t cz, real_t r_search, real_t threshold,
  real_t vol_factor, shell_buffer_t* shell, real_t rmin,
  const mesh_template_t* tpl) {

  const real_t* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const real_t* my = SIF_ASSUME_ALIGNED(mesh->y);
  const real_t* mz = SIF_ASSUME_ALIGNED(mesh->z);

  const real_t inv_l = 1.0f / mesh->cell_length;
  const int32_t center_ix = (int32_t)(cx * inv_l);
  const int32_t center_iy = (int32_t)(cy * inv_l);
  const int32_t center_iz = (int32_t)(cz * inv_l);

  const int32_t N = (int32_t)mesh->n_cells;
  const real_t box_L = mesh->box_length;

  const real_t r_search2 = r_search * r_search;
  const real_t r_core2 = rmin * rmin;

  uint32_t n_core = 0;
  shell->count = 0;

  /* Above this many particles inside rmin the void is already denser than the
   * threshold allows, so it can be abandoned before doing any distance work. */
  const real_t expected_core = vol_factor * rmin * rmin * rmin;
  uint32_t max_core_particles = 0xFFFFFFFFu;
  if (expected_core >= __N_MIN)
    max_core_particles = (uint32_t)(expected_core * (1.0f + threshold));

  /* Cheap pre-pass: cell occupancies alone can exceed the core budget. */
  uint32_t guaranteed_core = 0;
  for (uint32_t i = 0; i < tpl->count; i++) {
    if (tpl->type[i] != __CELL_FULLY_CORE)
      continue;

    int32_t ix = center_ix + tpl->dx[i];
    if (ix < 0) ix += N; else if (ix >= N) ix -= N;
    int32_t iy = center_iy + tpl->dy[i];
    if (iy < 0) iy += N; else if (iy >= N) iy -= N;
    int32_t iz = center_iz + tpl->dz[i];
    if (iz < 0) iz += N; else if (iz >= N) iz -= N;

    const uint64_t flat = (uint64_t)ix * N * N + (uint64_t)iy * N + iz;
    guaranteed_core +=
      (uint32_t)(mesh->cell_offsets[flat + 1] - mesh->cell_offsets[flat]);

    if (guaranteed_core > max_core_particles)
      return -1.0f;
  }

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    /* Shift the query center instead of the particles, so the periodic image
     * is handled with one subtraction per axis. */
    int32_t ix = center_ix + tpl->dx[i];
    real_t cx_eff = cx;
    if (ix < 0) { ix += N; cx_eff += box_L; }
    else if (ix >= N) { ix -= N; cx_eff -= box_L; }

    int32_t iy = center_iy + tpl->dy[i];
    real_t cy_eff = cy;
    if (iy < 0) { iy += N; cy_eff += box_L; }
    else if (iy >= N) { iy -= N; cy_eff -= box_L; }

    int32_t iz = center_iz + tpl->dz[i];
    real_t cz_eff = cz;
    if (iz < 0) { iz += N; cz_eff += box_L; }
    else if (iz >= N) { iz -= N; cz_eff -= box_L; }

    const uint64_t flat = (uint64_t)ix * N * N + (uint64_t)iy * N + iz;
    const uint64_t p_start = mesh->cell_offsets[flat];
    const uint64_t p_end = mesh->cell_offsets[flat + 1];

    if (p_start == p_end)
      continue;

    const uint64_t p_count = p_end - p_start;

    if (type == __CELL_FULLY_CORE) {
      n_core += (uint32_t)p_count;
    } else if (type == __CELL_FULLY_SHELL) {
      if (__shell_reserve(shell, shell->count + (uint32_t)p_count) != SIF_OK)
        return -1.0f;

      real_t* out = shell->data + shell->count;
#pragma omp simd
      for (uint64_t p = 0; p < p_count; p++) {
        const real_t dx = mx[p_start + p] - cx_eff;
        const real_t dy = my[p_start + p] - cy_eff;
        const real_t dz = mz[p_start + p] - cz_eff;
        out[p] = dx * dx + dy * dy + dz * dz;
      }
      shell->count += (uint32_t)p_count;
    } else {
      if (__shell_reserve(shell, shell->count + (uint32_t)p_count) != SIF_OK)
        return -1.0f;

      for (uint64_t p = p_start; p < p_end; p++) {
        const real_t dx = mx[p] - cx_eff;
        const real_t dy = my[p] - cy_eff;
        const real_t dz = mz[p] - cz_eff;
        const real_t d2 = dx * dx + dy * dy + dz * dz;

        n_core += (d2 <= r_core2);
        shell->data[shell->count] = d2;
        shell->count += ((d2 > r_core2) & (d2 <= r_search2));
      }
    }

    if (n_core > max_core_particles)
      return -1.0f;
  }

  const uint32_t total_N = n_core + shell->count;
  if (total_N == 0)
    return -1.0f;

  /* If the whole search sphere is already denser than the threshold there is
   * no radius in range that satisfies it. */
  const real_t expected_search = vol_factor * r_search * r_search * r_search;
  if (expected_search >= __N_MIN) {
    if (((real_t)total_N / expected_search) - 1.0f <= threshold)
      return -1.0f;
  }

  if (shell->count > 0)
    __sort_real_asc(shell->data, shell->count);

  /* Walk inwards from the outermost shell particle. Comparisons are done on
   * squared distances so no square root is needed until the answer is found:
   *   n_in / (vol_factor * d^3) - 1 <= threshold
   *   <=> n_in^2 <= ((threshold + 1) * vol_factor)^2 * (d^2)^3
   */
  const real_t vol_factor2 = vol_factor * vol_factor;
  const real_t K = (threshold + 1.0f) * vol_factor;
  const real_t K2 = K * K;
  const real_t n_min2 = __N_MIN * __N_MIN;

  uint32_t current_N = total_N;

  for (int32_t i = (int32_t)shell->count - 1; i >= 0; i--) {
    const real_t d2_test = shell->data[i];
    if (d2_test == 0.0f) {
      current_N--;
      continue;
    }

    const uint32_t n_in = current_N - 1;
    const real_t d2_cube = d2_test * d2_test * d2_test;

    /* Below the noise floor the estimate stops being meaningful. */
    if (vol_factor2 * d2_cube < n_min2)
      return -1.0f;

    const real_t rn_in = (real_t)n_in;
    if (rn_in * rn_in <= K2 * d2_cube)
      return REAL_SQRT(d2_test);

    current_N--;
  }

  return -1.0f;
}

/* --- Speculative batch evaluation --- */

typedef struct {
  uint8_t status;  /* one of __BATCH_* below */
  real_t r_scaled;
  real_t cx, cy, cz;
  int32_t proxy_pole;
  uint32_t ix, iy, iz;
} batch_result_t;

#define __BATCH_REJECTED_PROXY   1
#define __BATCH_REJECTED_MESH    2
#define __BATCH_REJECTED_RESCALE 3
#define __BATCH_ACCEPTED         4

/* --- Run context --- */

typedef struct {
  real_t* sorted_radii;
  sif_catalog_t* cat;
  sif_bitmask_t* mask;
  sif_cell_linked_list_t* void_cll;
  sif_chain_mesh_t* mesh;
  fft_workspace_t* fft_ws;
  candidate_buffer_t candidates;

  batch_result_t* batch_results;
  uint64_t* batch_indices;

  shell_buffer_t* shells; /* one per thread */
  int n_threads;
} rescaled_ctx_t;

static void __ctx_release(rescaled_ctx_t* ctx, sif_grid_t* grid) {
  if (!ctx)
    return;

  if (ctx->fft_ws) {
    real_t* recovered = fft_workspace_take_real_buffer(ctx->fft_ws);
    if (recovered)
      grid->delta = recovered;
    fft_workspace_free(ctx->fft_ws);
    ctx->fft_ws = NULL;
  }

  if (ctx->shells) {
    for (int t = 0; t < ctx->n_threads; t++)
      sif_free_aligned(ctx->shells[t].data);
    sif_free_aligned(ctx->shells);
    ctx->shells = NULL;
  }

  sif_free_aligned(ctx->batch_results);
  sif_free_aligned(ctx->batch_indices);
  sif_candidate_buffer_free(&ctx->candidates);
  sif_chain_mesh_free(ctx->mesh);
  sif_cell_linked_list_free(ctx->void_cll);
  sif_bitmask_free(ctx->mask);
  sif_free_aligned(ctx->sorted_radii);

  /* NULL on the success path: the caller took the catalog. */
  sif_catalog_free(ctx->cat);

  ctx->batch_results = NULL;
  ctx->batch_indices = NULL;
  ctx->mesh = NULL;
  ctx->void_cll = NULL;
  ctx->mask = NULL;
  ctx->sorted_radii = NULL;
  ctx->cat = NULL;
}

static int __ctx_init(rescaled_ctx_t* ctx, sif_grid_t* grid,
  const sif_field_t* field, const real_t* radii, uint32_t n_radii) {

  memset(ctx, 0, sizeof(*ctx));

  ctx->n_threads = system_get_max_threads();
  if (ctx->n_threads < 1)
    ctx->n_threads = 1;

  ctx->sorted_radii = sif_malloc_aligned(n_radii * sizeof(real_t));
  if (!ctx->sorted_radii) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the radii array");
    return SIF_ERR_ALLOC;
  }
  memcpy(ctx->sorted_radii, radii, n_radii * sizeof(real_t));
  __sort_radii_desc(ctx->sorted_radii, n_radii);

  ctx->cat = sif_catalog_alloc(__CATALOG_INITIAL_CAPACITY);
  if (!ctx->cat)
    return SIF_ERR_ALLOC;

  ctx->mask = sif_bitmask_alloc(grid->total_cells);
  if (!ctx->mask)
    return SIF_ERR_ALLOC;

  ctx->void_cll = sif_cell_linked_list_alloc(
    __VOID_CLL_CELLS, grid->box_length, ctx->cat->capacity, SIF_PBC_PERIODIC);
  if (!ctx->void_cll)
    return SIF_ERR_ALLOC;

  uint32_t mesh_n_cells = (uint32_t)(grid->box_length / 4.0f);
  if (mesh_n_cells < 16)
    mesh_n_cells = 16;
  else if (mesh_n_cells > 256)
    mesh_n_cells = 256;

  ctx->mesh = sif_chain_mesh_alloc(
    mesh_n_cells, grid->box_length, field, false, false, false);
  if (!ctx->mesh) {
    SIF_LOG_ERROR(__TAG, "failed to build the particle chain mesh");
    return SIF_ERR_ALLOC;
  }

  ctx->batch_results =
    sif_malloc_aligned(__BATCH_SIZE * sizeof(batch_result_t));
  ctx->batch_indices = sif_malloc_aligned(__BATCH_SIZE * sizeof(uint64_t));
  if (!ctx->batch_results || !ctx->batch_indices)
    return SIF_ERR_ALLOC;

  /* One shell scratch per thread, grown on demand. This used to be a fixed
   * 256 MiB per thread regardless of the actual particle load. */
  ctx->shells =
    sif_calloc_aligned((size_t)ctx->n_threads, sizeof(shell_buffer_t));
  if (!ctx->shells)
    return SIF_ERR_ALLOC;

  for (int t = 0; t < ctx->n_threads; t++) {
    if (__shell_reserve(&ctx->shells[t], __SHELL_INITIAL_CAPACITY) != SIF_OK) {
      SIF_LOG_ERROR(__TAG, "failed to allocate the per-thread shell buffers");
      return SIF_ERR_ALLOC;
    }
  }

  system_state_t* state = get_system_state();
  ctx->fft_ws = fft_workspace_alloc(state->fft_mgr, grid->n_cells);
  if (!ctx->fft_ws) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the FFT workspace");
    return SIF_ERR_ALLOC;
  }

  fft_grid_forward(ctx->fft_ws, grid);

  sif_free_aligned(grid->delta);
  grid->delta = NULL;

  if (fft_workspace_init_backward(ctx->fft_ws, state->fft_mgr) != SIF_OK) {
    SIF_LOG_ERROR(__TAG, "failed to initialize the backward FFT");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

static int __accept_void(rescaled_ctx_t* ctx, const sif_grid_t* grid, real_t cx,
  real_t cy, real_t cz, real_t r) {

  int status = sif_catalog_append(ctx->cat, cx, cy, cz, r);
  if (status != SIF_OK)
    return status;

  status =
    sif_cell_linked_list_ensure_capacity(ctx->void_cll, ctx->cat->n_voids);
  if (status != SIF_OK)
    return status;

  status = sif_cell_linked_list_insert(
    ctx->void_cll, ctx->cat->n_voids - 1, cx, cy, cz);
  if (status != SIF_OK)
    return status;

  __mark_sphere(ctx->mask, cx, cy, cz, r, grid->n_cells, grid->p2_mask,
    grid->cell_length);

  return SIF_OK;
}

/* --- Driver --- */

sif_catalog_t* sif_finder_rescaled_spherical(sif_grid_t* grid,
  sif_field_t* field, const real_t* radii, uint32_t n_radii, real_t threshold,
  real_t overlap_fraction, sif_option_t options) {

  if (!grid || !grid->delta || !field || !radii || n_radii == 0) {
    SIF_LOG_ERROR(__TAG, "invalid grid, field or radii");
    return NULL;
  }

  rescaled_ctx_t ctx;
  if (__ctx_init(&ctx, grid, field, radii, n_radii) != SIF_OK) {
    __ctx_release(&ctx, grid);
    return NULL;
  }

  const uint8_t require_min = (options & SIF_FINDER_CENTER_IS_MINIMUM) ? 1 : 0;
  const real_t max_radius = ctx.sorted_radii[0];

  /* Loop invariants: the mean tracer density never changes between radii. */
  const real_t box_volume =
    grid->box_length * grid->box_length * grid->box_length;
  const real_t mean_density = (real_t)field->n_particles / box_volume;
  const real_t vol_factor = (4.0f / 3.0f) * M_PI * mean_density;

  sif_timer_t timer;
  int failed = 0;

  for (uint32_t i = 0; i < n_radii && !failed; i++) {
    sif_timer_start(&timer);

    const real_t radius = ctx.sorted_radii[i];
    finder_radius_stats_t stats = {0};

    if (fft_apply_filter(ctx.fft_ws, FILTER_TOP_HAT, radius,
          grid->box_length) != SIF_OK) {
      failed = 1;
      break;
    }
    grid->delta = fft_grid_backward(ctx.fft_ws);

    if (sif_finder_scan_candidates(grid, ctx.mask, threshold, require_min,
          &ctx.candidates) != SIF_OK) {
      failed = 1;
      break;
    }
    stats.n_candidates = ctx.candidates.count;

    const real_t rmin = __RMIN_FACTOR * radius;
    const real_t r_buffer = REAL_MAX(radius, 3.0f * grid->cell_length);
    const real_t r_search = radius + r_buffer;

    mesh_template_t tpl;
    if (__template_build(&tpl, ctx.mesh->cell_length, rmin, r_search) !=
        SIF_OK) {
      failed = 1;
      break;
    }

    const int32_t proxy_pole =
      (int32_t)(radius * (1.0f - overlap_fraction) / grid->cell_length) - 1;

    uint64_t k = 0;
    while (k < ctx.candidates.count && !failed) {
      uint64_t batch_count = 0;

      /* Phase 0: skip already-masked candidates sequentially. Doing this
       * outside the parallel region avoids paying thread sync for millions of
       * trivially dead candidates. */
      while (k < ctx.candidates.count && batch_count < __BATCH_SIZE) {
        if (sif_bitmask_get(ctx.mask, ctx.candidates.items[k].flat_idx)) {
          stats.rejected_masked++;
        } else {
          ctx.batch_indices[batch_count++] = k;
        }
        k++;
      }

      if (batch_count == 0)
        break;

      /* Phase 1: evaluate the batch in parallel against the catalog as it
       * stood when the batch started. Read-only on mask and catalog.
       *
       * The team size is pinned so that omp_get_thread_num() can never index
       * past the per-thread shell buffers allocated in __ctx_init. */
#pragma omp parallel for schedule(dynamic, 16) num_threads(ctx.n_threads)
      for (uint64_t b = 0; b < batch_count; b++) {
        const int tid = system_get_thread_num();
        const uint64_t flat =
          ctx.candidates.items[ctx.batch_indices[b]].flat_idx;
        batch_result_t* res = &ctx.batch_results[b];

        uint32_t ix, iy, iz;
        sif_unflatten_index(grid, flat, &ix, &iy, &iz);

        res->ix = ix;
        res->iy = iy;
        res->iz = iz;
        res->proxy_pole = proxy_pole;

        if (proxy_pole > 0 &&
            __check_overlap_cells(ctx.mask, grid->n_cells, grid->p2_mask, ix,
              iy, iz, (uint32_t)proxy_pole)) {
          res->status = __BATCH_REJECTED_PROXY;
          continue;
        }

        real_t cx = (real_t)ix * grid->cell_length;
        real_t cy = (real_t)iy * grid->cell_length;
        real_t cz = (real_t)iz * grid->cell_length;

        if (options & SIF_FINDER_REFINE_CENTER_HESSIAN)
          __refine_center_hessian(grid, ix, iy, iz, &cx, &cy, &cz);

        res->cx = cx;
        res->cy = cy;
        res->cz = cz;

        if (__check_overlap_mesh(ctx.cat, cx, cy, cz, radius, max_radius,
              grid->box_length, ctx.void_cll, grid->p2_mask,
              overlap_fraction)) {
          res->status = __BATCH_REJECTED_MESH;
          continue;
        }

        const real_t r_scaled = __find_exact_radius(ctx.mesh, cx, cy, cz,
          r_search, threshold, vol_factor, &ctx.shells[tid], rmin, &tpl);

        if (r_scaled < 0.0f) {
          res->status = __BATCH_REJECTED_RESCALE;
          continue;
        }

        res->r_scaled = r_scaled;
        res->status = __BATCH_ACCEPTED;
      }

      /* Phase 2: commit sequentially, re-checking everything that the parallel
       * phase could not have seen (voids accepted earlier in this same
       * batch). */
      for (uint64_t b = 0; b < batch_count; b++) {
        const batch_result_t* res = &ctx.batch_results[b];

        switch (res->status) {
        case __BATCH_REJECTED_PROXY: stats.rejected_proxy++; continue;
        case __BATCH_REJECTED_MESH: stats.rejected_mesh++; continue;
        case __BATCH_REJECTED_RESCALE: stats.rejected_rescale++; continue;
        default: break;
        }

        const uint64_t flat =
          ctx.candidates.items[ctx.batch_indices[b]].flat_idx;

        if (sif_bitmask_get(ctx.mask, flat)) {
          stats.rejected_masked++;
          continue;
        }

        if (res->proxy_pole > 0 &&
            __check_overlap_cells(ctx.mask, grid->n_cells, grid->p2_mask,
              res->ix, res->iy, res->iz, (uint32_t)res->proxy_pole)) {
          stats.rejected_proxy++;
          continue;
        }

        /* The rescaled radius differs from the proxy, so the shrunk-sphere
         * probe has to be repeated at the true size. */
        const int32_t exact_pole =
          (int32_t)(res->r_scaled * (1.0f - overlap_fraction) /
                    grid->cell_length) -
          1;
        if (exact_pole > 0 && exact_pole != res->proxy_pole &&
            __check_overlap_cells(ctx.mask, grid->n_cells, grid->p2_mask,
              res->ix, res->iy, res->iz, (uint32_t)exact_pole)) {
          stats.rejected_exact++;
          continue;
        }

        if (__check_overlap_mesh(ctx.cat, res->cx, res->cy, res->cz,
              res->r_scaled, max_radius, grid->box_length, ctx.void_cll,
              grid->p2_mask, overlap_fraction)) {
          stats.rejected_exact++;
          continue;
        }

        if (__accept_void(&ctx, grid, res->cx, res->cy, res->cz,
              res->r_scaled) != SIF_OK) {
          SIF_LOG_ERROR(__TAG, "failed to store an accepted void, aborting");
          failed = 1;
          break;
        }
        stats.accepted++;
      }
    }

    __template_free(&tpl);

    sif_timer_stop(&timer);
    sif_finder_log_radius(__TAG, radius, &stats, ctx.cat->n_voids,
      sif_timer_elapsed_ms(&timer) / 1000.0);
  }

  if (!failed && (options & SIF_FINDER_PRESERVE_GRID)) {
    if (fft_apply_filter(ctx.fft_ws, FILTER_NONE, 0, 0) == SIF_OK) {
      grid->delta = fft_grid_backward(ctx.fft_ws);
      SIF_LOG_INFO(__TAG, "recovered original density grid");
    }
  }

  sif_catalog_trim(ctx.cat);

  sif_catalog_t* result = ctx.cat;
  ctx.cat = NULL; /* ownership passes to the caller */
  __ctx_release(&ctx, grid);

  return result;
}
