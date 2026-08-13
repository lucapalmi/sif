/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/finder/spherical_finder.h"

#include <stdlib.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/cell_linked_list.h"

#include "core/system_internal.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"
#include "sif/utils/timer.h"

#include "utils.h"

SIF_DEFINE_QUICKSORT(sort_radii_desc, sif_real, a > b)

#define TAG "spherical_finder"

/* Coarse grid for the void-vs-void overlap index. */
#define VOID_CLL_CELLS 32

/* Initial catalog capacity. It grows on demand, this only avoids the first
 * few reallocations on a typical run. */
#define CATALOG_INITIAL_CAPACITY 250000

/*
 * Everything the finder owns for the duration of a run. Grouping it lets a
 * single teardown handle every exit path, so the body can just return.
 */
typedef struct {
  sif_real* sorted_radii;
  sif_catalog_t* cat;
  sif_bitmask_t* mask;
  sif_cell_linked_list_t* cll;
  sif_fft_workspace_t* fft_ws;
  sif_candidate_buffer_t candidates;
} spherical_ctx_t;

static void ctx_release(spherical_ctx_t* ctx, sif_grid_t* grid) {
  if (!ctx)
    return;

  /* A real-space buffer only exists once the caller's original delta has been
   * released, so handing it back is what keeps it from leaking. Before that
   * point there is nothing to take and grid->values must stay untouched. */
  if (ctx->fft_ws) {
    sif_real* recovered = sif__fft_workspace_take_real_buffer(ctx->fft_ws);
    if (recovered)
      grid->values = recovered;
    sif__fft_workspace_free(ctx->fft_ws);
    ctx->fft_ws = NULL;
  }

  sif__candidate_buffer_free(&ctx->candidates);
  sif_cell_linked_list_free(ctx->cll);
  sif_bitmask_free(ctx->mask);
  sif_free_aligned(ctx->sorted_radii);

  /* NULL on the success path: the caller took the catalog. */
  sif_catalog_free(ctx->cat);

  ctx->cll = NULL;
  ctx->mask = NULL;
  ctx->sorted_radii = NULL;
  ctx->cat = NULL;
}

static int ctx_init(spherical_ctx_t* ctx, sif_grid_t* grid,
  const sif_real* radii, uint32_t n_radii) {

  memset(ctx, 0, sizeof(*ctx));

  ctx->sorted_radii = sif_malloc_aligned(n_radii * sizeof(sif_real));
  if (!ctx->sorted_radii) {
    SIF_LOG_ERROR(TAG, "failed to allocate the radii array");
    return SIF_ERR_ALLOC;
  }
  memcpy(ctx->sorted_radii, radii, n_radii * sizeof(sif_real));
  sort_radii_desc(ctx->sorted_radii, n_radii);

  ctx->cat = sif_catalog_alloc(CATALOG_INITIAL_CAPACITY);
  if (!ctx->cat)
    return SIF_ERR_ALLOC;

  ctx->mask = sif_bitmask_alloc(grid->total_cells);
  if (!ctx->mask)
    return SIF_ERR_ALLOC;

  /* sif__overlap_exact always wraps its query with PBC, so the list has to
   * bin with PBC too. */
  ctx->cll = sif_cell_linked_list_alloc(
    VOID_CLL_CELLS, grid->box_length, ctx->cat->capacity, SIF_PBC_PERIODIC);
  if (!ctx->cll)
    return SIF_ERR_ALLOC;

  sif_system_state_t* state = sif__system_state();
  ctx->fft_ws = sif__fft_workspace_alloc(state->fft_mgr, grid->n_cells);
  if (!ctx->fft_ws) {
    SIF_LOG_ERROR(TAG, "failed to allocate the FFT workspace");
    return SIF_ERR_ALLOC;
  }

  /* Must succeed before the field is released: on failure the caller's grid
   * has to come back untouched. */
  if (sif__fft_grid_forward(ctx->fft_ws, grid) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "the forward FFT failed");
    return SIF_ERR_ALLOC;
  }

  /* The density field is no longer needed: from here on the finder works out
   * of the FFT workspace's real-space buffer, which saves a full grid. */
  sif_free_aligned(grid->values);
  grid->values = NULL;

  if (sif__fft_workspace_init_backward(ctx->fft_ws, state->fft_mgr) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "failed to initialize the backward FFT");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

/*
 * Commits one accepted void: appends it, indexes it and marks its footprint.
 */
static int accept_void(spherical_ctx_t* ctx, const sif_grid_t* grid,
  sif_real cx, sif_real cy, sif_real cz, sif_real r) {

  int status = sif_catalog_append(ctx->cat, cx, cy, cz, r);
  if (status != SIF_OK)
    return status;

  status = sif_cell_linked_list_ensure_capacity(ctx->cll, ctx->cat->n_voids);
  if (status != SIF_OK)
    return status;

  status =
    sif_cell_linked_list_insert(ctx->cll, ctx->cat->n_voids - 1, cx, cy, cz);
  if (status != SIF_OK)
    return status;

  sif__mark_sphere(
    ctx->mask, cx, cy, cz, r, grid->n_cells, grid->p2_mask, grid->cell_length);

  return SIF_OK;
}

sif_catalog_t* sif_finder_spherical(sif_grid_t* grid, const sif_real* radii,
  uint32_t n_radii, sif_real threshold, sif_real overlap_fraction,
  sif_option options) {

  if (!grid || !grid->values || !radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "invalid grid or radii");
    return NULL;
  }

  /* A mass grid handed to something that expects a density contrast produces
   * numbers rather than an error. SIF_GRID_EMPTY is not flagged: that is a
   * grid the caller filled directly, and only the caller knows what is in it.
   */
  if (grid->content == SIF_GRID_MASS) {
    SIF_LOG_WARNING(TAG,
      "this grid still holds masses; call sif_grid_to_density_contrast first");
  }

  spherical_ctx_t ctx;
  if (ctx_init(&ctx, grid, radii, n_radii) != SIF_OK) {
    ctx_release(&ctx, grid);
    return NULL;
  }

  const sif_real max_radius = ctx.sorted_radii[0];

  sif_timer_t timer;
  int failed = 0;

  for (uint32_t i = 0; i < n_radii && !failed; i++) {
    sif_timer_start(&timer);

    const sif_real radius = ctx.sorted_radii[i];
    sif_finder_radius_stats_t stats = {0};

    if (sif__fft_apply_filter(ctx.fft_ws, SIF__FILTER_TOP_HAT, radius,
          grid->box_length) != SIF_OK) {
      failed = 1;
      break;
    }
    grid->values = sif__fft_grid_backward(ctx.fft_ws);

    if (sif__finder_scan_candidates(
          grid, ctx.mask, threshold, &ctx.candidates) != SIF_OK) {
      failed = 1;
      break;
    }
    stats.n_candidates = ctx.candidates.count;

    /* The proxy sphere is shrunk by overlap_fraction so that permitted
     * overlaps are not rejected by the cheap probe. */
    const int32_t proxy_pole =
      (int32_t)(radius * (1.0f - overlap_fraction) / grid->cell_length) - 1;

    for (uint64_t k = 0; k < ctx.candidates.count; k++) {
      const uint64_t flat = ctx.candidates.items[k].flat_idx;

      if (sif_bitmask_get(ctx.mask, flat)) {
        stats.rejected_masked++;
        continue;
      }

      uint32_t ix, iy, iz;
      sif__unflatten_index(grid, flat, &ix, &iy, &iz);

      if (proxy_pole > 0 &&
          sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask, ix, iy, iz,
            (uint32_t)proxy_pole)) {
        stats.rejected_proxy++;
        continue;
      }

      const sif_real cx = (sif_real)ix * grid->cell_length;
      const sif_real cy = (sif_real)iy * grid->cell_length;
      const sif_real cz = (sif_real)iz * grid->cell_length;

      if (sif__overlap_exact(ctx.cat, cx, cy, cz, radius, max_radius,
            grid->box_length, ctx.cll, grid->p2_mask, overlap_fraction)) {
        stats.rejected_mesh++;
        continue;
      }

      if (accept_void(&ctx, grid, cx, cy, cz, radius) != SIF_OK) {
        SIF_LOG_ERROR(TAG, "failed to store an accepted void, aborting");
        failed = 1;
        break;
      }
      stats.accepted++;
    }

    sif_timer_stop(&timer);
    sif__finder_log_radius(TAG, radius, &stats, ctx.cat->n_voids,
      sif_timer_elapsed_ms(&timer) / 1000.0);
  }

  if (!failed && !(options & SIF_FINDER_CONSUME_GRID)) {
    if (sif__fft_apply_filter(ctx.fft_ws, SIF__FILTER_NONE, 0, 0) == SIF_OK) {
      grid->values = sif__fft_grid_backward(ctx.fft_ws);
      SIF_LOG_INFO(TAG, "recovered original density grid");
    }
  }

  sif_catalog_trim(ctx.cat);

  sif_catalog_t* result = ctx.cat;
  ctx.cat = NULL; /* ownership passes to the caller */
  ctx_release(&ctx, grid);

  return result;
}
