/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The exodus finder.
 *
 * The grid locates voids; the tracers size them. A candidate centre comes from
 * the smoothed density field exactly as in the plain finder, but its radius is
 * then measured against the particles: the void is grown outward from the
 * centre until the enclosed number density first rises to the threshold, and
 * that crossing radius is what goes in the catalogue. So a radius is a property
 * of the tracer distribution rather than of the ladder that happened to find
 * it, and two runs with different ladders describe the same voids.
 *
 * What the rung still fixes is the centre and the search range. A cell only
 * becomes a candidate at rungs at or below its own crossing radius, so the
 * rung is a lower bound on the answer and r_search, about twice the rung, is
 * the upper one -- the rung brackets the radius, it does not set it.
 *
 *
 * Finding the crossing radius
 * ---------------------------
 *
 * The condition is on the *enclosed* density, so the quantity of interest is
 * n(<d) / (rho_mean * V(d)) - 1 <= threshold, evaluated at every particle
 * distance d in turn. On a weighted mesh n(<d) is the weight enclosed and
 * rho_mean the mean weight density, and nothing below changes but the sums.
 * The kernel is in exodus_rescale.c. Written out, that asks for the k-th
 * nearest distance for every k, which is a full sort of everything inside the
 * search sphere: at r_search = 200 in a 2250 box, ten million distances sorted
 * per candidate, and there are millions of candidates.
 *
 * Three observations collapse that cost.
 *
 * First, the answer is the *outermost* crossing, so the walk starts at
 * r_search and moves inward and stops at the first radius that satisfies the
 * condition. Nothing inside that radius is ever examined.
 *
 * Second, deciding which shell holds the crossing needs only counts, not
 * distances. A histogram of squared distances gives, for any bin, the number of
 * particles inside its lower edge -- and the most permissive test a bin could
 * possibly pass is its outermost particle carrying the smallest interior count.
 * If that fails, no particle in the bin can qualify and the whole bin is
 * skipped without a single distance being looked at. That test is what makes
 * the walk cheap: it is a proof about a shell, obtained from a counter.
 *
 * Third, once a bin does survive the skip test, the exact radius has to come
 * from particles after all -- but only from that bin. It is re-scanned, its
 * distances sorted, and walked outside-in. A bin is a far thinner shell than a
 * mesh cell, so the re-scan visits few cells; see RESOLVE_* for why several
 * neighbouring bins are resolved together rather than one at a time.
 *
 * Squared distances throughout, and bins uniform in d^2, so no square root is
 * evaluated until the answer is known. The density condition squares cleanly
 * because both sides are positive:
 *
 *   n_in / (vol_factor * d^3) - 1 <= threshold
 *     <=>  n_in^2 <= ((1 + threshold) * vol_factor)^2 * (d^2)^3
 *
 * Bins uniform in d^2 are also finer in radius further out, which is the half
 * of the range the walk actually spends its time in.
 *
 *
 * Why it is speculative
 * ---------------------
 *
 * Rescaling one candidate is independent of every other, so candidates are
 * evaluated in parallel -- but accepting a void masks the region around it and
 * invalidates candidates that were being evaluated at the same time. The run
 * therefore evaluates a batch against a frozen catalogue, then commits
 * sequentially and re-tests each acceptance against the catalogue as it now
 * stands. Commit order is unchanged by the batching, so the catalogue does not
 * depend on the batch size or the thread count; see BATCH_MAX.
 *
 *
 * Where a centre sits
 * -------------------
 *
 * Two kinds of candidates, the SIF_FINDER_SEEDS_* options.
 *
 * Cells, the finder of 0.4: every cell below the threshold at a rung is a
 * candidate, deepest first, centred on its node. Where a minimum is refused
 * the cell next to it can be accepted instead, a void on the slope of a
 * larger one, and the catalogue depends on the grid.
 *
 * Minima: the candidates are the local minima of the rung's field after a
 * Gaussian of RUNG_MINIMA_SIGMA of the rung -- one more transform -- each
 * placed between the nodes by one Newton step on the 27 cells around it, see
 * sif__finder_scan_refined(). A smoothing tied to the rung rather than to the
 * cell is what keeps the minima the same on every grid that resolves them.
 * A crossing past the rung's reach is then measured with the reach extended
 * rather than refused, and a void displaces the smaller ones it overlaps
 * instead of being refused by them; see find_conflicts().
 *
 *
 * Surveys
 * -------
 *
 * sif_finder_exodus_survey() runs this same driver with the box mean replaced
 * by a random catalogue: the candidates are scanned on the smoothed data over
 * the smoothed randoms, and each radius is grown against the random weight a
 * sphere encloses instead of against its volume. See "Survey rescaling" in
 * exodus_rescale.c for the kernel, and survey_check_margin() in
 * exodus_survey.c for why nothing else had to learn that the survey is not
 * periodic.
 */

#include "sif/finder/exodus_finder.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/chain_mesh.h"

#include "core/system_internal.h"
#include "structures/field_internal.h"
#include "utils/logger_internal.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"
#include "sif/utils/timer.h"

#include "exodus_checkpoint.h"
#include "exodus_internal.h"
#include "utils.h"
SIF_DEFINE_QUICKSORT(sort_radii_desc, sif_real, a > b)

#define TAG "exodus_finder"

/*
 * Speculation window.
 *
 * A batch is evaluated in parallel against the catalogue as it stood when the
 * batch began, so every void accepted part-way through invalidates the work
 * already done on the candidates behind it. Candidates arrive sorted by depth
 * and the deepest cells cluster inside the same underdensity, so a long window
 * spends most of its effort on candidates that the first accept will bury.
 *
 * The right window is therefore "however many candidates usually pass before
 * something is accepted", which is not a constant: it varies with the tracer
 * density, the threshold, the thread count, and from one radius to the next
 * within a single run. So it is measured rather than chosen. A batch that
 * accepts nothing cost nothing to speculate on and widens the window by one
 * floor's worth; a batch that accepts divides it down, the more accepts the
 * harder. Probing up slowly and backing off fast is what keeps the window off
 * the ceiling -- measured end to end, the result is insensitive to BATCH_MAX,
 * and beats the best single fixed window precisely because no fixed window
 * suits every radius.
 *
 * This only changes how candidates are grouped, never the order in which they
 * are committed, so the catalogue is identical for any window.
 */
#define BATCH_MAX 4096u

/* Below a few candidates per thread the parallel region stops paying for
 * itself, whatever the accept rate says. */
#define BATCH_PER_THREAD 8u
#define BATCH_FLOOR      32u

/* Coarse grid for the void-vs-void overlap index. */
#define VOID_CLL_CELLS 32

#define CATALOGUE_INITIAL_CAPACITY 250000

/*
 * A void is localised by the rung that found it: its centre is the deepest
 * cell of the field smoothed at that rung's radius. When a void turns out to
 * be much larger than the rung, the smoothed field is flat across its whole
 * interior and the "deepest cell" is chosen out of noise -- the void is still
 * found, and with about the right radius, but pinned at an arbitrary point
 * inside itself.
 *
 * Nothing about that is visible in the catalogue, so it is reported instead.
 * Beyond this ratio a void was found at a rung more than twice too small,
 * which is only reachable when the ladder-gap guard had to widen the search;
 * a ladder dense enough that every void meets a rung near its own scale never
 * trips it.
 */
#define MISMATCH_RATIO 2.0f

/* --- Extended reach --- */

/* A crossing past a rung's reach is measured with the reach extended in
 * steps of EXT_GROWTH up to the top rung's, so that the cost follows the
 * void's own crossing rather than the top of the ladder. Level k's core ends
 * EXT_SLACK below level k - 1's reach, which keeps it inside a radius already
 * known to be underdense whichever rung enters the chain at k. */
#define EXT_GROWTH     1.5f
#define EXT_SLACK      1.2f
#define EXT_MAX_LEVELS 32u

/* Wall-clock seconds, for the per-rung breakdown in the trace log. */
static double wall_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* --- Speculative batch evaluation --- */

typedef struct {
  uint8_t status;         /* one of BATCH_* below */
  uint8_t rescale_reason; /* a RESCALE_*, set only when the rescaling failed */
  sif_real r_scaled;
  sif_real cx, cy, cz;
  uint32_t ix, iy, iz;
} batch_result_t;

#define BATCH_REJECTED_PROXY   1
#define BATCH_REJECTED_MESH    2
#define BATCH_REJECTED_RESCALE 3
#define BATCH_ACCEPTED         4
#define BATCH_REJECTED_WITNESS 5

/* --- Run context --- */

typedef struct {
  sif_real* sorted_radii;
  sif_catalogue_t* cat;
  sif_bitmask_t* mask;
  sif_cell_linked_list_t* void_cll;
  const sif_chain_mesh_t* mesh; /* borrowed from the caller, never freed */
  sif_fft_workspace_t* fft_ws;

  /*
   * A rung's candidates: under SIF_FINDER_SEEDS_CELLS every cell below the
   * threshold, on its node -- the plain scan of 0.4; under
   * SIF_FINDER_SEEDS_MINIMA the local minima of the rung's field smoothed at
   * its scale, each placed between the nodes with its offset in `offsets`,
   * and `below` the scan's scratch. Cells are the finder of 0.4, which
   * refuses a crossing past the reach and a void that overlaps any accepted
   * one; minima extend the reach and displace.
   */
  int seed_minima;

  /* Under the minima, the accepted voids a new one collides with,
   * and how many voids the rung has taken back out of the catalogue: their
   * radii are negated until the rung ends and the catalogue is compacted. */
  uint64_t* conflicts;
  uint64_t n_conflicts, conflicts_capacity;
  uint64_t n_removed;

  sif_candidate_buffer_t candidates;
  sif_offset_buffer_t offsets;
  sif_bitmask_t* below;

  batch_result_t* batch_results;
  uint64_t* batch_indices;

  radial_scratch_t* scratch; /* one per thread */
  int n_threads;

  /*
   * Largest radius in the catalogue so far.
   *
   * sif__overlap_exact() sizes its search box from this, so it has to be the
   * real maximum and not the largest radius on the ladder. Rescaling grows a
   * void out to r_search, about twice the rung it was found at, so the ladder
   * maximum understates the catalogue by up to a factor of two -- and a void
   * whose centre falls outside the search box is never compared against, which
   * lets a genuine overlap through. Overestimating only widens the box.
   */
  sif_real max_accepted_r;

  /*
   * What a survey run adds, all NULL for a box. The random grid is smoothed
   * alongside the data one in a workspace of its own, and `occupied` marks the
   * observed grid cells: those the randoms reach, with their chance holes
   * closed (sif__exodus_survey_occupancy()).
   */
  sif_grid_t* random_grid;
  const sif_chain_mesh_t* random_mesh; /* borrowed, never freed */
  sif_fft_workspace_t* random_ws;
  sif_bitmask_t* occupied;
} exodus_ctx_t;

/* The node the rung's k-th candidate sits on. */
static inline uint64_t candidate_node(const exodus_ctx_t* ctx, uint64_t k) {
  return ctx->candidates.items[k].flat_idx;
}

/*
 * The rung's k-th candidate centre, from its node and its offset off that
 * node -- none for a candidate that was not refined, where this is
 * i * cell_length exactly as it always was. An offset can carry a centre on
 * node 0 or n - 1 across the face, and it is wrapped back into the box; that
 * only happens in a periodic box, since a survey's observed nodes are a margin
 * away from the faces.
 */
static inline void candidate_centre(const exodus_ctx_t* ctx, uint64_t k,
  const sif_grid_t* grid, uint32_t ix, uint32_t iy, uint32_t iz,
  sif_real c[3]) {

  const uint32_t node[3] = {ix, iy, iz};
  sif_real off[3] = {0.0f, 0.0f, 0.0f};
  const uint32_t refined = ctx->candidates.items[k].refined;
  if (refined) {
    const sif_offset_t* o = &ctx->offsets.items[refined - 1];
    off[0] = o->dx;
    off[1] = o->dy;
    off[2] = o->dz;
  }

  const sif_real box = grid->box_length;
  for (int a = 0; a < 3; a++) {
    sif_real x = ((sif_real)node[a] + off[a]) * grid->cell_length;
    if (x < 0.0f)
      x += box;
    if (x >= box)
      x -= box;
    c[a] = x;
  }
}

/*
 * Whether the rung's k-th candidate, at radius r, certainly overlaps an
 * accepted void because its node is masked -- the cheap rejection, taken
 * before anything is measured.
 *
 * A masked node lies within r_acc of an accepted centre, so a centre on the
 * node is within r_acc too, and two voids collide below r + r_acc - f *
 * min(r, r_acc), which is at least r_acc. A centre refined off its node is a
 * further |off| away, and the mask then proves a collision only while
 * |off| < r - f * min(r, r_acc) -- of which r * (1 - f) is the bound that
 * holds whatever r_acc is. Short of that, near f = 1, the candidate goes on to
 * the exact test rather than being dropped on its node's word.
 *
 * The collision distance grows with r, so a rejection at the rung holds at
 * every radius the rescaling can return.
 */
static inline int masked_overlap(const exodus_ctx_t* ctx, uint64_t k,
  sif_real r, sif_real overlap_fraction, sif_real cell_length) {

  const sif_candidate_t* c = &ctx->candidates.items[k];
  if (!sif_bitmask_get(ctx->mask, c->flat_idx))
    return 0;
  if (!c->refined)
    return 1;

  const sif_offset_t* o = &ctx->offsets.items[c->refined - 1];
  const sif_real off =
    SIF_REAL_SQRT(o->dx * o->dx + o->dy * o->dy + o->dz * o->dz) * cell_length;
  return off < r * (1.0f - overlap_fraction);
}

static void ctx_release(exodus_ctx_t* ctx, sif_grid_t* grid) {
  if (!ctx)
    return;

  if (ctx->fft_ws) {
    sif_real* recovered = sif__fft_workspace_take_real_buffer(ctx->fft_ws);
    if (recovered)
      grid->values = recovered;
    sif__fft_workspace_free(ctx->fft_ws);
    ctx->fft_ws = NULL;
  }

  if (ctx->random_ws) {
    sif_real* recovered = sif__fft_workspace_take_real_buffer(ctx->random_ws);
    if (recovered)
      ctx->random_grid->values = recovered;
    sif__fft_workspace_free(ctx->random_ws);
    ctx->random_ws = NULL;
  }

  sif_bitmask_free(ctx->occupied);
  ctx->occupied = NULL;

  if (ctx->scratch) {
    for (int t = 0; t < ctx->n_threads; t++)
      sif__exodus_scratch_free(&ctx->scratch[t]);
    sif_free_aligned(ctx->scratch);
    ctx->scratch = NULL;
  }

  sif_free_aligned(ctx->batch_results);
  sif_free_aligned(ctx->batch_indices);
  free(ctx->conflicts);
  sif__candidate_buffer_free(&ctx->candidates);
  sif__offset_buffer_free(&ctx->offsets);
  sif_bitmask_free(ctx->below);
  sif_cell_linked_list_free(ctx->void_cll);
  sif_bitmask_free(ctx->mask);
  sif_free_aligned(ctx->sorted_radii);

  /* NULL on the success path: the caller took the catalogue. */
  sif_catalogue_free(ctx->cat);

  ctx->batch_results = NULL;
  ctx->batch_indices = NULL;
  ctx->mesh = NULL;
  ctx->void_cll = NULL;
  ctx->below = NULL;
  ctx->mask = NULL;
  ctx->sorted_radii = NULL;
  ctx->cat = NULL;
}

/* The grid half of ctx_init() for one grid: transform, correct the window,
 * release the real-space values, and get ready to smooth. With save_memory
 * the grid goes to disk instead of being transformed, and each smoothing
 * transforms it afresh (sif__fft_workspace_stream()). */
static int ctx_init_grid(sif_fft_workspace_t** ws_out, sif_grid_t* grid,
  sif_real min_radius, sif_option opt, int save_memory) {

  sif_system_state_t* state = sif__system_state();
  if (!state) {
    *ws_out = NULL;
    return SIF_ERR_UNINITIALISED;
  }

  /* Must succeed before the field is released: on failure the caller's grid
   * has to come back untouched. */
  int status;
  if (save_memory) {
    status = sif__fft_workspace_stream(state->fft_mgr, grid, ws_out);
    if (status != SIF_OK)
      return status;
  } else {
    sif_fft_workspace_t* ws =
      sif__fft_workspace_alloc(state->fft_mgr, grid->n_cells);
    *ws_out = ws;
    if (!ws)
      return SIF_ERR_ALLOC;

    status = sif__fft_grid_forward(ws, grid);
    if (status != SIF_OK)
      return status;
  }

  /* Before the first filter and after the transform, which is the only window
   * in which the assignment window is separable from the smoothing one. */
  if (sif__finder_deconvolve_window(TAG, *ws_out, grid, min_radius, opt) !=
      SIF_OK)
    return SIF_ERR_INVALID;

  sif_free_aligned(grid->values);
  grid->values = NULL;

  return sif__fft_workspace_init_backward(*ws_out, state->fft_mgr);
}

/* random_grid and random_mesh are NULL for a box run. */
static int ctx_init(exodus_ctx_t* ctx, sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_option opt, sif_grid_t* random_grid, const sif_chain_mesh_t* random_mesh,
  int save_memory) {

  memset(ctx, 0, sizeof(*ctx));

  ctx->mesh = mesh;
  ctx->random_grid = random_grid;
  ctx->random_mesh = random_mesh;
  const int survey = (random_mesh != NULL);

  ctx->n_threads = sif__system_max_threads();
  if (ctx->n_threads < 1)
    ctx->n_threads = 1;

  ctx->sorted_radii = sif_malloc_aligned(n_radii * sizeof(sif_real));
  if (!ctx->sorted_radii) {
    SIF_LOG_ERROR(TAG, "out of memory: %u radii", n_radii);
    return SIF_ERR_ALLOC;
  }
  memcpy(ctx->sorted_radii, radii, n_radii * sizeof(sif_real));
  sort_radii_desc(ctx->sorted_radii, n_radii);

  ctx->cat = sif_catalogue_alloc(CATALOGUE_INITIAL_CAPACITY);
  if (!ctx->cat)
    return SIF_ERR_ALLOC;

  ctx->mask = sif_bitmask_alloc(grid->total_cells);
  if (!ctx->mask)
    return SIF_ERR_ALLOC;
  ctx->seed_minima = (opt & SIF_FINDER_SEEDS_MINIMA) != 0;
  if (ctx->seed_minima) {
    ctx->below = sif_bitmask_alloc(grid->total_cells);
    if (!ctx->below)
      return SIF_ERR_ALLOC;
  }

  ctx->void_cll = sif_cell_linked_list_alloc(
    VOID_CLL_CELLS, grid->box_length, ctx->cat->capacity, SIF_PBC_PERIODIC);
  if (!ctx->void_cll)
    return SIF_ERR_ALLOC;

  ctx->batch_results = sif_malloc_aligned(BATCH_MAX * sizeof(batch_result_t));
  ctx->batch_indices = sif_malloc_aligned(BATCH_MAX * sizeof(uint64_t));
  if (!ctx->batch_results || !ctx->batch_indices)
    return SIF_ERR_ALLOC;

  /* One radial scratch per thread. Its size is set by the bin count, not by
   * the particle load, so it stays in the tens of KiB whatever the search
   * radius and the tracer density are. */
  ctx->scratch =
    sif_calloc_aligned((size_t)ctx->n_threads, sizeof(radial_scratch_t));
  if (!ctx->scratch)
    return SIF_ERR_ALLOC;

  for (int t = 0; t < ctx->n_threads; t++) {
    if (sif__exodus_scratch_init(
          &ctx->scratch[t], mesh->weights != NULL, survey) != SIF_OK) {
      SIF_LOG_ERROR(
        TAG, "out of memory: radial scratch for %d threads", ctx->n_threads);
      return SIF_ERR_ALLOC;
    }
  }

  /* The occupancy reads the random mesh only, so it goes first: nothing a
   * failure here leaves behind has touched either grid. */
  if (survey && sif__exodus_survey_occupancy(
                  &ctx->occupied, ctx->random_mesh, grid) != SIF_OK)
    return SIF_ERR_ALLOC;

  /* The radii are sorted descending, so the last is the smallest. */
  const sif_real min_radius = ctx->sorted_radii[n_radii - 1];

  int status = ctx_init_grid(&ctx->fft_ws, grid, min_radius, opt, save_memory);
  if (status != SIF_OK || !survey)
    return status;

  status =
    ctx_init_grid(&ctx->random_ws, random_grid, min_radius, opt, save_memory);

  /* The data grid is already inside its workspace by now. Transform it back,
   * so that ctx_release() hands the caller the grid they gave rather than an
   * uninitialised buffer. */
  if (status != SIF_OK &&
      sif__fft_apply_filter(ctx->fft_ws, SIF__FILTER_NONE, 0, 0) == SIF_OK)
    grid->values = sif__fft_grid_backward(ctx->fft_ws);

  return status;
}

static int accept_void(exodus_ctx_t* ctx, const sif_grid_t* grid, sif_real cx,
  sif_real cy, sif_real cz, sif_real r) {

  int status = sif_catalogue_append(ctx->cat, cx, cy, cz, r);
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

  sif__mark_sphere(
    ctx->mask, cx, cy, cz, r, grid->n_cells, grid->p2_mask, grid->cell_length);

  /* Updated here rather than scanned for later, so every overlap test after
   * this point sizes its search box against a catalogue that includes this
   * void. */
  if (r > ctx->max_accepted_r)
    ctx->max_accepted_r = r;

  return SIF_OK;
}

/*
 * Point 1 of the void definition, as the minima keep it: a void is refused
 * only by a larger one. The ladder goes largest first, but a void found at a
 * small rung can still outgrow voids accepted at rungs above it, its
 * crossing lying past that rung's reach. It then takes their place: the smaller
 * voids it collides with leave the catalogue and it comes in.
 *
 * find_conflicts() lists the accepted voids a sphere collides with, by the
 * rule sif__overlap_exact() applies, and returns the largest radius among
 * them (-1 for none); displace_void() takes one back out.
 */
static int find_conflicts(exodus_ctx_t* ctx, sif_real cx, sif_real cy,
  sif_real cz, sif_real r, sif_real f, sif_real box, sif_real* largest) {

  const sif_catalogue_t* cat = ctx->cat;
  const sif_cell_linked_list_t* cll = ctx->void_cll;
  const sif_real half = 0.5f * box;
  const sif_real max_r = ctx->max_accepted_r;
  const sif_real reach = r + max_r - f * (r < max_r ? r : max_r);
  const int32_t n = (int32_t)cll->n_cells;
  const sif_real x[3] = {cx, cy, cz};

  int32_t lo[3], hi[3];
  for (int a = 0; a < 3; a++) {
    lo[a] = (int32_t)SIF_REAL_FLOOR((x[a] - reach) * cll->inv_cell_length);
    hi[a] = (int32_t)SIF_REAL_FLOOR((x[a] + reach) * cll->inv_cell_length);
    /* A window as wide as the list would visit cells twice. */
    if (hi[a] - lo[a] >= n) {
      lo[a] = 0;
      hi[a] = n - 1;
    }
  }

  ctx->n_conflicts = 0;
  *largest = -1.0f;
  for (int32_t a = lo[0]; a <= hi[0]; a++) {
    const int32_t wa = ((a % n) + n) % n;
    for (int32_t b = lo[1]; b <= hi[1]; b++) {
      const int32_t wb = ((b % n) + n) % n;
      for (int32_t c = lo[2]; c <= hi[2]; c++) {
        const int32_t wc = ((c % n) + n) % n;
        const uint64_t cell = ((uint64_t)wa * n + wb) * n + wc;
        for (int32_t i = cll->head[cell]; i != -1; i = cll->next[i]) {
          sif_real dx = SIF_REAL_ABS(cx - cat->cx[i]);
          sif_real dy = SIF_REAL_ABS(cy - cat->cy[i]);
          sif_real dz = SIF_REAL_ABS(cz - cat->cz[i]);
          dx = dx > half ? box - dx : dx;
          dy = dy > half ? box - dy : dy;
          dz = dz > half ? box - dz : dz;
          const sif_real ri = cat->radii[i];
          const sif_real tot = r + ri - f * (r < ri ? r : ri);
          if (dx * dx + dy * dy + dz * dz >= tot * tot)
            continue;
          if (ctx->n_conflicts == ctx->conflicts_capacity) {
            const uint64_t cap =
              ctx->conflicts_capacity ? 2 * ctx->conflicts_capacity : 64;
            uint64_t* grown = realloc(ctx->conflicts, cap * sizeof(*grown));
            if (!grown)
              return SIF_ERR_ALLOC;
            ctx->conflicts = grown;
            ctx->conflicts_capacity = cap;
          }
          ctx->conflicts[ctx->n_conflicts++] = (uint64_t)i;
          if (ri > *largest)
            *largest = ri;
        }
      }
    }
  }
  return SIF_OK;
}

/*
 * The smallest radius past which a sphere at c certainly collides with an
 * accepted void of radius at least min_rj, or INFINITY if none is within
 * reach of r_hi.
 *
 * Most candidates that are rescaled are then refused by the overlap re-check:
 * they clear the catalogue at the rung but not at the radius the rescaling
 * returns. A collision only grows with the radius -- r + r_j - f * min(r, r_j)
 * never falls as r grows, for any f up to 1 -- so each void j it could be with
 * gives a radius r*_j past which the collision is certain, and the smallest of
 * them is where the candidate's fate is sealed. sif__exodus_witness() then
 * says whether the rescaling would certainly return more than that, and if it
 * would, the candidate is refused without it.
 *
 * Each r*_j is solved in double with the separation padded by
 * COLLISION_SLACK of the lengths involved, many times anything the float
 * test in sif__overlap_exact() can be off by, so past r*_j that test cannot
 * fail to see the collision. The padding only gives up a few candidates the
 * witness could have refused, which are then rescaled as before.
 */
#define COLLISION_SLACK 1e-5

static double collision_radius(const exodus_ctx_t* ctx, sif_real cx,
  sif_real cy, sif_real cz, sif_real r_hi, sif_real min_rj, sif_real f,
  sif_real box) {

  const sif_catalogue_t* cat = ctx->cat;
  const sif_cell_linked_list_t* cll = ctx->void_cll;
  const double L = (double)box;
  const double half = 0.5 * L;
  const double ff = (double)f;
  const sif_real reach = r_hi + ctx->max_accepted_r;
  const int32_t n = (int32_t)cll->n_cells;
  const sif_real x[3] = {cx, cy, cz};

  int32_t lo[3], hi[3];
  for (int a = 0; a < 3; a++) {
    lo[a] = (int32_t)SIF_REAL_FLOOR((x[a] - reach) * cll->inv_cell_length);
    hi[a] = (int32_t)SIF_REAL_FLOOR((x[a] + reach) * cll->inv_cell_length);
    if (hi[a] - lo[a] >= n) {
      lo[a] = 0;
      hi[a] = n - 1;
    }
  }

  double best = INFINITY;
  for (int32_t a = lo[0]; a <= hi[0]; a++) {
    const int32_t wa = ((a % n) + n) % n;
    for (int32_t b = lo[1]; b <= hi[1]; b++) {
      const int32_t wb = ((b % n) + n) % n;
      for (int32_t c = lo[2]; c <= hi[2]; c++) {
        const int32_t wc = ((c % n) + n) % n;
        const uint64_t cell = ((uint64_t)wa * n + wb) * n + wc;
        for (int32_t i = cll->head[cell]; i != -1; i = cll->next[i]) {
          const double rj = (double)cat->radii[i];
          if (cat->radii[i] < min_rj)
            continue;

          double dx = fabs((double)cx - (double)cat->cx[i]);
          double dy = fabs((double)cy - (double)cat->cy[i]);
          double dz = fabs((double)cz - (double)cat->cz[i]);
          dx = dx > half ? L - dx : dx;
          dy = dy > half ? L - dy : dy;
          dz = dz > half ? L - dz : dz;
          const double D = sqrt(dx * dx + dy * dy + dz * dz) +
                           COLLISION_SLACK * (L + (double)r_hi + rj);

          /* tot(r) = r + rj - f min(r, rj): rj at r = 0, (2 - f) rj at
           * r = rj, slope 1 - f below that and 1 above. */
          double r_star;
          if (D <= rj)
            r_star = 0.0;
          else if (D <= (2.0 - ff) * rj)
            r_star = (D - rj) / (1.0 - ff);
          else
            r_star = D - (1.0 - ff) * rj;

          if (r_star < best)
            best = r_star;
        }
      }
    }
  }
  return best;
}

/*
 * The rung's overlap tests at the rung radius, run once over every candidate
 * before the batches start, and the candidates they refuse dropped from the
 * list in place, order kept.
 *
 * At the small rungs nearly every candidate falls to these tests -- a
 * million at a time, against the ten thousand that reach a rescaling -- and
 * inside the batches they ran in a parallel region per batch, a thousand and
 * more regions a rung, each sized for the rescalings and not for them. Here
 * they are one loop.
 *
 * Only under the cell seeds. There the catalogue only gains voids during a
 * rung, and each test only finds more collisions as voids are added, so a
 * candidate refused against the catalogue as the rung begins would be refused
 * at its turn as well. The batches still run every test again on the ones
 * kept, against the catalogue as it then stands, so what is accepted is
 * exactly what it was. Under the minima a displacement takes voids out, and a
 * refusal does not hold.
 *
 * The list is cut into chunks, each compacted by one thread, and the chunks
 * then closed up in order.
 */
#define PREFILTER_CHUNKS_PER_THREAD 16

static int prefilter_cells(exodus_ctx_t* ctx, const sif_grid_t* grid,
  sif_real radius, sif_real overlap_fraction, int32_t proxy_pole,
  sif_finder_radius_stats_t* stats) {

  const uint64_t n = ctx->candidates.count;
  if (n == 0)
    return SIF_OK;

  uint64_t n_chunks =
    (uint64_t)PREFILTER_CHUNKS_PER_THREAD * (uint64_t)ctx->n_threads;
  if (n_chunks > n)
    n_chunks = n;

  uint64_t* kept = malloc(n_chunks * sizeof(*kept));
  if (!kept) {
    SIF_LOG_ERROR(TAG, "out of memory: candidate pre-filter of %" PRIu64
                       " chunks", n_chunks);
    return SIF_ERR_ALLOC;
  }

  sif_candidate_t* items = ctx->candidates.items;
  uint64_t n_masked = 0, n_proxy = 0, n_overlap = 0;

#pragma omp parallel for schedule(dynamic, 1) num_threads(ctx->n_threads) \
  reduction(+ : n_masked, n_proxy, n_overlap)
  for (uint64_t c = 0; c < n_chunks; c++) {
    const uint64_t lo = n * c / n_chunks;
    const uint64_t hi = n * (c + 1) / n_chunks;
    uint64_t w = lo;

    for (uint64_t k = lo; k < hi; k++) {
      if (masked_overlap(ctx, k, radius, overlap_fraction, grid->cell_length)) {
        n_masked++;
        continue;
      }

      uint32_t ix, iy, iz;
      sif__unflatten_index(grid, items[k].flat_idx, &ix, &iy, &iz);
      if (proxy_pole > 0 &&
          sif__overlap_quick(ctx->mask, grid->n_cells, grid->p2_mask, ix, iy,
            iz, (uint32_t)proxy_pole)) {
        n_proxy++;
        continue;
      }

      sif_real cc[3];
      candidate_centre(ctx, k, grid, ix, iy, iz, cc);
      if (sif__overlap_exact(ctx->cat, cc[0], cc[1], cc[2], radius,
            ctx->max_accepted_r, grid->box_length, ctx->void_cll,
            grid->p2_mask, overlap_fraction)) {
        n_overlap++;
        continue;
      }

      items[w++] = items[k];
    }
    kept[c] = w - lo;
  }

  uint64_t total = 0;
  for (uint64_t c = 0; c < n_chunks; c++) {
    const uint64_t lo = n * c / n_chunks;
    if (lo != total && kept[c] > 0)
      memmove(&items[total], &items[lo], kept[c] * sizeof(*items));
    total += kept[c];
  }
  free(kept);

  ctx->candidates.count = total;
  stats->rejected_masked += n_masked;
  stats->rejected_proxy += n_proxy;
  stats->rejected_overlap += n_overlap;
  return SIF_OK;
}

static int displace_void(
  exodus_ctx_t* ctx, const sif_grid_t* grid, uint64_t i) {

  sif_catalogue_t* cat = ctx->cat;
  const sif_real cx = cat->cx[i], cy = cat->cy[i], cz = cat->cz[i];
  const sif_real r = cat->radii[i];

  int status = sif_cell_linked_list_remove(ctx->void_cll, i, cx, cy, cz);
  if (status != SIF_OK)
    return status;
  cat->radii[i] = -r;
  ctx->n_removed++;

  /* Its cells, then back the ones another void it intersects still covers.
   * The walk leaves ctx->conflicts alone: the caller is iterating it. */
  sif__clear_sphere(
    ctx->mask, cx, cy, cz, r, grid->n_cells, grid->p2_mask, grid->cell_length);
  const sif_cell_linked_list_t* cll = ctx->void_cll;
  const sif_real box = grid->box_length, half = 0.5f * box;
  const sif_real reach = r + ctx->max_accepted_r;
  const int32_t n = (int32_t)cll->n_cells;
  const sif_real x[3] = {cx, cy, cz};
  int32_t lo[3], hi[3];
  for (int a = 0; a < 3; a++) {
    lo[a] = (int32_t)SIF_REAL_FLOOR((x[a] - reach) * cll->inv_cell_length);
    hi[a] = (int32_t)SIF_REAL_FLOOR((x[a] + reach) * cll->inv_cell_length);
    if (hi[a] - lo[a] >= n) {
      lo[a] = 0;
      hi[a] = n - 1;
    }
  }
  for (int32_t a = lo[0]; a <= hi[0]; a++)
    for (int32_t b = lo[1]; b <= hi[1]; b++)
      for (int32_t c = lo[2]; c <= hi[2]; c++) {
        const uint64_t cell =
          ((uint64_t)(((a % n) + n) % n) * n + (uint64_t)(((b % n) + n) % n)) *
            n +
          (uint64_t)(((c % n) + n) % n);
        for (int32_t j = cll->head[cell]; j != -1; j = cll->next[j]) {
          sif_real dx = SIF_REAL_ABS(cx - cat->cx[j]);
          sif_real dy = SIF_REAL_ABS(cy - cat->cy[j]);
          sif_real dz = SIF_REAL_ABS(cz - cat->cz[j]);
          dx = dx > half ? box - dx : dx;
          dy = dy > half ? box - dy : dy;
          dz = dz > half ? box - dz : dz;
          const sif_real t = r + cat->radii[j];
          if (dx * dx + dy * dy + dz * dz < t * t)
            sif__mark_sphere(ctx->mask, cat->cx[j], cat->cy[j], cat->cz[j],
              cat->radii[j], grid->n_cells, grid->p2_mask, grid->cell_length);
        }
      }
  return SIF_OK;
}

/* Drops the voids the rung displaced and re-indexes the rest, so the
 * catalogue a checkpoint saves holds only voids that stand. */
static int compact_catalogue(exodus_ctx_t* ctx) {
  if (ctx->n_removed == 0)
    return SIF_OK;

  sif_catalogue_t* cat = ctx->cat;
  uint64_t j = 0;
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    if (!(cat->radii[i] > 0.0f))
      continue;
    cat->cx[j] = cat->cx[i];
    cat->cy[j] = cat->cy[i];
    cat->cz[j] = cat->cz[i];
    cat->radii[j] = cat->radii[i];
    if (cat->footprint) {
      cat->footprint[j] = cat->footprint[i];
      cat->footprint_shell[j] = cat->footprint_shell[i];
    }
    j++;
  }
  cat->n_voids = j;

  sif_cell_linked_list_t* cll = ctx->void_cll;
  const uint64_t n = cll->n_cells;
  for (uint64_t c = 0; c < n * n * n; c++)
    cll->head[c] = -1;
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    const int status =
      sif_cell_linked_list_insert(cll, i, cat->cx[i], cat->cy[i], cat->cz[i]);
    if (status != SIF_OK)
      return status;
  }
  ctx->n_removed = 0;
  return SIF_OK;
}

/* What sif__exodus_check_weights() makes of one weight. */
#define WEIGHT_OK        0
#define WEIGHT_NEGATIVE  1
#define WEIGHT_NONFINITE 2

/*
 * Classified by its bits rather than with isfinite() and a comparison: the
 * release build is compiled with -ffast-math, which entitles the compiler to
 * assume no NaN or infinity exists and fold exactly those tests away. A bit
 * pattern cannot be reasoned about that way. Negative zero is a zero, and
 * passes.
 */
static inline int classify_weight(sif_real w) {
  if (sizeof(sif_real) == sizeof(uint32_t)) {
    uint32_t b;
    memcpy(&b, &w, sizeof(b));
    if (((b >> 23) & 0xFFu) == 0xFFu)
      return WEIGHT_NONFINITE;
    return ((b >> 31) && (b << 1)) ? WEIGHT_NEGATIVE : WEIGHT_OK;
  }

  uint64_t b;
  memcpy(&b, &w, sizeof(b));
  if (((b >> 52) & 0x7FFu) == 0x7FFu)
    return WEIGHT_NONFINITE;
  return ((b >> 63) && (b << 1)) ? WEIGHT_NEGATIVE : WEIGHT_OK;
}

/*
 * Whether a weighted mesh is one the rescaling can use.
 *
 * The bin-skipping walk rests on the enclosed weight never shrinking as the
 * radius grows, which a negative weight breaks: the skip test would then throw
 * away shells that do hold the crossing, with nothing to show that it had. A
 * NaN or an infinity poisons every sum it enters. Both are refused up front,
 * counted, rather than left to produce a catalogue that looks normal.
 */
int sif__exodus_check_weights(const sif_chain_mesh_t* mesh) {
  if (!mesh->cell_weights) {
    SIF_LOG_ERROR(TAG, "mesh: weights but no per-cell weight table");
    return SIF_ERR_INVALID;
  }

  uint64_t n_negative = 0;
  uint64_t n_nonfinite = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_negative, n_nonfinite)
  for (uint64_t p = 0; p < mesh->n_particles; p++) {
    const int kind = classify_weight(mesh->weights[p]);
    n_nonfinite += (kind == WEIGHT_NONFINITE);
    n_negative += (kind == WEIGHT_NEGATIVE);
  }

  if (n_negative > 0 || n_nonfinite > 0) {
    SIF_LOG_ERROR(TAG,
      "mesh: weights must be finite and non-negative (%" PRIu64
      " negative, %" PRIu64 " not finite, of %" PRIu64 ")",
      n_negative, n_nonfinite, mesh->n_particles);
    return SIF_ERR_INVALID;
  }

  if (!(mesh->total_weight > 0.0)) {
    SIF_LOG_ERROR(TAG, "mesh: weights sum to zero (no mean density)");
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/* --- Driver --- */

/* Whether this build can run the seeds `options` ask for on this kind of
 * run: either in a box, only SIF_FINDER_SEEDS_CELLS in a survey, for now. */
int sif__exodus_check_seeds(sif_option options, int survey) {
  if (survey && (options & SIF_FINDER_SEEDS_MINIMA)) {
    SIF_LOG_ERROR(TAG, "options: SIF_FINDER_SEEDS_MINIMA: a survey (expected "
                       "SIF_FINDER_SEEDS_CELLS, for now)");
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/*
 * The checks the extras need that the rest of the arguments do not: a prior
 * catalogue the finder can place, and no checkpoint path asked for alongside
 * the option that turns checkpoints off.
 */
int sif__exodus_check_extras(const sif_finder_exodus_extras_t* extras,
  sif_option options, const sif_real offset[3], sif_real box_length) {

  if (!extras)
    return SIF_OK;

  if (extras->checkpoint && (options & SIF_FINDER_NO_CHECKPOINT)) {
    SIF_LOG_ERROR(TAG, "checkpoint: %s, with SIF_FINDER_NO_CHECKPOINT set",
      extras->checkpoint);
    return SIF_ERR_INVALID;
  }
  if (extras->checkpoint &&
      strlen(extras->checkpoint) + 32 >= SIF__CHECKPOINT_PATH_MAX) {
    SIF_LOG_ERROR(TAG, "checkpoint: path of %zu characters (expected under %d)",
      strlen(extras->checkpoint), SIF__CHECKPOINT_PATH_MAX - 32);
    return SIF_ERR_INVALID;
  }

  const sif_catalogue_t* prior = extras->prior;
  if (!prior)
    return SIF_OK;

  if (prior->units != SIF_COORDINATES_CARTESIAN) {
    SIF_LOG_ERROR(
      TAG, "prior: centres on the sky (expected Cartesian coordinates)");
    return SIF_ERR_INVALID;
  }

  for (uint64_t i = 0; i < prior->n_voids; i++) {
    const sif_real r = prior->radii[i];
    if (!(r > 0.0f) || !isfinite(r)) {
      SIF_LOG_ERROR(TAG,
        "prior: void %" PRIu64 ": radius %g (expected "
        "positive and finite)",
        i, (double)r);
      return SIF_ERR_INVALID;
    }

    /* Where the finder will put it, which is where it has to fit. */
    const sif_real c[3] = {prior->cx[i] + (offset ? offset[0] : 0.0f),
      prior->cy[i] + (offset ? offset[1] : 0.0f),
      prior->cz[i] + (offset ? offset[2] : 0.0f)};
    for (int k = 0; k < 3; k++) {
      if (!(c[k] >= 0.0f && c[k] < box_length)) {
        SIF_LOG_ERROR(TAG,
          "prior: void %" PRIu64 ": centre (%g, %g, %g)%s outside the box "
          "[0, %g)",
          i, (double)prior->cx[i], (double)prior->cy[i], (double)prior->cz[i],
          offset ? ", moved by the offset," : "", (double)box_length);
        return SIF_ERR_INVALID;
      }
    }
  }

  return SIF_OK;
}

/* --- variants --- */

static const struct {
  const char* name;
  sif_real overlap_fraction;
  sif_option options;
} VARIANTS[SIF__EXODUS_N_VARIANTS] = {
  [SIF_EXODUS_EXCURSION_SET] = {"excursion-set", 0.0f, SIF_FINDER_SEEDS_CELLS},
  [SIF_EXODUS_EXCURSION_PEAK] = {"excursion-peak", 1.0f,
    SIF_FINDER_SEEDS_MINIMA},
};

static int variant_valid(sif_exodus_variant_t variant) {
  return (unsigned)variant < (unsigned)SIF__EXODUS_N_VARIANTS;
}

int sif_finder_exodus_preset(
  sif_exodus_variant_t variant, sif_real* overlap_fraction, sif_option* opt) {

  if (!variant_valid(variant)) {
    SIF_LOG_ERROR(TAG, "variant: %d (expected 0 to %d)", (int)variant,
      SIF__EXODUS_N_VARIANTS - 1);
    return SIF_ERR_INVALID;
  }
  if (overlap_fraction)
    *overlap_fraction = VARIANTS[variant].overlap_fraction;
  if (opt)
    *opt = VARIANTS[variant].options;
  return SIF_OK;
}

const char* sif_finder_exodus_variant_name(sif_exodus_variant_t variant) {
  return variant_valid(variant) ? VARIANTS[variant].name : NULL;
}

int sif_finder_exodus_variant_from_name(
  const char* name, sif_exodus_variant_t* variant) {

  for (int v = 0; name && v < SIF__EXODUS_N_VARIANTS; v++) {
    if (strcmp(name, VARIANTS[v].name) == 0) {
      if (variant)
        *variant = (sif_exodus_variant_t)v;
      return SIF_OK;
    }
  }
  SIF_LOG_ERROR(TAG,
    "variant: \"%s\" (expected excursion-set or excursion-peak)",
    name ? name : "(null)");
  return SIF_ERR_INVALID;
}

sif_catalogue_t* sif_finder_exodus(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_exodus_variant_t variant,
  const sif_finder_exodus_extras_t* extras) {

  sif_real overlap_fraction;
  sif_option options;
  if (sif_finder_exodus_preset(variant, &overlap_fraction, &options) !=
      SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  return sif_finder_exodus_raw(
    grid, mesh, radii, n_radii, threshold, overlap_fraction, options, extras);
}

sif_catalogue_t* sif_finder_exodus_raw(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option options,
  const sif_finder_exodus_extras_t* extras) {

  if (!grid || !grid->values || !mesh) {
    SIF_LOG_ERROR(TAG, "%s",
      !grid           ? "grid: NULL"
      : !grid->values ? "grid: no values"
                      : "mesh: NULL");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (sif__finder_check_args(
        TAG, radii, n_radii, threshold, overlap_fraction) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  /* A grid still holding densities, handed to something that expects a
   * density contrast, produces numbers rather than an error. SIF_GRID_EMPTY is
   * not flagged: that is a grid the caller filled directly, and only the caller
   * knows what is in it. */
  if (grid->content == SIF_GRID_DENSITY) {
    SIF_LOG_WARNING(TAG,
      "this grid holds a density, not a density contrast; call "
      "sif_grid_to_density_contrast first");
  }

  if (!mesh->x || mesh->n_particles == 0) {
    SIF_LOG_ERROR(TAG, "mesh: no particles");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  /* Grid and mesh coordinates are used interchangeably throughout, so the two
   * have to describe the same box. */
  if (mesh->box_length != grid->box_length) {
    SIF_LOG_ERROR(TAG, "mesh: box of %g, grid: box of %g (must match)",
      (double)mesh->box_length, (double)grid->box_length);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (mesh->weights && sif__exodus_check_weights(mesh) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (sif__exodus_check_seeds(options, 0) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (sif__exodus_check_extras(extras, options, NULL, grid->box_length) !=
      SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  return sif__exodus_run(grid, mesh, radii, n_radii, threshold,
    overlap_fraction, options, NULL, NULL, NULL, extras);
}

/*
 * The largest search radius a run can reach: the first rung's, since the
 * ladder only descends and the gap guard never reaches past the previous rung.
 * Same expression as in the loop.
 */
sif_real sif__exodus_max_search_radius(const sif_real* radii, uint32_t n_radii,
  sif_real cell_length, sif_option options) {

  sif_real r_max = radii[0];
  for (uint32_t i = 1; i < n_radii; i++)
    if (radii[i] > r_max)
      r_max = radii[i];

  const sif_real search_factor = (sif_real)sif__finder_search_factor(options);
  return r_max +
         SIF_REAL_MAX((search_factor - 1.0f) * r_max, 3.0f * cell_length);
}

/* --- checkpoints --- */

/* See sif__exodus_halt_after_set(). */
static uint32_t halt_after = 0;

void sif__exodus_halt_after_set(uint32_t n_rungs) { halt_after = n_rungs; }

/* The Gaussian minima are found on, under SIF_FINDER_SEEDS_MINIMA, as a
 * fraction of the rung. 0.15 tracks the minima of a 1500^3 grid to within a
 * few percent from 600^3 up, in the L2800 tests; 0.30 does as well but merges
 * more of them. */
#define RUNG_MINIMA_SIGMA 0.15f

typedef struct {
  sif_fft_workspace_t* ws;
  sif_grid_t* grid;
  sif_real radius;
  sif_real sigma;
} resmooth_t;

static int resmooth_rung(void* arg) {
  resmooth_t* r = arg;
  const int status = sif__fft_apply_filter_smoothed(
    r->ws, SIF__FILTER_TOP_HAT, r->radius, r->sigma, r->grid->box_length);
  if (status != SIF_OK)
    return status;
  r->grid->values = sif__fft_grid_backward(r->ws);
  return r->grid->values ? SIF_OK : SIF_ERR_ALLOC;
}

/* A run's checkpoint, and where it stands. See exodus_checkpoint.c for why
 * the catalogue is all it needs to hold. */
typedef struct {
  int active;   /* written after every rung */
  int on_disk;  /* this run wrote one, or resumed one */
  int progress; /* the one on disk holds a finished rung */
  int keep;     /* left in place on success, for the caller */
  char path[SIF__CHECKPOINT_PATH_MAX];
  sif_finder_exodus_checkpoint_t header;
} checkpoint_t;

/* Where a run starts: the voids its catalogue holds before the first rung it
 * runs, which rung that is, and the mismatch count carried over. */
typedef struct {
  sif_catalogue_t* voids;
  uint32_t first_rung;
  uint64_t n_mismatched;
} run_start_t;

/* The prior voids, moved to where the finder works. */
static sif_catalogue_t* prior_in_box(
  const sif_catalogue_t* prior, const sif_real offset[3]) {

  const uint64_t n = prior ? prior->n_voids : 0;
  sif_catalogue_t* voids = sif_catalogue_alloc(n);
  if (!voids)
    return NULL;

  for (uint64_t i = 0; i < n; i++) {
    const int status =
      sif_catalogue_append(voids, prior->cx[i] + (offset ? offset[0] : 0.0f),
        prior->cy[i] + (offset ? offset[1] : 0.0f),
        prior->cz[i] + (offset ? offset[2] : 0.0f), prior->radii[i]);
    if (status != SIF_OK) {
      sif_catalogue_free(voids);
      return NULL;
    }
  }
  return voids;
}

/*
 * Decides where the run starts, before any of the expensive setup: a
 * checkpoint that does not describe this run, or cannot be written, should
 * fail the call in milliseconds, not after the grid's transform.
 *
 * A checkpoint the caller named and that exists is resumed. Anything else
 * starts from the prior voids, if any, and writes the first checkpoint now --
 * which is also what proves the path writable while the caller is still
 * watching.
 */
static int run_start(checkpoint_t* ck, run_start_t* start,
  const sif_grid_t* grid, const sif_chain_mesh_t* mesh,
  const sif_chain_mesh_t* random_mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option options,
  const sif_real offset[3], const sif_finder_exodus_extras_t* extras) {

  memset(ck, 0, sizeof(*ck));
  memset(start, 0, sizeof(*start));

  const sif_catalogue_t* prior = extras ? extras->prior : NULL;
  const char* named = extras ? extras->checkpoint : NULL;
  ck->keep = extras && extras->keep_checkpoint;

  if (options & SIF_FINDER_NO_CHECKPOINT) {
    start->voids = prior_in_box(prior, offset);
    return start->voids ? SIF_OK : SIF_ERR_ALLOC;
  }

  sif_real* sorted = sif_malloc_aligned(n_radii * sizeof(sif_real));
  if (!sorted) {
    SIF_LOG_ERROR(TAG, "out of memory: %u radii", n_radii);
    return SIF_ERR_ALLOC;
  }
  memcpy(sorted, radii, n_radii * sizeof(sif_real));
  sort_radii_desc(sorted, n_radii);

  sif_finder_exodus_checkpoint_t* h = &ck->header;
  h->is_survey = random_mesh != NULL;
  h->is_weighted = mesh->weights != NULL;
  h->options = options & (SIF__FINDER_SEARCH_MASK | SIF_FINDER_KEEP_CIC_WINDOW |
                           SIF__FINDER_SEEDS_MASK | SIF_FINDER_ALIASING_WINDOW);
  h->n_radii = n_radii;
  h->grid_n_cells = grid->n_cells;
  h->assignment = ((uint32_t)sif__fft_grid_window_order(grid) & 0xffu) |
                  ((grid->interlaced ? 1u : 0u) << 8);
  h->box_length = (double)grid->box_length;
  h->threshold = (double)threshold;
  h->overlap_fraction = (double)overlap_fraction;
  h->n_particles = mesh->n_particles;
  h->total_weight = mesh->total_weight;
  if (random_mesh) {
    h->random_n_particles = random_mesh->n_particles;
    h->random_total_weight = random_mesh->total_weight;
  }
  for (int k = 0; random_mesh && offset && k < 3; k++)
    h->offset[k] = (double)offset[k];
  if (prior && prior->n_voids > 0) {
    h->n_prior = prior->n_voids;
    h->prior_crc32 = sif__exodus_checkpoint_crc(prior);
  }

  if (named)
    snprintf(ck->path, sizeof(ck->path), "%s", named);
  else
    sif__exodus_checkpoint_name(ck->path, sizeof(ck->path));

  int status = SIF_OK;

  if (named && access(named, F_OK) == 0) {
    sif_finder_exodus_checkpoint_t got;
    sif_real* got_radii = NULL;

    status =
      sif__exodus_checkpoint_load(named, &got, &got_radii, &start->voids);
    if (status == SIF_OK)
      status = sif__exodus_checkpoint_match(named, h, sorted, &got, got_radii);
    sif_free_aligned(got_radii);

    if (status == SIF_OK) {
      start->first_rung = got.next_rung;
      start->n_mismatched = got.n_mismatched;
      ck->active = 1;
      ck->on_disk = 1;
      ck->progress = 1;
      SIF_LOG_INFO(TAG,
        "checkpoint: %s: resuming with %u of %u rungs done, %" PRIu64 " voids",
        named, got.next_rung, n_radii, got.n_voids);
    }
    sif_free_aligned(sorted);
    return status;
  }

  start->voids = prior_in_box(prior, offset);
  if (!start->voids) {
    sif_free_aligned(sorted);
    return SIF_ERR_ALLOC;
  }

  status = sif__exodus_checkpoint_write(ck->path, h, sorted, start->voids,
    named ? NULL : "running without a checkpoint");
  sif_free_aligned(sorted);

  if (status == SIF_OK) {
    ck->active = 1;
    ck->on_disk = 1;
    SIF_LOG_INFO(TAG, "checkpoint: %s", ck->path);
  }

  /* Only a checkpoint the caller asked for by name is worth failing over. */
  return named ? status : SIF_OK;
}

/* After rung i: the catalogue so far, and the rungs it covers. */
static void checkpoint_rung(checkpoint_t* ck, const exodus_ctx_t* ctx,
  uint32_t rungs_done, uint64_t n_mismatched) {

  if (!ck->active)
    return;

  ck->header.next_rung = rungs_done;
  ck->header.n_mismatched = n_mismatched;

  /* A failure is logged as a warning and leaves the previous checkpoint in
   * place; the next rung tries again. */
  if (sif__exodus_checkpoint_write(ck->path, &ck->header, ctx->sorted_radii,
        ctx->cat, "checkpoint not updated, the previous one stands") == SIF_OK)
    ck->progress = 1;
}

/* A run that completed has no use for its checkpoint, and neither has one
 * that failed before finishing a rung: resuming it would start over. Any other
 * keeps it, and says where. */
static void checkpoint_finish(const checkpoint_t* ck, int status) {
  if (!ck->on_disk)
    return;

  if (status != SIF_OK && ck->progress) {
    SIF_LOG_INFO(TAG, "checkpoint: %s: kept, resume the run from it", ck->path);
    return;
  }
  if (status == SIF_OK && ck->keep)
    return;

  if (unlink(ck->path) != 0)
    SIF_LOG_WARNING(
      TAG, "checkpoint: %s: cannot remove: %s", ck->path, strerror(errno));
}

/*
 * The run both finders share. Everything that decides which candidates become
 * voids -- the rung ladder, the speculative batches, the overlap re-checks --
 * is one piece of code, so the survey finder cannot drift from the box one in
 * anything but the three places it has to differ: the field the candidates
 * are scanned on, the rescaling, and the footprint it reports at the end.
 * random_grid and random_mesh are NULL for a box run, and the box run then
 * takes exactly the path it always has.
 */
sif_catalogue_t* sif__exodus_run(sif_grid_t* grid, const sif_chain_mesh_t* mesh,
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option options, sif_grid_t* random_grid,
  const sif_chain_mesh_t* random_mesh, const sif_real offset[3],
  const sif_finder_exodus_extras_t* extras) {

  const int weighted = (mesh->weights != NULL);
  const int survey = (random_mesh != NULL);

  checkpoint_t ck;
  run_start_t start;
  int status = run_start(&ck, &start, grid, mesh, random_mesh, radii, n_radii,
    threshold, overlap_fraction, options, offset, extras);
  if (status != SIF_OK) {
    sif_catalogue_free(start.voids);
    sif__error_status_set(status);
    return NULL;
  }

  exodus_ctx_t ctx;
  const int init = ctx_init(&ctx, grid, mesh, radii, n_radii, options,
    random_grid, random_mesh, extras && extras->save_memory);
  if (init != SIF_OK) {
    sif_catalogue_free(start.voids);
    checkpoint_finish(&ck, init);
    sif__error_status_set(init);
    ctx_release(&ctx, grid);
    return NULL;
  }

  /* The voids already found, put back exactly as the rungs that found them
   * left them: the overlap mask, the void index and the largest radius are
   * all accept_void()'s doing, and none depends on the order. */
  for (uint64_t v = 0; v < start.voids->n_voids && status == SIF_OK; v++)
    status = accept_void(&ctx, grid, start.voids->cx[v], start.voids->cy[v],
      start.voids->cz[v], start.voids->radii[v]);
  sif_catalogue_free(start.voids);
  start.voids = NULL;

  /* Loop invariants: the mean tracer density never changes between radii. */
  const sif_real box_volume =
    grid->box_length * grid->box_length * grid->box_length;
  const sif_real mean_density = (sif_real)mesh->n_particles / box_volume;
  const sif_real vol_factor = (4.0f / 3.0f) * SIF_PI * mean_density;

  /* What the density condition is measured against. On a weighted mesh that
   * is the mean weight density, and the tracer count above is kept only for
   * sizing the histogram. Rounded and divided exactly as the count is, so that
   * a mesh whose weights are all 1 reproduces the unweighted run bit for
   * bit. */
  const sif_real mass_factor =
    weighted
      ? (4.0f / 3.0f) * SIF_PI * ((sif_real)mesh->total_weight / box_volume)
      : vol_factor;

  if (weighted)
    SIF_LOG_TRACE(TAG,
      "weighted mesh: total weight %.6g over %" PRIu64 " tracers",
      mesh->total_weight, mesh->n_particles);

  /* The survey's normalisation: how much data weight each unit of random
   * weight stands for, and the threshold folded into it. */
  const sif_real survey_alpha =
    survey ? (sif_real)(mesh->total_weight / random_mesh->total_weight) : 0.0f;
  const sif_real survey_K = (1.0f + threshold) * survey_alpha;

  if (survey)
    SIF_LOG_TRACE(TAG,
      "survey: %" PRIu64 " randoms of total weight %.6g, alpha = %.6g, %" PRIu64
      " of %" PRIu64 " grid cells observed",
      random_mesh->n_particles, random_mesh->total_weight, (double)survey_alpha,
      sif_bitmask_count_set(ctx.occupied), grid->total_cells);

  sif_timer_t timer;
  uint64_t total_mismatched = start.n_mismatched;

  /*
   * Past the reach. Under the cell seeds a candidate whose crossing lies past
   * its rung's reach is refused, as in 0.4. Under the minima every larger rung
   * has already run, so nothing later would measure that void: it is
   * measured here and stands or falls by the overlap test like any other.
   * The reach is extended level by level (see EXT_GROWTH), from the bottom
   * of the ladder to the top rung's reach; the levels are the same for every
   * rung, so they are built once. Level k measures [ext_rmin[k],
   * ext_reach[k]] with the template etpl[k].
   */
  uint32_t n_ext = 0;
  sif_real ext_reach[EXT_MAX_LEVELS], ext_rmin[EXT_MAX_LEVELS];
  mesh_template_t etpl[EXT_MAX_LEVELS];
  memset(etpl, 0, sizeof(etpl));
  if (ctx.seed_minima && !survey && status == SIF_OK) {
    const sif_real r_ext = sif__exodus_max_search_radius(
      ctx.sorted_radii, n_radii, grid->cell_length, options);
    const sif_real r_low = ctx.sorted_radii[n_radii - 1];
    /* Down to the bottom rung, so every rung's reach lies above level 0. */
    while (n_ext < EXT_MAX_LEVELS &&
           (n_ext == 0 ||
             r_ext / SIF_REAL_POW(EXT_GROWTH, (sif_real)(n_ext - 1)) > r_low))
      n_ext++;
    for (uint32_t k = 0; k < n_ext; k++)
      ext_reach[k] =
        r_ext / SIF_REAL_POW(EXT_GROWTH, (sif_real)(n_ext - 1 - k));
    for (uint32_t k = 0; k < n_ext && status == SIF_OK; k++) {
      ext_rmin[k] =
        (k ? ext_reach[k - 1] : ext_reach[0] / EXT_GROWTH) / EXT_SLACK;
      status = sif__exodus_template_build(&etpl[k], ctx.mesh->n_cells,
        ctx.mesh->cell_length, ext_rmin[k], ext_reach[k]);
    }
  }

  for (uint32_t i = start.first_rung; i < n_radii && status == SIF_OK; i++) {
    sif_timer_start(&timer);
    const double t_rung = wall_s();

    const sif_real radius = ctx.sorted_radii[i];
    sif_finder_radius_stats_t stats = {0};

    /* Localisation quality for this rung, see MISMATCH_RATIO. */
    double ratio_sum = 0.0;
    sif_real ratio_min = 0.0f;
    sif_real ratio_max = 0.0f;
    uint64_t n_mismatched = 0;

    /* Accepted voids whose centre was refined off its node. */
    uint64_t n_refined = 0;

    /* Every rescaling this rung asks for, whatever becomes of it. */
    rescale_report_t rescale = {0};

    status = sif__fft_apply_filter(
      ctx.fft_ws, SIF__FILTER_TOP_HAT, radius, grid->box_length);
    if (status != SIF_OK)
      break;
    grid->values = sif__fft_grid_backward(ctx.fft_ws);

    if (survey) {
      status = sif__fft_apply_filter(
        ctx.random_ws, SIF__FILTER_TOP_HAT, radius, grid->box_length);
      if (status != SIF_OK)
        break;
      random_grid->values = sif__fft_grid_backward(ctx.random_ws);
      sif__exodus_survey_contrast(grid->values, random_grid->values,
        ctx.occupied, survey_alpha, grid->total_cells);
    }

    sif_refine_stats_t refine = {0};
    const int plain = !ctx.seed_minima;
    if (plain) {
      status =
        sif__finder_scan_candidates(grid, ctx.mask, threshold, &ctx.candidates);
    } else {
      /* Candidates below the threshold on the top-hat field, minima of it
       * smoothed at the rung's scale -- one more transform. */
      resmooth_t rs = {.ws = ctx.fft_ws,
        .grid = grid,
        .radius = radius,
        .sigma = RUNG_MINIMA_SIGMA * radius};
      status =
        sif__finder_scan_refined(grid, ctx.mask, ctx.below, threshold, survey,
          1, 1, &ctx.candidates, &ctx.offsets, &refine, resmooth_rung, &rs);
    }
    if (status != SIF_OK)
      break;

    uint64_t n_candidates = ctx.candidates.count;
    const double t_scanned = wall_s();
    double t_evaluate = 0.0, t_commit = 0.0;
    uint64_t n_batches = 0;
    stats.n_candidates = n_candidates;

    if (!plain) {
      SIF_LOG_TRACE(
        TAG, "local minima:              %7" PRIu64, refine.n_minima);
      SIF_LOG_TRACE(
        TAG, "minima left on their node: %7" PRIu64, refine.n_fallback);
    }

    const sif_real rmin = RMIN_FACTOR * radius;

    /* The reach past the rung, from SIF_FINDER_SEARCH_*. The three-cell floor
     * is kept underneath it: at the small end of a ladder the factor can ask
     * for less than the grid can resolve, and a shell thinner than a few cells
     * bins nothing useful. At factor 2 this is the expression it replaces. */
    const sif_real search_factor = (sif_real)sif__finder_search_factor(options);
    sif_real r_search = radius + SIF_REAL_MAX((search_factor - 1.0f) * radius,
                                   3.0f * grid->cell_length);

    /*
     * Consecutive rungs of the radius ladder have to overlap in the void sizes
     * they can return, or sizes in between are reachable at no rung at all and
     * simply never appear in the catalogue.
     *
     * What a rung reaches is [radius, r_search] in the crossing radius, since
     * detection puts its own floor at the rung (see RMIN_FACTOR). So the
     * previous, larger rung returned nothing below r_prev, and this one has to
     * reach up to r_prev to meet it -- no further, and no less.
     *
     * Stated in the crossing radius rather than in the void's geometric size,
     * which is what an earlier form of this guard used: the smallest void the
     * previous rung could see had size |t|^(1/3) * r_prev, but the radius it
     * would have reported for that void is that size divided by |t|^(1/3),
     * which is r_prev again. Reaching only to the size leaves the band between
     * them at no rung at all.
     *
     * This is what makes SIF_FINDER_SEARCH_* safe to lower: whatever factor is
     * asked for, a rung still reaches its predecessor, so no size range can
     * fall between two rungs. The factor only decides how far PAST that a rung
     * keeps looking -- which is reach the larger rung has already covered, and
     * which costs factor^3 - 1 to provide. It binds whenever the ladder steps
     * by more than the factor: at 2.0 almost never, at 1.5 still almost never
     * for any ladder finer than a 50% step. Erring wide costs time, erring
     * narrow loses voids, and this line is why narrow cannot.
     */
    if (i > 0 && ctx.sorted_radii[i - 1] > r_search)
      r_search = ctx.sorted_radii[i - 1];

    mesh_template_t tpl;
    status = sif__exodus_template_build(
      &tpl, ctx.mesh->n_cells, ctx.mesh->cell_length, rmin, r_search);
    if (status != SIF_OK)
      break;

    /* The first level of the extended reach this rung enters: the lowest
     * EXT_SLACK past its own reach, or the top if none is. */
    uint32_t ext0 = n_ext;
    for (uint32_t k = 0; k < n_ext && ext0 == n_ext; k++)
      if (ext_reach[k] >= EXT_SLACK * r_search)
        ext0 = k;
    if (ext0 == n_ext && n_ext > 0 && ext_reach[n_ext - 1] > r_search)
      ext0 = n_ext - 1;

    /* The random mesh is free to use cells of its own, so it gets its own
     * template. Zeroed for a box run, where sif__exodus_template_free() makes
     * it a no-op. */
    mesh_template_t rtpl = {0};
    if (survey)
      status = sif__exodus_template_build(
        &rtpl, random_mesh->n_cells, random_mesh->cell_length, rmin, r_search);
    if (status != SIF_OK) {
      sif__exodus_template_free(&tpl);
      break;
    }

    const int32_t proxy_pole =
      (int32_t)(radius * (1.0f - overlap_fraction) / grid->cell_length) - 1;

    /* The collision witness, see collision_radius(): for the box's unweighted
     * kernel, the one it mirrors. The core is the rescaling's own. */
    const int witness = !survey && !weighted;
    const sif_real r_core2 = rmin * rmin;
    const sif_real witness_min_rj =
      ctx.seed_minima ? r_search * (1.0f + (sif_real)COLLISION_SLACK) : 0.0f;

    if (!ctx.seed_minima) {
      const double t_filter = wall_s();
      status = prefilter_cells(
        &ctx, grid, radius, overlap_fraction, proxy_pole, &stats);
      t_evaluate += wall_s() - t_filter;
      if (status != SIF_OK) {
        sif__exodus_template_free(&tpl);
        sif__exodus_template_free(&rtpl);
        break;
      }
      n_candidates = ctx.candidates.count;
    }

    /* The window is re-learned per radius: the accept rate at the largest
     * radius says little about the smallest. */
    uint64_t batch_floor = (uint64_t)BATCH_PER_THREAD * (uint64_t)ctx.n_threads;
    if (batch_floor < BATCH_FLOOR)
      batch_floor = BATCH_FLOOR;
    if (batch_floor > BATCH_MAX)
      batch_floor = BATCH_MAX;

    uint64_t batch_cap = batch_floor;

    /* Re-learned per radius alongside the batch window: the annulus geometry,
     * and with it the bin thickness, changes with every smoothing radius. */
    uint32_t resolve_window = RESOLVE_INITIAL;

    uint64_t k = 0;
    while (k < n_candidates && status == SIF_OK) {
      uint64_t batch_count = 0;

      /* Phase 0: skip already-masked candidates sequentially. Doing this
       * outside the parallel region avoids paying thread sync for millions of
       * trivially dead candidates. */
      while (k < n_candidates && batch_count < batch_cap) {
        if (masked_overlap(
              &ctx, k, radius, overlap_fraction, grid->cell_length)) {
          stats.rejected_masked++;
        } else {
          ctx.batch_indices[batch_count++] = k;
        }
        k++;
      }

      if (batch_count == 0)
        break;
      n_batches++;
      const double t_batch = wall_s();

      /* Phase 1: evaluate the batch in parallel against the catalogue as it
       * stood when the batch started. Read-only on mask and catalogue.
       *
       * The team size is pinned so that omp_get_thread_num() can never index
       * past the per-thread shell buffers allocated in ctx_init. */
#pragma omp parallel for schedule(dynamic, 16) num_threads(ctx.n_threads)
      for (uint64_t b = 0; b < batch_count; b++) {
        const int tid = sif__system_thread_num();
        const uint64_t cand = ctx.batch_indices[b];
        const uint64_t flat = candidate_node(&ctx, cand);
        batch_result_t* res = &ctx.batch_results[b];

        uint32_t ix, iy, iz;
        sif__unflatten_index(grid, flat, &ix, &iy, &iz);

        res->ix = ix;
        res->iy = iy;
        res->iz = iz;

        if (proxy_pole > 0 &&
            sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask, ix, iy,
              iz, (uint32_t)proxy_pole)) {
          res->status = BATCH_REJECTED_PROXY;
          continue;
        }

        /* The pole probe above read the node, which is within half a cell
         * of the centre on every axis. A masked pole p cells out is then
         * within sqrt((p + 1/2)^2 + 1/2) < p + 1 cells of the centre, and the
         * "- 1" in proxy_pole is that cell: the probe still only rejects a
         * candidate that certainly overlaps. */
        sif_real c[3];
        candidate_centre(&ctx, cand, grid, ix, iy, iz, c);
        const sif_real cx = c[0];
        const sif_real cy = c[1];
        const sif_real cz = c[2];

        res->cx = cx;
        res->cy = cy;
        res->cz = cz;

        if (sif__overlap_exact(ctx.cat, cx, cy, cz, radius, ctx.max_accepted_r,
              grid->box_length, ctx.void_cll, grid->p2_mask,
              overlap_fraction)) {
          res->status = BATCH_REJECTED_MESH;
          continue;
        }

        /* A collision certain at any radius past rho, and a rescaling
         * certain to return more: the re-check would refuse it. The
         * catalogue only gains voids before this candidate's commit (a
         * displacement ends the batch), so a collision here is one there.
         * Under the minima the void has to be one that refuses rather than
         * yields, larger than anything the closed sphere can return. */
        if (witness) {
          const double rho = collision_radius(&ctx, cx, cy, cz, r_search,
            witness_min_rj, overlap_fraction, grid->box_length);
          if (rho < (double)r_search) {
            sif_real rho2 = (sif_real)(rho * rho * (1.0 + COLLISION_SLACK));
            if (rho2 < r_core2)
              rho2 = r_core2;
            if (sif__exodus_witness(ctx.mesh, cx, cy, cz, rho2, r_search,
                  threshold, vol_factor, &tpl, ctx.seed_minima)) {
              res->status = BATCH_REJECTED_WITNESS;
              continue;
            }
          }
        }

        uint8_t reason = RESCALE_OK;
        sif_real r_scaled;

        if (survey) {
          const mesh_query_t dq = survey_query(ctx.mesh, &tpl, cx, cy, cz);
          const mesh_query_t rq = survey_query(random_mesh, &rtpl, cx, cy, cz);
          r_scaled = sif__exodus_find_radius_survey(&dq, &rq, r_search,
            survey_K, &ctx.scratch[tid], rmin, resolve_window, &reason);
        } else if (weighted) {
          r_scaled = sif__exodus_find_radius_weighted(ctx.mesh, cx, cy, cz,
            r_search, threshold, vol_factor, mass_factor, &ctx.scratch[tid],
            rmin, &tpl, resolve_window, &reason);
        } else {
          r_scaled = sif__exodus_find_radius(ctx.mesh, cx, cy, cz, r_search,
            threshold, vol_factor, &ctx.scratch[tid], rmin, &tpl,
            resolve_window, &reason);
        }

        for (uint32_t e = ext0;
          e < n_ext && r_scaled < 0.0f && reason == RESCALE_BEYOND_SEARCH;
          e++) {
          reason = RESCALE_OK;
          r_scaled =
            weighted ? sif__exodus_find_radius_weighted(ctx.mesh, cx, cy, cz,
                         ext_reach[e], threshold, vol_factor, mass_factor,
                         &ctx.scratch[tid], ext_rmin[e], &etpl[e],
                         resolve_window, &reason)
                     : sif__exodus_find_radius(ctx.mesh, cx, cy, cz,
                         ext_reach[e], threshold, vol_factor, &ctx.scratch[tid],
                         ext_rmin[e], &etpl[e], resolve_window, &reason);
        }

        if (r_scaled < 0.0f) {
          res->rescale_reason = reason;
          res->status = BATCH_REJECTED_RESCALE;
          continue;
        }

        res->r_scaled = r_scaled;
        res->status = BATCH_ACCEPTED;
      }

      const double t_evaluated = wall_s();
      t_evaluate += t_evaluated - t_batch;

      /* Phase 2: commit sequentially, re-checking everything that the parallel
       * phase could not have seen (voids accepted earlier in this same
       * batch). */
      const uint64_t accepted_before = stats.accepted;
      int displaced = 0;

      for (uint64_t b = 0; b < batch_count && !displaced; b++) {
        const batch_result_t* res = &ctx.batch_results[b];

        switch (res->status) {
        case BATCH_REJECTED_PROXY:
          stats.rejected_proxy++;
          continue;
        case BATCH_REJECTED_MESH:
          stats.rejected_overlap++;
          continue;
        case BATCH_REJECTED_RESCALE:
          stats.rejected_rescale++;
          rescale.fail[res->rescale_reason]++;
          continue;
        case BATCH_REJECTED_WITNESS:
          stats.rejected_witness++;
          continue;
        default:
          break;
        }

        /* Past the switch the rescaling converged, whatever the re-checks
         * below go on to decide about the void itself. */
        sif__exodus_rescale_report_add(&rescale, res->r_scaled / radius);

        /* Under the minima a void is refused only by a larger one, and
         * displaces the smaller ones it collides with; see find_conflicts().
         * Phases 0 and 1 tested the candidate against the catalogue as the
         * batch began, and the displacement below can take out a void those
         * tests would have refused it by, so they are repeated against the
         * catalogue as it stands: the outcome is then the one a strictly
         * sequential pass would give, however the batches fell. The mask
         * tests further down would refuse a void without asking which is
         * larger, so they are left to the cell seeds. */
        if (ctx.seed_minima) {
          if (masked_overlap(&ctx, ctx.batch_indices[b], radius,
                overlap_fraction, grid->cell_length)) {
            stats.rejected_masked++;
            continue;
          }
          if (proxy_pole > 0 &&
              sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask,
                res->ix, res->iy, res->iz, (uint32_t)proxy_pole)) {
            stats.rejected_proxy++;
            continue;
          }
          if (sif__overlap_exact(ctx.cat, res->cx, res->cy, res->cz, radius,
                ctx.max_accepted_r, grid->box_length, ctx.void_cll,
                grid->p2_mask, overlap_fraction)) {
            stats.rejected_overlap++;
            continue;
          }

          sif_real largest;
          status = find_conflicts(&ctx, res->cx, res->cy, res->cz,
            res->r_scaled, overlap_fraction, grid->box_length, &largest);
          if (status != SIF_OK)
            break;
          if (largest >= res->r_scaled) {
            stats.rejected_exact++;
            continue;
          }
          for (uint64_t k = 0; k < ctx.n_conflicts && status == SIF_OK; k++)
            status = displace_void(&ctx, grid, ctx.conflicts[k]);
          if (status != SIF_OK)
            break;
          stats.displaced += ctx.n_conflicts;

          /* Accepts only add to the catalogue, so a rejection against the
           * batch's frozen one still holds once they are in; a displacement
           * takes voids out, and the rest of the batch was refused by voids
           * that may now be gone. It ends the batch: the next one scans
           * again from the candidate after this, against the catalogue as
           * it now stands, so where the batches fell -- which follows the
           * thread count -- cannot decide what is found. */
          if (ctx.n_conflicts > 0) {
            const uint64_t next = ctx.batch_indices[b] + 1;
            /* Phase 0's refusals past this one are counted again there. */
            stats.rejected_masked -= (k - next) - (batch_count - b - 1);
            k = next;
            displaced = 1;
          }
          goto accept;
        }

        if (masked_overlap(&ctx, ctx.batch_indices[b], res->r_scaled,
              overlap_fraction, grid->cell_length)) {
          stats.rejected_masked++;
          continue;
        }

        /* Both overlap tests run again here, because the catalogue has moved on
         * since the batch was evaluated -- but at the rescaled radius, which
         * is the sphere the candidate is actually claiming. The pole distance
         * phase 1 used belonged to the rung, and the rung is only a lower
         * bound on that. */
        const int32_t exact_pole =
          (int32_t)(res->r_scaled * (1.0f - overlap_fraction) /
                    grid->cell_length) -
          1;
        if (exact_pole > 0 &&
            sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask, res->ix,
              res->iy, res->iz, (uint32_t)exact_pole)) {
          stats.rejected_proxy_recheck++;
          continue;
        }

        if (sif__overlap_exact(ctx.cat, res->cx, res->cy, res->cz,
              res->r_scaled, ctx.max_accepted_r, grid->box_length, ctx.void_cll,
              grid->p2_mask, overlap_fraction)) {
          stats.rejected_exact++;
          continue;
        }

      accept:
        status =
          accept_void(&ctx, grid, res->cx, res->cy, res->cz, res->r_scaled);
        if (status != SIF_OK)
          break;
        stats.accepted++;
        n_refined += ctx.candidates.items[ctx.batch_indices[b]].refined != 0;

        const sif_real ratio = res->r_scaled / radius;
        ratio_sum += (double)ratio;
        if (stats.accepted == 1 || ratio < ratio_min)
          ratio_min = ratio;
        if (ratio > ratio_max)
          ratio_max = ratio;
        if (ratio > MISMATCH_RATIO)
          n_mismatched++;
      }

      t_commit += wall_s() - t_evaluated;

      /* Retune the window. Nothing accepted means nothing was invalidated, so
       * the speculation was free and can be widened; an accept means the tail
       * of this batch was wasted, and the more accepts the further the window
       * overshot. */
      const uint64_t n_accepted = stats.accepted - accepted_before;

      if (n_accepted == 0) {
        batch_cap += batch_floor;
        if (batch_cap > BATCH_MAX)
          batch_cap = BATCH_MAX;
      } else {
        batch_cap /= (n_accepted + 1);
        if (batch_cap < batch_floor)
          batch_cap = batch_floor;
      }

      /* Resize the refinement window from the spans the batch just reported.
       * Reading the per-thread counters here is safe: the parallel region has
       * closed, and the window is only ever written outside it. Scans that
       * found nothing are deliberately not counted -- they walk the bins to the
       * bottom, so their span says nothing about where answers live, and a
       * wider window only ever helps them. */
      uint64_t span_sum = 0, span_count = 0;
      for (int t = 0; t < ctx.n_threads; t++) {
        span_sum += ctx.scratch[t].span_sum;
        span_count += ctx.scratch[t].span_count;
        ctx.scratch[t].span_sum = 0;
        ctx.scratch[t].span_count = 0;
      }

      if (span_count > 0) {
        uint64_t w = (span_sum * RESOLVE_MARGIN + span_count - 1) / span_count;
        if (w < RESOLVE_MIN)
          w = RESOLVE_MIN;
        if (w > RESOLVE_MAX)
          w = RESOLVE_MAX;
        resolve_window = (uint32_t)w;
      }
    }

    sif__exodus_template_free(&tpl);
    sif__exodus_template_free(&rtpl);

    sif_timer_stop(&timer);

    sif__exodus_rescale_report_log(&rescale);
    SIF_LOG_TRACE(TAG,
      "wall time:                 scan %.2fs, evaluate %.2fs in %" PRIu64
      " batches, commit %.2fs",
      t_scanned - t_rung, t_evaluate, n_batches, t_commit);

    if (ctx.seed_minima)
      SIF_LOG_TRACE(TAG, "accepted between the nodes: %6" PRIu64, n_refined);

    /* The same ratio again, but over the voids that survived to the catalogue:
     * that is the population the localisation warning below is about. */
    if (stats.accepted > 0) {
      SIF_LOG_TRACE(TAG,
        "accepted r/R:              min %.2f, mean %.2f, max %.2f  (%" PRIu64
        " beyond %.1fx)",
        (double)ratio_min, ratio_sum / (double)stats.accepted,
        (double)ratio_max, n_mismatched, (double)MISMATCH_RATIO);
    }

    total_mismatched += n_mismatched;

    if (status == SIF_OK)
      status = compact_catalogue(&ctx);

    sif__finder_log_radius(TAG, radius, &stats, ctx.cat->n_voids,
      sif_timer_elapsed_ms(&timer) / 1000.0);

    if (status == SIF_OK)
      checkpoint_rung(&ck, &ctx, i + 1, total_mismatched);

    if (status == SIF_OK && halt_after == i + 1) {
      SIF_LOG_ERROR(
        TAG, "halted after rung %u of %u (test hook)", i + 1, n_radii);
      status = SIF_ERR_INVALID;
    }
  }

  for (uint32_t k = 0; k < n_ext; k++)
    sif__exodus_template_free(&etpl[k]);

  /* Restored after a failure too: the caller gets back the grid they gave,
   * not the last rung's smoothing of it. */
  if (!(options & SIF_FINDER_CONSUME_GRID)) {
    /* Under save_memory this reads the grid back from its scratch file, and
     * can fail. The grid would then hold a half-written transform, so the run
     * fails rather than hand it back as restored; its checkpoint is kept, and
     * resuming from it returns the catalogue at once. */
    int restored = sif__fft_apply_filter(ctx.fft_ws, SIF__FILTER_NONE, 0, 0);
    if (restored == SIF_OK) {
      grid->values = sif__fft_grid_backward(ctx.fft_ws);
      SIF_LOG_INFO(TAG, "recovered original density grid");
    }
    if (survey && restored == SIF_OK) {
      restored = sif__fft_apply_filter(ctx.random_ws, SIF__FILTER_NONE, 0, 0);
      if (restored == SIF_OK) {
        random_grid->values = sif__fft_grid_backward(ctx.random_ws);
        SIF_LOG_INFO(TAG, "recovered original random grid");
      }
    }
    if (restored != SIF_OK && status == SIF_OK) {
      SIF_LOG_ERROR(TAG, "the grid could not be restored; failing the run");
      status = restored;
    }
  }

  /* A partial catalogue is not handed back: it would look like a run that
   * found fewer voids. */
  if (status != SIF_OK) {
    checkpoint_finish(&ck, status);
    sif__error_status_set(status);
    ctx_release(&ctx, grid);
    return NULL;
  }

  if (total_mismatched > 0 && ctx.cat->n_voids > 0) {
    SIF_LOG_WARNING(TAG,
      "%" PRIu64 " of %" PRIu64
      " voids (%.1f%%) are more than %.1fx the radius of the rung that found "
      "them; their centres are localised at that rung's scale, not their own. "
      "A denser radius ladder would place them better",
      total_mismatched, ctx.cat->n_voids,
      100.0 * (double)total_mismatched / (double)ctx.cat->n_voids,
      (double)MISMATCH_RATIO);
  }

  sif_catalogue_trim(ctx.cat);

  if (survey &&
      sif__exodus_survey_footprint(ctx.cat, ctx.occupied, grid) != SIF_OK) {
    checkpoint_finish(&ck, SIF_ERR_ALLOC);
    sif__error_status_set(SIF_ERR_ALLOC);
    ctx_release(&ctx, grid);
    return NULL;
  }

  checkpoint_finish(&ck, SIF_OK);

  sif_catalogue_t* result = ctx.cat;
  ctx.cat = NULL; /* ownership passes to the caller */
  ctx_release(&ctx, grid);

  return result;
}

/*
 * box / cell, rounded up -- except that a ratio within GRID_CELLS_SLACK of an
 * integer is taken as that integer. A cell meant to divide the box rarely
 * does so exactly once it has been through sif_real: 1 mps on a 256^3 box is
 * box / cbrt(16777216), and a cube root a few ulps short would turn 256 cells
 * into 257. Float rounding is ~1e-7 a step; 1e-5 covers a chain of them and
 * is still far below any cell size anyone asks for on purpose.
 */
#define GRID_CELLS_SLACK 1e-5

uint32_t sif_finder_grid_cells_from_size(uint64_t n_particles,
  sif_real box_length, sif_real cell_size, sif_finder_units_t units,
  sif_real volume) {

  /* classify_weight() tests the bits, so a NaN or an infinity is refused
   * under -ffast-math too; zero passes it and is caught by the sign test. */
  if (classify_weight(box_length) != WEIGHT_OK || !(box_length > 0.0f)) {
    SIF_LOG_ERROR(
      TAG, "box_length: %g, expected positive and finite", (double)box_length);
    return 0;
  }
  if (classify_weight(cell_size) != WEIGHT_OK || !(cell_size > 0.0f)) {
    SIF_LOG_ERROR(
      TAG, "cell_size: %g, expected positive and finite", (double)cell_size);
    return 0;
  }
  if (units != SIF_FINDER_UNITS_PHYSICAL && units != SIF_FINDER_UNITS_MPS) {
    SIF_LOG_ERROR(TAG,
      "units: %d, expected SIF_FINDER_UNITS_PHYSICAL or "
      "SIF_FINDER_UNITS_MPS",
      (int)units);
    return 0;
  }
  if (units == SIF_FINDER_UNITS_MPS && n_particles == 0) {
    SIF_LOG_ERROR(TAG, "n_particles: 0, a cell size in mps needs tracers");
    return 0;
  }
  if (units == SIF_FINDER_UNITS_MPS && classify_weight(volume) != WEIGHT_OK) {
    SIF_LOG_ERROR(TAG,
      "volume: %g, expected positive and finite, or 0 for the "
      "box",
      (double)volume);
    return 0;
  }

  double cell = (double)cell_size;
  if (units == SIF_FINDER_UNITS_MPS) {
    const double box = (double)box_length;
    const double v = volume > 0.0f ? (double)volume : box * box * box;
    cell *= cbrt(v / (double)n_particles);
  }

  const double ratio = (double)box_length / cell;
  const double n = ceil(ratio * (1.0 - GRID_CELLS_SLACK));

  if (n > (double)UINT32_MAX) {
    SIF_LOG_ERROR(TAG,
      "cell_size: %g of a %g box is %.3g cells per side "
      "(expected at most %" PRIu32 ")",
      cell, (double)box_length, n, UINT32_MAX);
    return 0;
  }

  return n < 1.0 ? 1u : (uint32_t)n;
}

#undef GRID_CELLS_SLACK
