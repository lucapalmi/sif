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
 * rho_mean the mean weight density, and nothing below changes but the sums;
 * see find_exact_radius_impl(). Written out, that asks for the k-th nearest
 * distance for every k, which is a full sort of everything inside the search
 * sphere: at r_search = 200 in a 2250 box, ten million distances sorted per
 * candidate, and there are millions of candidates.
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
 */

#include "sif/finder/exodus_finder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/chain_mesh.h"

#include "core/system_internal.h"

#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"
#include "sif/utils/timer.h"

#include "utils.h"

/* A distance collected by the refinement on a weighted mesh, with the weight
 * that has to leave the enclosed total when the walk passes it. */
typedef struct {
  sif_real d2;
  sif_real w;
} refine_pair_t;

SIF_DEFINE_QUICKSORT(sort_radii_desc, sif_real, a > b)
SIF_DEFINE_QUICKSORT(sort_real_asc, sif_real, a < b)
SIF_DEFINE_QUICKSORT(sort_pairs_asc, refine_pair_t, a.d2 < b.d2)

#define TAG "exodus_finder"

/*
 * Speculation window.
 *
 * A batch is evaluated in parallel against the catalog as it stood when the
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
 * are committed, so the catalog is identical for any window.
 */
#define BATCH_MAX 4096u

/* Below a few candidates per thread the parallel region stops paying for
 * itself, whatever the accept rate says. */
#define BATCH_PER_THREAD 8u
#define BATCH_FLOOR      32u

/* Coarse grid for the void-vs-void overlap index. */
#define VOID_CLL_CELLS 32

#define CATALOG_INITIAL_CAPACITY 250000

/*
 * Floor on the rescaled radius, as a fraction of the rung that found the void.
 * Fixed, so that a catalog depends only on the grid, the radius ladder and the
 * threshold.
 *
 * At 1 it coincides with the detection limit rather than sitting below it. A
 * void of geometric size p registers at rung R only once p >= |t|^(1/3) R, and
 * its crossing radius is p * |t|^(-1/3) -- so a cell is a candidate at R
 * exactly when its crossing radius is at least R. A rung can therefore never
 * legitimately return less than it, and any floor under R fences off a range
 * that only measurement error can reach: a top-hat evaluated on a grid whose
 * cells are not small against R reads a little too deep, and the tracers then
 * decline to support the radius the grid implied.
 *
 * Rejecting those costs nothing, because the same void meets the next rung
 * down with a crossing radius comfortably inside its own bracket, and gets a
 * centre smoothed at its own scale into the bargain. What it buys is that a
 * radius in the catalog is never one the grid and the tracers disagreed about.
 *
 * The enclosed-density condition is the only thing that decides a radius;
 * nothing here or below asks how the tracers inside are arranged.
 */
#define RMIN_FACTOR 1.00f

/*
 * --- Per-thread scratch for the radial pass ---
 *
 * The rescaling evaluates the enclosed density on a histogram of squared
 * distances instead of on the distances themselves, so what a thread has to
 * hold stops scaling with how many particles the search sphere contains. At
 * r_search = 200 in a 2250 box that used to be ~10^7 floats -- 38 MiB per
 * thread, allocated in every thread, sorted in full for every candidate.
 *
 * The histogram alone would only localize the answer to a bin, so the exact
 * radius still comes from particles: once the walk reaches the bin that holds
 * the crossing, that one bin is re-scanned and sorted. Bins are uniform in
 * d^2, which keeps square roots out of the hot loop entirely, and the template
 * carries per-cell distance bounds so the re-scan only visits the cells that
 * actually reach into the bin.
 */

/* Roughly this many particles per bin. Sizing the histogram to the expected
 * occupancy keeps it (and the memset that clears it) small when the search
 * sphere is nearly empty, which is the common case at small radii. */
#define PARTICLES_PER_BIN 16.0f

#define MAX_RADIAL_BINS 2048u
#define MIN_RADIAL_BINS 64u

/* Initial capacity of the refinement buffer, which only ever holds the
 * contents of a single bin. */
#define REFINE_INITIAL_CAPACITY 4096u

/* Distances are computed in tiles so the arithmetic stays vectorizable even
 * though the histogram update that follows it cannot be. */
#define DIST_TILE 64u

/*
 * Refinement window.
 *
 * The histogram keeps counts, not distances, so the exact radius has to be
 * recovered from particles once the walk reaches the bin that holds it. A bin
 * is a far thinner shell than a mesh cell -- at r = 40 it is about 1/256 of
 * one -- so recovering it means visiting every cell the shell passes through
 * and recomputing every distance in them, then keeping the handful that landed
 * in the bin. Measured, that is 68k distances computed to keep 141.
 *
 * Adjacent bins are adjacent shells, so consecutive recoveries repeat almost
 * exactly the same traversal. Resolving a window of W neighbouring bins in one
 * pass therefore costs what resolving one costs -- W bins together are still
 * thinner than a cell, so the same cells are visited either way -- and spares
 * the repeats. Measured, it takes the recoveries per scan from 4.0 to 1.0.
 *
 * W is not a constant: the bin count adapts to the expected occupancy, bins are
 * uniform in d^2 so their radial thickness varies with d, and the annulus
 * geometry changes with every smoothing radius. What W has to cover is how many
 * bins the walk spans before it finds its answer, which is measured directly --
 * every successful scan reports its span -- and W is then set to a multiple of
 * the mean.
 *
 * Widening past that buys nothing and starts costing: the collected buffer
 * grows with W and has to be sorted, and eventually the window does get thick
 * enough to pull in extra cells.
 */
#define RESOLVE_MARGIN  2u /* W = margin x mean span */
#define RESOLVE_MIN     1u
#define RESOLVE_MAX     64u
#define RESOLVE_INITIAL 8u

/*
 * A void is localized by the rung that found it: its center is the deepest
 * cell of the field smoothed at that rung's radius. When a void turns out to
 * be much larger than the rung, the smoothed field is flat across its whole
 * interior and the "deepest cell" is chosen out of noise -- the void is still
 * found, and with about the right radius, but pinned at an arbitrary point
 * inside itself.
 *
 * Nothing about that is visible in the catalog, so it is reported instead.
 * Beyond this ratio a void was found at a rung more than twice too small,
 * which is only reachable when the ladder-gap guard had to widen the search;
 * a ladder dense enough that every void meets a rung near its own scale never
 * trips it.
 */
#define MISMATCH_RATIO 2.0f

/*
 * A weighted mesh swaps both buffers for their weighted twins: the histogram
 * holds weight per bin instead of tracers per bin, and the refinement keeps
 * each distance's weight beside it. Only the pair a run needs is allocated, so
 * an unweighted run holds exactly what it always has.
 */
typedef struct {
  uint32_t* bins;        /* unweighted: tracers per bin */
  double* wbins;         /* weighted: weight per bin */
  sif_real* refine;      /* unweighted: squared distances */
  refine_pair_t* pairs;  /* weighted: squared distances and their weights */
  uint32_t refine_count; /* entries in whichever of the two is in use */
  uint32_t refine_capacity;

  /* How far down the bins the walk had to reach before the answer turned up,
   * summed over the scans that found one. This is what sizes the refinement
   * window; see RESOLVE_* below. Accumulated per thread and folded in during
   * the sequential commit, so no synchronization is needed. */
  uint64_t span_sum;
  uint64_t span_count;
} radial_scratch_t;

/*
 * Grows a refinement buffer of `elem_size` entries to hold at least `needed`,
 * keeping the first `count`. Returns the new buffer, or NULL with the old one
 * left intact.
 *
 * The buffer is filled incrementally, one mesh cell at a time, so a grow can
 * land in the middle of a bin's collection and whatever has been gathered so
 * far has to survive it. (It must: dropping it silently loses particles and
 * the walk then returns a radius that is too small. It only stays hidden
 * while a single bin fits in the initial capacity.)
 */
static void* refine_grow(void* old, uint32_t* capacity, uint32_t count,
  uint32_t needed, size_t elem_size) {

  uint32_t new_capacity = *capacity ? *capacity : REFINE_INITIAL_CAPACITY;
  while (new_capacity < needed) {
    if (new_capacity > UINT32_MAX / 2)
      return NULL;
    new_capacity *= 2;
  }

  void* grown = sif_malloc_aligned((size_t)new_capacity * elem_size);
  if (!grown)
    return NULL;

  if (count > 0)
    memcpy(grown, old, (size_t)count * elem_size);

  sif_free_aligned(old);
  *capacity = new_capacity;

  return grown;
}

static SIF_ALWAYS_INLINE int refine_reserve(
  radial_scratch_t* s, uint32_t needed, const int weighted) {

  if (needed <= s->refine_capacity)
    return SIF_OK;

  void* old = weighted ? (void*)s->pairs : (void*)s->refine;
  void* grown = refine_grow(old, &s->refine_capacity, s->refine_count, needed,
    weighted ? sizeof(refine_pair_t) : sizeof(sif_real));
  if (!grown)
    return SIF_ERR_ALLOC;

  if (weighted)
    s->pairs = grown;
  else
    s->refine = grown;

  return SIF_OK;
}

static void scratch_free(radial_scratch_t* s) {
  sif_free_aligned(s->bins);
  sif_free_aligned(s->wbins);
  sif_free_aligned(s->refine);
  sif_free_aligned(s->pairs);
  memset(s, 0, sizeof(*s));
}

static int scratch_init(radial_scratch_t* s, int weighted) {
  memset(s, 0, sizeof(*s));

  if (weighted)
    s->wbins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(double));
  else
    s->bins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(uint32_t));

  if (!s->bins && !s->wbins)
    return SIF_ERR_ALLOC;

  return refine_reserve(s, REFINE_INITIAL_CAPACITY, weighted);
}

/* --- Mesh traversal template --- */

/* Cell classification relative to the [rmin, r_search] annulus. */
#define CELL_FULLY_CORE   1 /* entirely inside rmin: count, never store */
#define CELL_STRADDLES    2 /* spans the rmin boundary: test each particle */
#define CELL_PARTIAL_EDGE 3 /* spans the r_search boundary: test each */
#define CELL_FULLY_SHELL  4 /* entirely inside the annulus: store all        */

typedef struct {
  int32_t* dx;
  int32_t* dy;
  int32_t* dz;
  uint8_t* type;
  /* Squared distance bounds of the cell relative to the query center. Kept so
   * the refinement pass can drop every cell that cannot reach into the bin it
   * is resolving, which turns that pass from a sphere scan into a shell scan.
   */
  sif_real* min_d2;
  sif_real* max_d2;
  uint32_t count;
} mesh_template_t;

static void template_free(mesh_template_t* tpl) {
  sif_free_aligned(tpl->dx);
  sif_free_aligned(tpl->dy);
  sif_free_aligned(tpl->dz);
  sif_free_aligned(tpl->type);
  sif_free_aligned(tpl->min_d2);
  sif_free_aligned(tpl->max_d2);
  memset(tpl, 0, sizeof(*tpl));
}

static int template_build(mesh_template_t* tpl, uint32_t mesh_n_cells,
  sif_real cell_length, sif_real rmin, sif_real r_search) {

  memset(tpl, 0, sizeof(*tpl));

  /*
   * The stencil is built once per radius and reused for every candidate, which
   * is only possible because it is expressed in cell *offsets* from whichever
   * cell the centre falls in. The distance bounds below are therefore taken
   * over every position the centre could occupy inside its own cell -- hence
   * the -1 and +1 -- which makes them conservative for any candidate and lets
   * one template serve millions of them.
   */
  const int32_t cell_radius = (int32_t)(r_search / cell_length) + 1;

  /*
   * The traversal wraps a cell index with a single add or subtract, which is
   * only enough while the stencil cannot reach more than once around the box.
   * Past that the index stays out of range and becomes a wild offset into
   * cell_offsets, so this is a hard precondition, not a quality concern.
   *
   * The bound is cell_radius < n_cells rather than <= : the center cell is
   * (int32_t)(cx / cell_length), which reaches n_cells for a cx that rounds up
   * to the box length, and that extra cell is what the strict inequality
   * covers.
   *
   * It only became reachable when the mesh started coming from the caller. A
   * coarse mesh and a large radius are each reasonable alone.
   */
  if (cell_radius >= (int32_t)mesh_n_cells) {
    SIF_LOG_ERROR(TAG,
      "the search sphere (r_search = %g, %d mesh cells) does not fit in a mesh "
      "of %u cells; use a finer mesh or smaller radii",
      (double)r_search, cell_radius, mesh_n_cells);
    return SIF_ERR_RANGE;
  }

  const uint64_t max_cells = (uint64_t)(2 * cell_radius + 1) *
                             (2 * cell_radius + 1) * (2 * cell_radius + 1);

  tpl->dx = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->dy = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->dz = sif_malloc_aligned(max_cells * sizeof(int32_t));
  tpl->type = sif_malloc_aligned(max_cells * sizeof(uint8_t));
  tpl->min_d2 = sif_malloc_aligned(max_cells * sizeof(sif_real));
  tpl->max_d2 = sif_malloc_aligned(max_cells * sizeof(sif_real));

  if (!tpl->dx || !tpl->dy || !tpl->dz || !tpl->type || !tpl->min_d2 ||
      !tpl->max_d2) {
    SIF_LOG_ERROR(TAG, "failed to allocate the mesh template");
    template_free(tpl);
    return SIF_ERR_ALLOC;
  }

  const sif_real r_search2 = r_search * r_search;
  const sif_real r_core2 = rmin * rmin;

  for (int32_t dx = -cell_radius; dx <= cell_radius; dx++) {
    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        /* Nearest and farthest a point in this cell can be from the center. */
        const sif_real min_x =
          (dx == 0) ? 0.0f : (SIF_REAL_ABS((sif_real)dx) - 1.0f) * cell_length;
        const sif_real min_y =
          (dy == 0) ? 0.0f : (SIF_REAL_ABS((sif_real)dy) - 1.0f) * cell_length;
        const sif_real min_z =
          (dz == 0) ? 0.0f : (SIF_REAL_ABS((sif_real)dz) - 1.0f) * cell_length;
        const sif_real min_dist2 =
          min_x * min_x + min_y * min_y + min_z * min_z;

        if (min_dist2 > r_search2)
          continue;

        const sif_real max_x =
          (SIF_REAL_ABS((sif_real)dx) + 1.0f) * cell_length;
        const sif_real max_y =
          (SIF_REAL_ABS((sif_real)dy) + 1.0f) * cell_length;
        const sif_real max_z =
          (SIF_REAL_ABS((sif_real)dz) + 1.0f) * cell_length;
        const sif_real max_dist2 =
          max_x * max_x + max_y * max_y + max_z * max_z;

        tpl->dx[tpl->count] = dx;
        tpl->dy[tpl->count] = dy;
        tpl->dz[tpl->count] = dz;
        tpl->min_d2[tpl->count] = min_dist2;
        tpl->max_d2[tpl->count] = max_dist2;

        /* The classification is what lets the histogram pass skip work: a cell
         * wholly inside the core needs no distances at all (it only adds to a
         * count), and one wholly inside the annulus needs no range test (every
         * particle in it is known to belong). Only the two boundary classes
         * pay for a test per particle. */
        if (max_dist2 <= r_core2) {
          tpl->type[tpl->count] = CELL_FULLY_CORE;
        } else if (min_dist2 > r_core2) {
          tpl->type[tpl->count] =
            (max_dist2 <= r_search2) ? CELL_FULLY_SHELL : CELL_PARTIAL_EDGE;
        } else {
          tpl->type[tpl->count] = CELL_STRADDLES;
        }
        tpl->count++;
      }
    }
  }

  return SIF_OK;
}

/* --- Radius rescaling --- */

/*
 * Everything a mesh traversal needs to turn a template entry into a particle
 * range. Both passes of the rescaling go through it, so they cannot disagree
 * about which particles a cell holds or which periodic image they are measured
 * against.
 */
typedef struct {
  const sif_chain_mesh_t* mesh;
  const mesh_template_t* tpl;
  sif_real cx, cy, cz;
  int32_t center_ix, center_iy, center_iz;
  int32_t n_cells;
  sif_real box_length;
} mesh_query_t;

/*
 * Resolves template entry `i` to its particle range, and to the periodic image
 * of the query center that range has to be measured against. Shifting the
 * center instead of the particles keeps the wrap at one addition per axis.
 * `flat` is the cell's index in the mesh, for anything kept per cell.
 *
 * @return 0 if the cell is empty
 */
static inline int query_cell(const mesh_query_t* q, uint32_t i,
  uint64_t* p_start, uint64_t* p_end, uint64_t* flat_out, sif_real* ex,
  sif_real* ey, sif_real* ez) {

  const int32_t N = q->n_cells;
  const sif_real box_L = q->box_length;

  int32_t ix = q->center_ix + q->tpl->dx[i];
  sif_real cx_eff = q->cx;
  if (ix < 0) {
    ix += N;
    cx_eff += box_L;
  } else if (ix >= N) {
    ix -= N;
    cx_eff -= box_L;
  }

  int32_t iy = q->center_iy + q->tpl->dy[i];
  sif_real cy_eff = q->cy;
  if (iy < 0) {
    iy += N;
    cy_eff += box_L;
  } else if (iy >= N) {
    iy -= N;
    cy_eff -= box_L;
  }

  int32_t iz = q->center_iz + q->tpl->dz[i];
  sif_real cz_eff = q->cz;
  if (iz < 0) {
    iz += N;
    cz_eff += box_L;
  } else if (iz >= N) {
    iz -= N;
    cz_eff -= box_L;
  }

  const uint64_t flat = (uint64_t)ix * N * N + (uint64_t)iy * N + iz;

  *p_start = q->mesh->cell_offsets[flat];
  *p_end = q->mesh->cell_offsets[flat + 1];
  *flat_out = flat;
  *ex = cx_eff;
  *ey = cy_eff;
  *ez = cz_eff;

  return *p_end > *p_start;
}

/*
 * The one place a squared distance is ever computed. Both passes call it, so
 * the histogram and the refinement classify a particle from bit-identical
 * inputs -- which is what lets the refinement trust the histogram's counts.
 */
static inline sif_real dist2(sif_real px, sif_real py, sif_real pz, sif_real cx,
  sif_real cy, sif_real cz) {
  const sif_real dx = px - cx;
  const sif_real dy = py - cy;
  const sif_real dz = pz - cz;
  return dx * dx + dy * dy + dz * dz;
}

/* Bin index for a squared distance, clamped: the classification of a cell is
 * geometric and conservative, so a particle can land a rounding error outside
 * the nominal range. */
static inline uint32_t bin_of(
  sif_real d2, sif_real r_core2, sif_real inv_bin_w, uint32_t n_bins) {

  const int32_t b = (int32_t)((d2 - r_core2) * inv_bin_w);
  if (b < 0)
    return 0;
  if ((uint32_t)b >= n_bins)
    return n_bins - 1;
  return (uint32_t)b;
}

#define BIN_FOUND     1
#define BIN_EXHAUSTED 0

/*
 * Resolves the exact radius inside a range of bins: re-scans the cells that
 * reach into them, sorts the squared distances that land there and walks them
 * outside-in. `current_N` is the number of particles at or below the top of
 * bin `bin_hi`, so the walk sees exactly the counts a full sorted scan would.
 * On a weighted mesh `current_W` is the same thing in weight, and is the one
 * that is read; the other is ignored.
 *
 * Widening the range from one bin to several does not change the answer. The
 * walk over the union in descending order is the same sequence of tests as
 * walking each bin in turn, and any bin swept in that would have failed its own
 * skip test contributes nothing: that test is a proof that none of its
 * particles qualify, so walking them costs comparisons and decides nothing.
 *
 * @return BIN_FOUND with *out set, BIN_EXHAUSTED if the range holds no
 * acceptable radius, or SIF_ERR_ALLOC if the refinement buffer could not grow
 */
static SIF_ALWAYS_INLINE int resolve_bin(const mesh_query_t* q,
  radial_scratch_t* scratch, uint32_t bin_lo, uint32_t bin_hi, sif_real lo,
  sif_real hi, sif_real r_core2, sif_real r_search2, sif_real inv_bin_w,
  uint32_t n_bins, uint64_t current_N, double current_W, sif_real K2,
  sif_real* out, const int weighted) {

  const sif_real* mx = SIF_ASSUME_ALIGNED(q->mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(q->mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(q->mesh->z);
  const sif_real* mw = weighted ? SIF_ASSUME_ALIGNED(q->mesh->weights) : NULL;
  const mesh_template_t* tpl = q->tpl;

  scratch->refine_count = 0;

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    /* A bin is a thin shell, so most of the template cannot touch it. */
    if (type == CELL_FULLY_CORE || tpl->max_d2[i] < lo || tpl->min_d2[i] > hi)
      continue;

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    if (refine_reserve(scratch,
          scratch->refine_count + (uint32_t)(p_end - p_start),
          weighted) != SIF_OK)
      return SIF_ERR_ALLOC;

    for (uint64_t p = p_start; p < p_end; p++) {
      const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);

      /* Cells that straddle a boundary carry particles outside the annulus. */
      if (type != CELL_FULLY_SHELL && !(d2 > r_core2 && d2 <= r_search2))
        continue;

      const uint32_t bi = bin_of(d2, r_core2, inv_bin_w, n_bins);
      if (bi < bin_lo || bi > bin_hi)
        continue;

      if (weighted)
        scratch->pairs[scratch->refine_count++] = (refine_pair_t){d2, mw[p]};
      else
        scratch->refine[scratch->refine_count++] = d2;
    }
  }

  if (scratch->refine_count > 1) {
    if (weighted)
      sort_pairs_asc(scratch->pairs, scratch->refine_count);
    else
      sort_real_asc(scratch->refine, scratch->refine_count);
  }

  /*
   * Outside-in over the exact distances, which is where the answer finally
   * comes from. current_N is everything at or below the top of the window, so
   * at each step the particle under test is the outermost one left and
   * current_N - 1 is what a sphere through it encloses -- the particle sitting
   * on the surface does not count as inside.
   *
   * Weighted, it is the same walk with the particle's weight in place of the
   * 1. The answer is still a particle distance: between two neighbouring
   * particles the enclosed weight is constant and the volume grows, so the
   * outermost radius that passes is always one where a particle sits. That
   * needs the weights to be non-negative, which the driver checks.
   */
  if (weighted) {
    for (int32_t k = (int32_t)scratch->refine_count - 1; k >= 0; k--) {
      const sif_real d2_test = scratch->pairs[k].d2;
      const double w_test = (double)scratch->pairs[k].w;

      if (d2_test == 0.0f) {
        current_W -= w_test;
        continue;
      }

      const sif_real w_in = (sif_real)(current_W - w_test);
      const sif_real d2_cube = d2_test * d2_test * d2_test;

      if (w_in * w_in <= K2 * d2_cube) {
        *out = SIF_REAL_SQRT(d2_test);
        return BIN_FOUND;
      }

      current_W -= w_test;
    }

    return BIN_EXHAUSTED;
  }

  for (int32_t k = (int32_t)scratch->refine_count - 1; k >= 0; k--) {
    const sif_real d2_test = scratch->refine[k];

    /* A particle exactly at the centre defines no radius, but it is still
     * inside every sphere considered below, so it leaves the count. */
    if (d2_test == 0.0f) {
      current_N--;
      continue;
    }

    const uint64_t n_in = current_N - 1;
    const sif_real d2_cube = d2_test * d2_test * d2_test;

    const sif_real rn_in = (sif_real)n_in;
    if (rn_in * rn_in <= K2 * d2_cube) {
      *out = SIF_REAL_SQRT(d2_test);
      return BIN_FOUND;
    }

    current_N--;
  }

  return BIN_EXHAUSTED;
}

/* --- Rescaling diagnostics --- */

/*
 * Why a rescaling produced no radius.
 *
 * The failures are not variations on one theme, and a bare count of them says
 * nothing about what to change. A crossing below the rung is a void smaller
 * than this rung can express; a search sphere still underdense at its outer
 * edge holds one larger than the rung can reach. Those two ask for opposite
 * corrections to the radius ladder -- extend it down, extend it up -- so they
 * are counted apart.
 */
typedef enum {
  RESCALE_OK = 0,
  RESCALE_EMPTY,         /* not one tracer inside r_search */
  RESCALE_BEYOND_SEARCH, /* still underdense out at r_search */
  RESCALE_BELOW_RUNG,    /* the crossing lies below the rung itself */
  RESCALE_DEGENERATE,    /* r_search <= rmin, so there is no annulus */
  RESCALE_ALLOC,         /* the refinement buffer could not grow */
  RESCALE_N_REASONS
} rescale_reason_t;

static const char* const RESCALE_REASON_LABEL[RESCALE_N_REASONS] = {
  "converged",
  "search sphere empty",
  "underdense out to r_search",
  "smaller than the rung",
  "degenerate search range",
  "refinement alloc failed",
};

/*
 * Buckets for the r/R histogram.
 *
 * The reachable range runs from 1 -- a rung is a lower bound on what it can
 * return, see RMIN_FACTOR -- to r_search / radius at the top, which is 2 or
 * more where the ladder-gap guard had to widen the search. The edges are
 * closely spaced near the bottom, where a well-matched ladder puts nearly
 * everything, and open out towards the tail that says the rung was far below
 * the void's own scale.
 */
#define RESCALE_N_BUCKETS 6

static const sif_real RESCALE_BUCKET_EDGE[RESCALE_N_BUCKETS - 1] = {
  1.05f, 1.10f, 1.25f, 1.50f, 2.00f};

static const char* const RESCALE_BUCKET_LABEL[RESCALE_N_BUCKETS] = {
  "1.00-1.05", "1.05-1.10", "1.10-1.25", "1.25-1.50", "1.50-2.00", ">=2.00"};

/* What every rescaling at one rung did, successes and failures alike. */
typedef struct {
  uint64_t n_ok;
  uint64_t fail[RESCALE_N_REASONS];
  uint64_t bucket[RESCALE_N_BUCKETS];
  double ratio_sum;
  sif_real ratio_min;
  sif_real ratio_max;
} rescale_report_t;

/*
 * Record one successful rescaling, as the ratio of the radius it returned to
 * the rung that found the candidate. Every success is counted, including the
 * ones the overlap re-checks go on to reject: what is being described here is
 * the rescaling, not the catalog.
 */
static void rescale_report_add(rescale_report_t* rep, sif_real ratio) {
  if (rep->n_ok == 0 || ratio < rep->ratio_min)
    rep->ratio_min = ratio;
  if (rep->n_ok == 0 || ratio > rep->ratio_max)
    rep->ratio_max = ratio;

  rep->ratio_sum += (double)ratio;
  rep->n_ok++;

  uint32_t b = 0;
  while (b < RESCALE_N_BUCKETS - 1 && ratio >= RESCALE_BUCKET_EDGE[b])
    b++;
  rep->bucket[b]++;
}

/*
 * The per-rung breakdown, at TRACE. Reasons that did not occur are left out: a
 * rung usually hits two or three of them, and a fixed table of mostly zeros
 * buries the ones it did hit.
 */
static void rescale_report_log(const rescale_report_t* rep) {
  uint64_t n_failed = 0;
  for (int r = RESCALE_OK + 1; r < RESCALE_N_REASONS; r++)
    n_failed += rep->fail[r];

  const uint64_t n_tried = rep->n_ok + n_failed;
  if (n_tried == 0)
    return;

  SIF_LOG_TRACE(TAG,
    "rescalings attempted:      %7" PRIu64 "  (%" PRIu64 " failed, %.1f%%)",
    n_tried, n_failed, 100.0 * (double)n_failed / (double)n_tried);

  for (int r = RESCALE_OK + 1; r < RESCALE_N_REASONS; r++) {
    if (rep->fail[r] == 0)
      continue;

    SIF_LOG_TRACE(
      TAG, "  %-26s %7" PRIu64, RESCALE_REASON_LABEL[r], rep->fail[r]);
  }

  if (rep->n_ok == 0)
    return;

  SIF_LOG_TRACE(TAG, "rescaled r/R:              min %.2f, mean %.2f, max %.2f",
    (double)rep->ratio_min, rep->ratio_sum / (double)rep->n_ok,
    (double)rep->ratio_max);

  /* One line for the shape of it, so the three numbers above are read against
   * where the bulk actually sits. */
  char line[256];
  line[0] = '\0';
  int off = 0;

  for (uint32_t b = 0; b < RESCALE_N_BUCKETS; b++) {
    if (rep->bucket[b] == 0)
      continue;

    const int n = snprintf(line + off, sizeof(line) - (size_t)off,
      "%s%s %.1f%%", off ? " | " : "", RESCALE_BUCKET_LABEL[b],
      100.0 * (double)rep->bucket[b] / (double)rep->n_ok);

    if (n < 0 || (size_t)n >= sizeof(line) - (size_t)off)
      break;
    off += n;
  }

  SIF_LOG_TRACE(TAG, "r/R distribution:          %s", line);
}

/*
 * The rescaling itself: the largest radius at which the enclosed number
 * density is still at or below (1 + threshold) times the mean.
 *
 * Three stages, each one there to avoid work the next would otherwise do.
 *
 *   1. A histogram of the annulus in squared distance. This is the structure
 *      that replaces sorting: it answers "how many particles lie inside this
 *      shell" for every shell at once, in one pass and in fixed memory. The
 *      core is counted here too, since everything inside rmin is inside every
 *      radius this can return -- but only counted: how the tracers inside a
 *      void are arranged is not part of the condition.
 *
 *   2. The inward walk, which skips whole bins on counts alone (see the
 *      n_in_min test below).
 *
 *   3. Exact resolution of the surviving bin and its window, in resolve_bin().
 *
 * On a weighted mesh every count becomes a sum of weights and the density is
 * the enclosed weight over the mean weight density, `mass_factor` in place of
 * `vol_factor`. Nothing else about the three stages changes; see
 * find_exact_radius_weighted(). `vol_factor` stays a count either way, since
 * the one thing it still sizes -- the histogram, from the tracers expected per
 * bin -- is about how many tracers there are, not what they weigh.
 *
 * @param[out] reason RESCALE_OK, or which of the three stages gave up.
 *
 * @return the rescaled radius, or -1 if no acceptable radius exists
 */
static SIF_ALWAYS_INLINE sif_real find_exact_radius_impl(
  const sif_chain_mesh_t* mesh, sif_real cx, sif_real cy, sif_real cz,
  sif_real r_search, sif_real threshold, sif_real vol_factor,
  sif_real mass_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason,
  const int weighted) {

  *reason = RESCALE_OK;

  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mw = weighted ? SIF_ASSUME_ALIGNED(mesh->weights) : NULL;
  const sif_real* cw = weighted ? mesh->cell_weights : NULL;

  const sif_real inv_l = 1.0f / mesh->cell_length;
  const int32_t N = (int32_t)mesh->n_cells;

  mesh_query_t q = {
    .mesh = mesh,
    .tpl = tpl,
    .cx = cx,
    .cy = cy,
    .cz = cz,
    .center_ix = (int32_t)(cx * inv_l),
    .center_iy = (int32_t)(cy * inv_l),
    .center_iz = (int32_t)(cz * inv_l),
    .n_cells = N,
    .box_length = mesh->box_length,
  };

  const sif_real r_search2 = r_search * r_search;
  const sif_real r_core2 = rmin * rmin;

  /* 64-bit: at 2048^3 tracers a single search sphere in a small box holds
   * more than UINT32_MAX. 2048^3 is 8,589,934,592, which is exactly 2^33 --
   * two full wraps of a 32-bit counter -- so the overflow is not a distant
   * corner, it is where these runs live. See the note on total_N below. */
  uint64_t n_core = 0;
  uint64_t n_shell = 0;

  /* The weighted counterparts, in double for the reason total_N is 64-bit: a
   * float sum over billions of tracers stops moving long before the end. */
  double w_core = 0.0;
  double w_shell = 0.0;

  /* Histogram geometry. Uniform in d^2, so no square roots are needed to bin;
   * the resulting bins are finer in radius further out, which is the half that
   * matters since the walk starts from the outside. */
  const sif_real span = r_search2 - r_core2;
  uint32_t n_bins = MAX_RADIAL_BINS;
  {
    const sif_real expected_shell =
      vol_factor * (r_search * r_search * r_search - rmin * rmin * rmin);
    const sif_real want = expected_shell / PARTICLES_PER_BIN;

    /* The !(a >= b) form also catches a NaN estimate. */
    if (!(want >= (sif_real)MIN_RADIAL_BINS))
      n_bins = MIN_RADIAL_BINS;
    else if (want < (sif_real)MAX_RADIAL_BINS)
      n_bins = (uint32_t)want;
  }

  if (!(span > 0.0f)) {
    *reason = RESCALE_DEGENERATE;
    return -1.0f;
  }

  /*
   * The cheap refusal, before a single distance is binned.
   *
   * Most candidates at the small rungs do not have a crossing inside their
   * search sphere at all: they sit deep in a void far larger than the rung, so
   * the whole sphere is still underdense and the rung has to defer to a bigger
   * one. That verdict needs one number, the count inside r_search -- and the
   * count needs no distances for any cell the template has already placed
   * wholly inside the core or wholly inside the annulus, because every
   * particle in those is inside r_search by construction. Only the cells
   * straddling the outer edge have to be measured, and those are a shell one
   * cell thick: the pass is O(surface) where the histogram is O(volume).
   *
   * On a deep threshold's small rungs this is 88% of all rescalings, each of
   * which currently bins its entire sphere and then discards the histogram on
   * the test below.
   *
   * Nothing about the decision changes -- the same candidates are refused for
   * the same reason, and the count is assembled from the same three cases the
   * histogram loop uses, so it agrees with total_N exactly. Only the moment of
   * refusal moves earlier. The test after the histogram is left in place as a
   * backstop; it now only ever fires if these two disagree, which they cannot.
   *
   * Weighted, a whole cell's weight comes from sif_chain_mesh_t::cell_weights,
   * which is what keeps this O(surface). That sum and the histogram's are
   * taken in different orders, so the two can differ in the last bit and the
   * backstop may, rarely, be the one that decides.
   */
  {
    const sif_real expected_search =
      mass_factor * r_search * r_search * r_search;
    uint64_t n_within = 0;
    double w_within = 0.0;

    for (uint32_t i = 0; i < tpl->count; i++) {
      const uint8_t type = tpl->type[i];

      uint64_t p_start, p_end, flat;
      sif_real ex, ey, ez;
      if (!query_cell(&q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
        continue;

      if (type == CELL_FULLY_CORE || type == CELL_FULLY_SHELL) {
        if (weighted)
          w_within += (double)cw[flat];
        else
          n_within += (p_end - p_start);
        continue;
      }

      if (weighted) {
        for (uint64_t p = p_start; p < p_end; p++) {
          if (dist2(mx[p], my[p], mz[p], ex, ey, ez) <= r_search2)
            w_within += (double)mw[p];
        }
      } else {
        for (uint64_t p = p_start; p < p_end; p++)
          n_within += (dist2(mx[p], my[p], mz[p], ex, ey, ez) <= r_search2);
      }
    }

    /* On a weighted mesh "empty" means nothing inside carries any weight. */
    if (weighted ? !(w_within > 0.0) : n_within == 0) {
      *reason = RESCALE_EMPTY;
      return -1.0f;
    }

    const sif_real within =
      weighted ? (sif_real)w_within : (sif_real)n_within;
    if ((within / expected_search) - 1.0f <= threshold) {
      *reason = RESCALE_BEYOND_SEARCH;
      return -1.0f;
    }
  }

  const sif_real inv_bin_w = (sif_real)n_bins / span;
  const sif_real bin_w = span / (sif_real)n_bins;

  uint32_t* bins = scratch->bins;
  double* wbins = scratch->wbins;
  if (weighted)
    memset(wbins, 0, (size_t)n_bins * sizeof(double));
  else
    memset(bins, 0, (size_t)n_bins * sizeof(uint32_t));

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(&q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    const uint64_t p_count = p_end - p_start;

    if (type == CELL_FULLY_CORE) {
      if (weighted)
        w_core += (double)cw[flat];
      else
        n_core += p_count;
    } else if (type == CELL_FULLY_SHELL) {
      /* Every particle here is known to be in the annulus, so the only work is
       * the distance itself. Tiling keeps that part vectorized despite the
       * scattered counter update that follows. */
      sif_real tile[DIST_TILE];

      for (uint64_t base = p_start; base < p_end; base += DIST_TILE) {
        const uint32_t m = (uint32_t)((p_end - base < (uint64_t)DIST_TILE)
                                        ? (p_end - base)
                                        : (uint64_t)DIST_TILE);
#pragma omp simd
        for (uint32_t t = 0; t < m; t++) {
          tile[t] = dist2(mx[base + t], my[base + t], mz[base + t], ex, ey, ez);
        }

        if (weighted) {
          /* Summed per tracer rather than taken from the cell table, so the
           * total the walk subtracts from is made of exactly the weights the
           * bins and the refinement hold. */
          for (uint32_t t = 0; t < m; t++) {
            const double w = (double)mw[base + t];
            wbins[bin_of(tile[t], r_core2, inv_bin_w, n_bins)] += w;
            w_shell += w;
          }
        } else {
          for (uint32_t t = 0; t < m; t++)
            bins[bin_of(tile[t], r_core2, inv_bin_w, n_bins)]++;
        }
      }

      if (!weighted)
        n_shell += p_count;
    } else if (weighted) {
      for (uint64_t p = p_start; p < p_end; p++) {
        const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);
        const double w = (double)mw[p];

        if (d2 <= r_core2) {
          w_core += w;
        } else if (d2 <= r_search2) {
          wbins[bin_of(d2, r_core2, inv_bin_w, n_bins)] += w;
          w_shell += w;
        }
      }
    } else {
      for (uint64_t p = p_start; p < p_end; p++) {
        const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);

        n_core += (d2 <= r_core2);

        if (d2 > r_core2 && d2 <= r_search2) {
          bins[bin_of(d2, r_core2, inv_bin_w, n_bins)]++;
          n_shell++;
        }
      }
    }
  }

  /*
   * WHY THIS IS 64-BIT.
   *
   * The count is over the whole search sphere, and that sphere is not small
   * next to the box in a dense run: at 2048^3 tracers in a 120.6 box, a
   * sphere of r_search = 76.6 holds ~9.2e9 of them. A uint32 wraps at
   * 4.29e9, and 9.2e9 wraps to ~6.3e8 -- an eightfold undercount, which the
   * BEYOND_SEARCH test below reads as a contrast of -0.93 and refuses. Every
   * candidate on the rung refuses, the rung reports no voids, and nothing
   * about it looks like arithmetic: the log says 'underdense out to
   * r_search', which is exactly what a genuinely large void looks like.
   *
   * The threshold is a density, so it scales: the wrap needs ~4.3e9 tracers
   * inside one sphere, which needs a high number density, which means a
   * small box. At 2048^3 it bites above r_search = 59.4 in a 120.6 box and
   * above 115.5 in a 234.5 one; the larger boxes never come close.
   */
  const uint64_t total_N = n_core + n_shell;
  const double total_W = w_core + w_shell;
  if (weighted ? !(total_W > 0.0) : total_N == 0) {
    *reason = RESCALE_EMPTY;
    return -1.0f;
  }

  /* The mirror of the walk below: if the whole search sphere is still
   * underdense, the crossing lies outside it and this rung cannot say where. A
   * larger rung will. */
  const sif_real expected_search = mass_factor * r_search * r_search * r_search;
  const sif_real total = weighted ? (sif_real)total_W : (sif_real)total_N;
  if ((total / expected_search) - 1.0f <= threshold) {
    *reason = RESCALE_BEYOND_SEARCH;
    return -1.0f;
  }

  /* Walk inwards bin by bin. Comparisons are done on squared distances so no
   * square root is needed until the answer is found:
   *   n_in / (mass_factor * d^3) - 1 <= threshold
   *   <=> n_in^2 <= ((threshold + 1) * mass_factor)^2 * (d^2)^3
   * with n_in the enclosed weight on a weighted mesh.
   */
  const sif_real K = (threshold + 1.0f) * mass_factor;
  const sif_real K2 = K * K;

  uint64_t above = 0;   /* particles beyond the bin under examination */
  double above_w = 0.0; /* their weight, on a weighted mesh */
  int32_t b_first = -1; /* outermost bin the walk actually had to open */

  for (int32_t b = (int32_t)n_bins - 1; b >= 0; b--) {
    const uint32_t count = weighted ? 0u : bins[b];
    const double count_w = weighted ? wbins[b] : 0.0;

    /* A bin holding only weightless tracers is skipped like an empty one. None
     * of them can be the answer: the next tracer out encloses the same weight
     * at a larger radius, so it passes whenever they would, and the walk has
     * already seen it. */
    if (weighted ? !(count_w > 0.0) : count == 0)
      continue;

    sif_real hi = r_core2 + (sif_real)(b + 1) * bin_w;

    /* The top bin also absorbs anything the clamp pulled back in, so its upper
     * edge has to be generous. Being too generous only costs a refinement that
     * finds nothing; being too tight would skip a real answer. */
    if (b == (int32_t)n_bins - 1)
      hi = r_search2 * (1.0f + 1e-6f);

    /*
     * The skip test, and the reason the histogram is enough.
     *
     * Density falls as the radius grows and rises as the interior count grows,
     * so the most permissive particle this bin could possibly hold is one
     * sitting at the bin's outer edge with only the particles below the bin
     * inside it. Both of those are known from counters alone: `hi` is the
     * edge, and total_N - above - count is everything below the bin.
     *
     * If that best case is still too dense, every real particle in the bin is
     * worse -- each is closer in, or has more inside it, or both -- so the bin
     * cannot contain the crossing and not one of its distances is computed.
     *
     * Weighted, "more inside" is "at least as much weight inside", which holds
     * only while no weight is negative. That is why the driver refuses them.
     */
    const sif_real n_in_min = weighted
                                ? (sif_real)(total_W - above_w - count_w)
                                : (sif_real)(total_N - above - count);
    if (n_in_min * n_in_min > K2 * hi * hi * hi) {
      if (weighted)
        above_w += count_w;
      else
        above += count;
      continue;
    }

    if (b_first < 0)
      b_first = b;

    /* Resolve this bin together with the `window - 1` below it, in one
     * traversal of the cells they share. */
    int32_t b_lo = b - (int32_t)(window - 1);
    if (b_lo < 0)
      b_lo = 0;

    uint64_t window_count = 0;
    double window_w = 0.0;
    for (int32_t bb = b_lo; bb <= b; bb++) {
      if (weighted)
        window_w += wbins[bb];
      else
        window_count += bins[bb];
    }

    const sif_real win_lo = r_core2 + (sif_real)b_lo * bin_w;

    sif_real radius = -1.0f;
    const int status = resolve_bin(&q, scratch, (uint32_t)b_lo, (uint32_t)b,
      win_lo, hi, r_core2, r_search2, inv_bin_w, n_bins, total_N - above,
      total_W - above_w, K2, &radius, weighted);

    if (status == BIN_FOUND) {
      /* Report how far the walk had to reach, so the next batch can size the
       * window to cover it in one pass. */
      const int32_t b_found =
        (int32_t)bin_of(radius * radius, r_core2, inv_bin_w, n_bins);
      int32_t bins_reached = b_first - b_found + 1;
      if (bins_reached < 1)
        bins_reached = 1;

      scratch->span_sum += (uint64_t)bins_reached;
      scratch->span_count++;

      return radius;
    }
    if (status != BIN_EXHAUSTED) {
      *reason = RESCALE_ALLOC;
      return -1.0f;
    }

    if (weighted)
      above_w += window_w;
    else
      above += window_count;
    b = b_lo; /* the loop's decrement then steps past the window */
  }

  *reason = RESCALE_BELOW_RUNG;
  return -1.0f;
}

/*
 * The two specializations of find_exact_radius_impl(). The flag is a literal
 * in each, so every branch on it folds away and the unweighted rescaling
 * compiles to the same code it did before weights existed: integer bins, no
 * weight loads, no double accumulators.
 */
SIF_HOT_LOOP static sif_real find_exact_radius(const sif_chain_mesh_t* mesh,
  sif_real cx, sif_real cy, sif_real cz, sif_real r_search, sif_real threshold,
  sif_real vol_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason) {

  return find_exact_radius_impl(mesh, cx, cy, cz, r_search, threshold,
    vol_factor, vol_factor, scratch, rmin, tpl, window, reason, 0);
}

SIF_HOT_LOOP static sif_real find_exact_radius_weighted(
  const sif_chain_mesh_t* mesh, sif_real cx, sif_real cy, sif_real cz,
  sif_real r_search, sif_real threshold, sif_real vol_factor,
  sif_real mass_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason) {

  return find_exact_radius_impl(mesh, cx, cy, cz, r_search, threshold,
    vol_factor, mass_factor, scratch, rmin, tpl, window, reason, 1);
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

/* --- Run context --- */

typedef struct {
  sif_real* sorted_radii;
  sif_catalog_t* cat;
  sif_bitmask_t* mask;
  sif_cell_linked_list_t* void_cll;
  const sif_chain_mesh_t* mesh; /* borrowed from the caller, never freed */
  sif_fft_workspace_t* fft_ws;
  sif_candidate_buffer_t candidates;

  batch_result_t* batch_results;
  uint64_t* batch_indices;

  radial_scratch_t* scratch; /* one per thread */
  int n_threads;

  /*
   * Largest radius in the catalog so far.
   *
   * sif__overlap_exact() sizes its search box from this, so it has to be the
   * real maximum and not the largest radius on the ladder. Rescaling grows a
   * void out to r_search, about twice the rung it was found at, so the ladder
   * maximum understates the catalog by up to a factor of two -- and a void
   * whose centre falls outside the search box is never compared against, which
   * lets a genuine overlap through. Overestimating only widens the box.
   */
  sif_real max_accepted_r;
} exodus_ctx_t;

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

  if (ctx->scratch) {
    for (int t = 0; t < ctx->n_threads; t++)
      scratch_free(&ctx->scratch[t]);
    sif_free_aligned(ctx->scratch);
    ctx->scratch = NULL;
  }

  sif_free_aligned(ctx->batch_results);
  sif_free_aligned(ctx->batch_indices);
  sif__candidate_buffer_free(&ctx->candidates);
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

static int ctx_init(exodus_ctx_t* ctx, sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_option opt) {

  memset(ctx, 0, sizeof(*ctx));

  ctx->mesh = mesh;

  ctx->n_threads = sif__system_max_threads();
  if (ctx->n_threads < 1)
    ctx->n_threads = 1;

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
    if (scratch_init(&ctx->scratch[t], mesh->weights != NULL) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "failed to allocate the per-thread radial scratch");
      return SIF_ERR_ALLOC;
    }
  }

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

  /* Before the first filter and after the transform, which is the only window
   * in which the assignment window is separable from the smoothing one. The
   * radii are sorted descending, so the last is the smallest. */
  if (sif__finder_deconvolve_cic(
        TAG, ctx->fft_ws, grid, ctx->sorted_radii[n_radii - 1], opt) != SIF_OK)
    return SIF_ERR_INVALID;

  sif_free_aligned(grid->values);
  grid->values = NULL;

  if (sif__fft_workspace_init_backward(ctx->fft_ws, state->fft_mgr) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "failed to initialize the backward FFT");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

static int accept_void(exodus_ctx_t* ctx, const sif_grid_t* grid, sif_real cx,
  sif_real cy, sif_real cz, sif_real r) {

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

  sif__mark_sphere(
    ctx->mask, cx, cy, cz, r, grid->n_cells, grid->p2_mask, grid->cell_length);

  /* Updated here rather than scanned for later, so every overlap test after
   * this point sizes its search box against a catalog that includes this
   * void. */
  if (r > ctx->max_accepted_r)
    ctx->max_accepted_r = r;

  return SIF_OK;
}

/* What check_weights() makes of one weight. */
#define WEIGHT_OK       0
#define WEIGHT_NEGATIVE 1
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
 * counted, rather than left to produce a catalog that looks normal.
 */
static int check_weights(const sif_chain_mesh_t* mesh) {
  if (!mesh->cell_weights) {
    SIF_LOG_ERROR(TAG, "the mesh carries weights but no per-cell weight table");
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
      "the mesh weights must be finite and non-negative: %" PRIu64
      " negative, %" PRIu64 " not finite, of %" PRIu64,
      n_negative, n_nonfinite, mesh->n_particles);
    return SIF_ERR_INVALID;
  }

  if (!(mesh->total_weight > 0.0)) {
    SIF_LOG_ERROR(TAG, "the mesh weights sum to zero; there is no mean density");
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/* --- Driver --- */

sif_catalog_t* sif_finder_exodus(sif_grid_t* grid, const sif_chain_mesh_t* mesh,
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option options) {

  if (!grid || !grid->values || !mesh || !radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "invalid grid, mesh or radii");
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
    SIF_LOG_ERROR(TAG, "the chain mesh holds no particles");
    return NULL;
  }

  /* Grid and mesh coordinates are used interchangeably throughout, so the two
   * have to describe the same box. */
  if (mesh->box_length != grid->box_length) {
    SIF_LOG_ERROR(TAG, "the mesh spans a box of %g but the grid spans %g",
      (double)mesh->box_length, (double)grid->box_length);
    return NULL;
  }

  const int weighted = (mesh->weights != NULL);
  if (weighted && check_weights(mesh) != SIF_OK)
    return NULL;

  exodus_ctx_t ctx;
  if (ctx_init(&ctx, grid, mesh, radii, n_radii, options) != SIF_OK) {
    ctx_release(&ctx, grid);
    return NULL;
  }

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
    weighted ? (4.0f / 3.0f) * SIF_PI *
                 ((sif_real)mesh->total_weight / box_volume)
             : vol_factor;

  if (weighted)
    SIF_LOG_TRACE(TAG,
      "weighted mesh: total weight %.6g over %" PRIu64 " tracers",
      mesh->total_weight, mesh->n_particles);

  sif_timer_t timer;
  int failed = 0;
  uint64_t total_mismatched = 0;

  for (uint32_t i = 0; i < n_radii && !failed; i++) {
    sif_timer_start(&timer);

    const sif_real radius = ctx.sorted_radii[i];
    sif_finder_radius_stats_t stats = {0};

    /* Localization quality for this rung, see MISMATCH_RATIO. */
    double ratio_sum = 0.0;
    sif_real ratio_min = 0.0f;
    sif_real ratio_max = 0.0f;
    uint64_t n_mismatched = 0;

    /* Every rescaling this rung asks for, whatever becomes of it. */
    rescale_report_t rescale = {0};

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

    const sif_real rmin = RMIN_FACTOR * radius;

    /* The reach past the rung, from SIF_FINDER_SEARCH_*. The three-cell floor
     * is kept underneath it: at the small end of a ladder the factor can ask
     * for less than the grid can resolve, and a shell thinner than a few cells
     * bins nothing useful. At factor 2 this is the expression it replaces. */
    const sif_real search_factor = (sif_real)sif__finder_search_factor(options);
    sif_real r_search =
      radius + SIF_REAL_MAX((search_factor - 1.0f) * radius,
                3.0f * grid->cell_length);

    /*
     * Consecutive rungs of the radius ladder have to overlap in the void sizes
     * they can return, or sizes in between are reachable at no rung at all and
     * simply never appear in the catalog.
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
    if (template_build(&tpl, ctx.mesh->n_cells, ctx.mesh->cell_length, rmin,
          r_search) != SIF_OK) {
      SIF_LOG_ERROR(
        TAG, "could not build the mesh template for r = %g", (double)radius);
      failed = 1;
      break;
    }

    const int32_t proxy_pole =
      (int32_t)(radius * (1.0f - overlap_fraction) / grid->cell_length) - 1;

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
    while (k < ctx.candidates.count && !failed) {
      uint64_t batch_count = 0;

      /* Phase 0: skip already-masked candidates sequentially. Doing this
       * outside the parallel region avoids paying thread sync for millions of
       * trivially dead candidates. */
      while (k < ctx.candidates.count && batch_count < batch_cap) {
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
       * past the per-thread shell buffers allocated in ctx_init. */
#pragma omp parallel for schedule(dynamic, 16) num_threads(ctx.n_threads)
      for (uint64_t b = 0; b < batch_count; b++) {
        const int tid = sif__system_thread_num();
        const uint64_t flat =
          ctx.candidates.items[ctx.batch_indices[b]].flat_idx;
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

        const sif_real cx = (sif_real)ix * grid->cell_length;
        const sif_real cy = (sif_real)iy * grid->cell_length;
        const sif_real cz = (sif_real)iz * grid->cell_length;

        res->cx = cx;
        res->cy = cy;
        res->cz = cz;

        if (sif__overlap_exact(ctx.cat, cx, cy, cz, radius, ctx.max_accepted_r,
              grid->box_length, ctx.void_cll, grid->p2_mask,
              overlap_fraction)) {
          res->status = BATCH_REJECTED_MESH;
          continue;
        }

        uint8_t reason = RESCALE_OK;
        const sif_real r_scaled =
          weighted
            ? find_exact_radius_weighted(ctx.mesh, cx, cy, cz, r_search,
                threshold, vol_factor, mass_factor, &ctx.scratch[tid], rmin,
                &tpl, resolve_window, &reason)
            : find_exact_radius(ctx.mesh, cx, cy, cz, r_search, threshold,
                vol_factor, &ctx.scratch[tid], rmin, &tpl, resolve_window,
                &reason);

        if (r_scaled < 0.0f) {
          res->rescale_reason = reason;
          res->status = BATCH_REJECTED_RESCALE;
          continue;
        }

        res->r_scaled = r_scaled;
        res->status = BATCH_ACCEPTED;
      }

      /* Phase 2: commit sequentially, re-checking everything that the parallel
       * phase could not have seen (voids accepted earlier in this same
       * batch). */
      const uint64_t accepted_before = stats.accepted;

      for (uint64_t b = 0; b < batch_count; b++) {
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
        default:
          break;
        }

        /* Past the switch the rescaling converged, whatever the re-checks
         * below go on to decide about the void itself. */
        rescale_report_add(&rescale, res->r_scaled / radius);

        const uint64_t flat =
          ctx.candidates.items[ctx.batch_indices[b]].flat_idx;

        if (sif_bitmask_get(ctx.mask, flat)) {
          stats.rejected_masked++;
          continue;
        }

        /* Both overlap tests run again here, because the catalog has moved on
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

        if (accept_void(&ctx, grid, res->cx, res->cy, res->cz, res->r_scaled) !=
            SIF_OK) {
          SIF_LOG_ERROR(TAG, "failed to store an accepted void, aborting");
          failed = 1;
          break;
        }
        stats.accepted++;

        const sif_real ratio = res->r_scaled / radius;
        ratio_sum += (double)ratio;
        if (stats.accepted == 1 || ratio < ratio_min)
          ratio_min = ratio;
        if (ratio > ratio_max)
          ratio_max = ratio;
        if (ratio > MISMATCH_RATIO)
          n_mismatched++;
      }

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

    template_free(&tpl);

    sif_timer_stop(&timer);

    rescale_report_log(&rescale);

    /* The same ratio again, but over the voids that survived to the catalog:
     * that is the population the localization warning below is about. */
    if (stats.accepted > 0) {
      SIF_LOG_TRACE(TAG,
        "accepted r/R:              min %.2f, mean %.2f, max %.2f  (%" PRIu64
        " beyond %.1fx)",
        (double)ratio_min, ratio_sum / (double)stats.accepted,
        (double)ratio_max, n_mismatched, (double)MISMATCH_RATIO);
    }

    total_mismatched += n_mismatched;

    sif__finder_log_radius(TAG, radius, &stats, ctx.cat->n_voids,
      sif_timer_elapsed_ms(&timer) / 1000.0);
  }

  if (!failed && !(options & SIF_FINDER_CONSUME_GRID)) {
    if (sif__fft_apply_filter(ctx.fft_ws, SIF__FILTER_NONE, 0, 0) == SIF_OK) {
      grid->values = sif__fft_grid_backward(ctx.fft_ws);
      SIF_LOG_INFO(TAG, "recovered original density grid");
    }
  }

  if (total_mismatched > 0 && ctx.cat->n_voids > 0) {
    SIF_LOG_WARNING(TAG,
      "%" PRIu64 " of %" PRIu64
      " voids (%.1f%%) are more than %.1fx the radius of the rung that found "
      "them; their centers are localized at that rung's scale, not their own. "
      "A denser radius ladder would place them better",
      total_mismatched, ctx.cat->n_voids,
      100.0 * (double)total_mismatched / (double)ctx.cat->n_voids,
      (double)MISMATCH_RATIO);
  }

  sif_catalog_trim(ctx.cat);

  sif_catalog_t* result = ctx.cat;
  ctx.cat = NULL; /* ownership passes to the caller */
  ctx_release(&ctx, grid);

  return result;
}
