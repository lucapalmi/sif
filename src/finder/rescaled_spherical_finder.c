/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/finder/rescaled_spherical_finder.h"

#include <math.h>
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

SIF_DEFINE_QUICKSORT(sort_radii_desc, sif_real, a > b)
SIF_DEFINE_QUICKSORT(sort_real_asc, sif_real, a < b)

/* Upper bound on the number of radial bins. Private to this file: it is a
 * tuning constant, not part of the library's interface. */
#define RESCALED_MAX_BINS 2048

#define TAG "rescaled_spherical_finder"

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
 * Rescaling constants. Fixed, so that a catalog depends only on the grid, the
 * radius ladder and the threshold.
 *
 * RMIN_FACTOR floors the rescaled radius at half the rung that found the
 * void. N_MIN is a noise floor on the density estimate: a sphere whose
 * expected tracer count at the mean density falls below it is too sparsely
 * sampled to say anything, so the walk abandons it. At 0 it never triggers.
 */
#define RMIN_FACTOR 0.50f
#define N_MIN       0.0f

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

#define MAX_RADIAL_BINS RESCALED_MAX_BINS
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

typedef struct {
  uint32_t* bins;
  sif_real* refine;
  uint32_t refine_count;
  uint32_t refine_capacity;

  /* How far down the bins the walk had to reach before the answer turned up,
   * summed over the scans that found one. This is what sizes the refinement
   * window; see RESOLVE_* below. Accumulated per thread and folded in during
   * the sequential commit, so no synchronization is needed. */
  uint64_t span_sum;
  uint64_t span_count;
} radial_scratch_t;

static int refine_reserve(radial_scratch_t* s, uint32_t needed) {
  if (needed <= s->refine_capacity)
    return SIF_OK;

  uint32_t new_capacity =
    s->refine_capacity ? s->refine_capacity : REFINE_INITIAL_CAPACITY;
  while (new_capacity < needed) {
    if (new_capacity > UINT32_MAX / 2)
      return SIF_ERR_ALLOC;
    new_capacity *= 2;
  }

  /* The buffer is filled incrementally, one mesh cell at a time, so a grow can
   * land in the middle of a bin's collection and whatever has been gathered so
   * far has to survive it. (It must: dropping it silently loses particles and
   * the walk then returns a radius that is too small. It only stays hidden
   * while a single bin fits in the initial capacity.) */
  sif_real* grown = sif_malloc_aligned((size_t)new_capacity * sizeof(sif_real));
  if (!grown)
    return SIF_ERR_ALLOC;

  if (s->refine_count > 0)
    memcpy(grown, s->refine, (size_t)s->refine_count * sizeof(sif_real));

  sif_free_aligned(s->refine);
  s->refine = grown;
  s->refine_capacity = new_capacity;

  return SIF_OK;
}

static void scratch_free(radial_scratch_t* s) {
  sif_free_aligned(s->bins);
  sif_free_aligned(s->refine);
  memset(s, 0, sizeof(*s));
}

static int scratch_init(radial_scratch_t* s) {
  memset(s, 0, sizeof(*s));

  s->bins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(uint32_t));
  if (!s->bins)
    return SIF_ERR_ALLOC;

  return refine_reserve(s, REFINE_INITIAL_CAPACITY);
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
 *
 * @return 0 if the cell is empty
 */
static inline int query_cell(const mesh_query_t* q, uint32_t i,
  uint64_t* p_start, uint64_t* p_end, sif_real* ex, sif_real* ey,
  sif_real* ez) {

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
 *
 * Widening the range from one bin to several does not change the answer. The
 * walk over the union in descending order is the same sequence of tests as
 * walking each bin in turn, and any bin swept in that would have failed its own
 * skip test contributes nothing: that test is a proof that none of its
 * particles qualify, so walking them costs comparisons and decides nothing.
 *
 * @return BIN_FOUND with *out set, BIN_EXHAUSTED if the range holds no
 * acceptable radius, or SIF_ERR_* if the whole search has to be abandoned
 */
SIF_HOT_LOOP static int resolve_bin(const mesh_query_t* q,
  radial_scratch_t* scratch, uint32_t bin_lo, uint32_t bin_hi, sif_real lo,
  sif_real hi, sif_real r_core2, sif_real r_search2, sif_real inv_bin_w,
  uint32_t n_bins, uint32_t current_N, sif_real K2, sif_real vol_factor2,
  sif_real n_min2, sif_real* out) {

  const sif_real* mx = SIF_ASSUME_ALIGNED(q->mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(q->mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(q->mesh->z);
  const mesh_template_t* tpl = q->tpl;

  scratch->refine_count = 0;

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    /* A bin is a thin shell, so most of the template cannot touch it. */
    if (type == CELL_FULLY_CORE || tpl->max_d2[i] < lo || tpl->min_d2[i] > hi)
      continue;

    uint64_t p_start, p_end;
    sif_real ex, ey, ez;
    if (!query_cell(q, i, &p_start, &p_end, &ex, &ey, &ez))
      continue;

    if (refine_reserve(scratch,
          scratch->refine_count + (uint32_t)(p_end - p_start)) != SIF_OK)
      return SIF_ERR_ALLOC;

    for (uint64_t p = p_start; p < p_end; p++) {
      const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);

      /* Cells that straddle a boundary carry particles outside the annulus. */
      if (type != CELL_FULLY_SHELL && !(d2 > r_core2 && d2 <= r_search2))
        continue;

      const uint32_t bi = bin_of(d2, r_core2, inv_bin_w, n_bins);
      if (bi < bin_lo || bi > bin_hi)
        continue;

      scratch->refine[scratch->refine_count++] = d2;
    }
  }

  if (scratch->refine_count > 1)
    sort_real_asc(scratch->refine, scratch->refine_count);

  for (int32_t k = (int32_t)scratch->refine_count - 1; k >= 0; k--) {
    const sif_real d2_test = scratch->refine[k];
    if (d2_test == 0.0f) {
      current_N--;
      continue;
    }

    const uint32_t n_in = current_N - 1;
    const sif_real d2_cube = d2_test * d2_test * d2_test;

    /* Below the noise floor the estimate stops being meaningful. */
    if (vol_factor2 * d2_cube < n_min2)
      return SIF_ERR_RANGE;

    const sif_real rn_in = (sif_real)n_in;
    if (rn_in * rn_in <= K2 * d2_cube) {
      *out = SIF_REAL_SQRT(d2_test);
      return BIN_FOUND;
    }

    current_N--;
  }

  return BIN_EXHAUSTED;
}

/*
 * Grows the void outwards from `center` until the enclosed number density
 * first rises above (1 + threshold) times the mean.
 *
 * @return the rescaled radius, or -1 if no acceptable radius exists
 */
SIF_HOT_LOOP static sif_real find_exact_radius(const sif_chain_mesh_t* mesh,
  sif_real cx, sif_real cy, sif_real cz, sif_real r_search, sif_real threshold,
  sif_real vol_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window) {

  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);

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

  uint32_t n_core = 0;
  uint32_t n_shell = 0;

  /* Above this many particles inside rmin the void is already denser than the
   * threshold allows, so it can be abandoned before doing any distance work. */
  const sif_real expected_core = vol_factor * rmin * rmin * rmin;
  uint32_t max_core_particles = 0xFFFFFFFFu;
  if (expected_core >= N_MIN)
    max_core_particles = (uint32_t)(expected_core * (1.0f + threshold));

  /* Cheap pre-pass: cell occupancies alone can exceed the core budget. */
  uint32_t guaranteed_core = 0;
  for (uint32_t i = 0; i < tpl->count; i++) {
    if (tpl->type[i] != CELL_FULLY_CORE)
      continue;

    int32_t ix = q.center_ix + tpl->dx[i];
    if (ix < 0)
      ix += N;
    else if (ix >= N)
      ix -= N;
    int32_t iy = q.center_iy + tpl->dy[i];
    if (iy < 0)
      iy += N;
    else if (iy >= N)
      iy -= N;
    int32_t iz = q.center_iz + tpl->dz[i];
    if (iz < 0)
      iz += N;
    else if (iz >= N)
      iz -= N;

    const uint64_t flat = (uint64_t)ix * N * N + (uint64_t)iy * N + iz;
    guaranteed_core +=
      (uint32_t)(mesh->cell_offsets[flat + 1] - mesh->cell_offsets[flat]);

    if (guaranteed_core > max_core_particles)
      return -1.0f;
  }

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

  if (!(span > 0.0f))
    return -1.0f;

  const sif_real inv_bin_w = (sif_real)n_bins / span;
  const sif_real bin_w = span / (sif_real)n_bins;

  uint32_t* bins = scratch->bins;
  memset(bins, 0, (size_t)n_bins * sizeof(uint32_t));

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    uint64_t p_start, p_end;
    sif_real ex, ey, ez;
    if (!query_cell(&q, i, &p_start, &p_end, &ex, &ey, &ez))
      continue;

    const uint64_t p_count = p_end - p_start;

    if (type == CELL_FULLY_CORE) {
      n_core += (uint32_t)p_count;
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

        for (uint32_t t = 0; t < m; t++)
          bins[bin_of(tile[t], r_core2, inv_bin_w, n_bins)]++;
      }

      n_shell += (uint32_t)p_count;
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

    if (n_core > max_core_particles)
      return -1.0f;
  }

  const uint32_t total_N = n_core + n_shell;
  if (total_N == 0)
    return -1.0f;

  /* If the whole search sphere is already denser than the threshold there is
   * no radius in range that satisfies it. */
  const sif_real expected_search = vol_factor * r_search * r_search * r_search;
  if (expected_search >= N_MIN) {
    if (((sif_real)total_N / expected_search) - 1.0f <= threshold)
      return -1.0f;
  }

  /* Walk inwards bin by bin. Comparisons are done on squared distances so no
   * square root is needed until the answer is found:
   *   n_in / (vol_factor * d^3) - 1 <= threshold
   *   <=> n_in^2 <= ((threshold + 1) * vol_factor)^2 * (d^2)^3
   */
  const sif_real vol_factor2 = vol_factor * vol_factor;
  const sif_real K = (threshold + 1.0f) * vol_factor;
  const sif_real K2 = K * K;
  const sif_real n_min2 = N_MIN * N_MIN;

  uint32_t above = 0;   /* particles beyond the bin under examination */
  int32_t b_first = -1; /* outermost bin the walk actually had to open */

  for (int32_t b = (int32_t)n_bins - 1; b >= 0; b--) {
    const uint32_t count = bins[b];
    if (count == 0)
      continue;

    sif_real hi = r_core2 + (sif_real)(b + 1) * bin_w;

    /* The top bin also absorbs anything the clamp pulled back in, so its upper
     * edge has to be generous. Being too generous only costs a refinement that
     * finds nothing; being too tight would skip a real answer. */
    if (b == (int32_t)n_bins - 1)
      hi = r_search2 * (1.0f + 1e-6f);

    /* The most permissive particle this bin could hold is its outermost one
     * carrying the smallest interior count. If even that is too dense, no
     * particle in the bin can qualify and none of them need to be touched. */
    const sif_real n_in_min = (sif_real)(total_N - above - count);
    if (n_in_min * n_in_min > K2 * hi * hi * hi) {
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

    uint32_t window_count = 0;
    for (int32_t bb = b_lo; bb <= b; bb++)
      window_count += bins[bb];

    const sif_real win_lo = r_core2 + (sif_real)b_lo * bin_w;

    sif_real radius = -1.0f;
    const int status = resolve_bin(&q, scratch, (uint32_t)b_lo, (uint32_t)b,
      win_lo, hi, r_core2, r_search2, inv_bin_w, n_bins, total_N - above, K2,
      vol_factor2, n_min2, &radius);

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
    if (status != BIN_EXHAUSTED)
      return -1.0f;

    above += window_count;
    b = b_lo; /* the loop's decrement then steps past the window */
  }

  return -1.0f;
}

/* --- Speculative batch evaluation --- */

typedef struct {
  uint8_t status; /* one of BATCH_* below */
  sif_real r_scaled;
  sif_real cx, cy, cz;
  int32_t proxy_pole;
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
} rescaled_ctx_t;

static void ctx_release(rescaled_ctx_t* ctx, sif_grid_t* grid) {
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

static int ctx_init(rescaled_ctx_t* ctx, sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii) {

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
    if (scratch_init(&ctx->scratch[t]) != SIF_OK) {
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

  sif_free_aligned(grid->values);
  grid->values = NULL;

  if (sif__fft_workspace_init_backward(ctx->fft_ws, state->fft_mgr) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "failed to initialize the backward FFT");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

static int accept_void(rescaled_ctx_t* ctx, const sif_grid_t* grid, sif_real cx,
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

  return SIF_OK;
}

/* --- Driver --- */

sif_catalog_t* sif_finder_rescaled_spherical(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option options) {

  if (!grid || !grid->values || !mesh || !radii || n_radii == 0) {
    SIF_LOG_ERROR(TAG, "invalid grid, mesh or radii");
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

  rescaled_ctx_t ctx;
  if (ctx_init(&ctx, grid, mesh, radii, n_radii) != SIF_OK) {
    ctx_release(&ctx, grid);
    return NULL;
  }

  const sif_real max_radius = ctx.sorted_radii[0];

  /* Loop invariants: the mean tracer density never changes between radii. */
  const sif_real box_volume =
    grid->box_length * grid->box_length * grid->box_length;
  const sif_real mean_density = (sif_real)mesh->n_particles / box_volume;
  const sif_real vol_factor = (4.0f / 3.0f) * SIF_PI * mean_density;

  sif_timer_t timer;
  int failed = 0;
  uint64_t total_mismatched = 0;

  for (uint32_t i = 0; i < n_radii && !failed; i++) {
    sif_timer_start(&timer);

    const sif_real radius = ctx.sorted_radii[i];
    sif_finder_radius_stats_t stats = {0};

    /* Localization quality for this rung, see MISMATCH_RATIO. */
    double ratio_sum = 0.0;
    sif_real ratio_max = 0.0f;
    uint64_t n_mismatched = 0;

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
    sif_real r_search = radius + SIF_REAL_MAX(radius, 3.0f * grid->cell_length);

    /*
     * Consecutive rungs of the radius ladder have to overlap in the void sizes
     * they can return, or sizes in between are reachable at no rung at all and
     * simply never appear in the catalog.
     *
     * What limits a rung from below is not RMIN_FACTOR but detection: an
     * empty void of size p smoothed with a top-hat of radius R registers as
     * delta = -(p/R)^3, so it only clears the threshold once p >= |t|^(1/3) R.
     * The previous, larger rung therefore found nothing below
     * |t|^(1/3) * r_prev, and this rung has to search at least that far out to
     * meet it.
     *
     * With the default r_search = 2 * radius this binds only when the ladder
     * steps by more than a factor of 2 / |t|^(1/3) -- about 2.7 at a threshold
     * of -0.4 -- so it is a guard against a sparse radius list silently losing
     * a size range, not a change to how a sensible run behaves. The estimate
     * assumes an empty spherical void and ignores the grid smoothing, which
     * makes it err wide; erring wide costs time, erring narrow loses voids.
     */
    if (i > 0) {
      sif_real detect = SIF_REAL_POW(SIF_REAL_ABS(threshold), 1.0f / 3.0f);
      if (detect > 1.0f)
        detect = 1.0f;

      const sif_real reach = detect * ctx.sorted_radii[i - 1];
      if (reach > r_search)
        r_search = reach;
    }

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
        res->proxy_pole = proxy_pole;

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

        if (sif__overlap_exact(ctx.cat, cx, cy, cz, radius, max_radius,
              grid->box_length, ctx.void_cll, grid->p2_mask,
              overlap_fraction)) {
          res->status = BATCH_REJECTED_MESH;
          continue;
        }

        const sif_real r_scaled =
          find_exact_radius(ctx.mesh, cx, cy, cz, r_search, threshold,
            vol_factor, &ctx.scratch[tid], rmin, &tpl, resolve_window);

        if (r_scaled < 0.0f) {
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
          stats.rejected_mesh++;
          continue;
        case BATCH_REJECTED_RESCALE:
          stats.rejected_rescale++;
          continue;
        default:
          break;
        }

        const uint64_t flat =
          ctx.candidates.items[ctx.batch_indices[b]].flat_idx;

        if (sif_bitmask_get(ctx.mask, flat)) {
          stats.rejected_masked++;
          continue;
        }

        if (res->proxy_pole > 0 &&
            sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask, res->ix,
              res->iy, res->iz, (uint32_t)res->proxy_pole)) {
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
            sif__overlap_quick(ctx.mask, grid->n_cells, grid->p2_mask, res->ix,
              res->iy, res->iz, (uint32_t)exact_pole)) {
          stats.rejected_exact++;
          continue;
        }

        if (sif__overlap_exact(ctx.cat, res->cx, res->cy, res->cz,
              res->r_scaled, max_radius, grid->box_length, ctx.void_cll,
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
    if (stats.accepted > 0) {
      SIF_LOG_TRACE(TAG,
        "localization:         mean r/R %.2f, max %.2f, %" PRIu64
        " beyond %.1fx",
        ratio_sum / (double)stats.accepted, (double)ratio_max, n_mismatched,
        (double)MISMATCH_RATIO);
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
