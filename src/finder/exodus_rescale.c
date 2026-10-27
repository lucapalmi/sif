/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The radius rescaling of the exodus finder: from a candidate centre, the
 * largest radius at which the tracers enclose no more than (1 + threshold)
 * times the mean, in a box or against a survey's randoms. Why it is computed
 * the way it is, see "Finding the crossing radius" in exodus_finder.c.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/sort.h"

#include "exodus_internal.h"
#include "utils.h"

SIF_DEFINE_QUICKSORT(sort_real_asc, sif_real, a < b)
SIF_DEFINE_QUICKSORT(sort_pairs_asc, refine_pair_t, a.d2 < b.d2)

#define TAG "exodus_finder"

/*
 * --- Per-thread scratch for the radial pass ---
 *
 * The rescaling evaluates the enclosed density on a histogram of squared
 * distances instead of on the distances themselves, so what a thread has to
 * hold stops scaling with how many particles the search sphere contains. At
 * r_search = 200 in a 2250 box that used to be ~10^7 floats -- 38 MiB per
 * thread, allocated in every thread, sorted in full for every candidate.
 *
 * The histogram alone would only localise the answer to a bin, so the exact
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

/* Distances are computed in tiles so the arithmetic stays vectorisable even
 * though the histogram update that follows it cannot be. */
#define DIST_TILE 64u

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

void sif__exodus_scratch_free(radial_scratch_t* s) {
  sif_free_aligned(s->bins);
  sif_free_aligned(s->wbins);
  sif_free_aligned(s->rbins);
  sif_free_aligned(s->refine);
  sif_free_aligned(s->pairs);
  memset(s, 0, sizeof(*s));
}

/* A survey run is always weighted in form -- data and randoms are summed in
 * double whether or not either mesh carries weights -- and adds the random
 * histogram. */
int sif__exodus_scratch_init(radial_scratch_t* s, int weighted, int survey) {
  memset(s, 0, sizeof(*s));

  if (weighted || survey)
    s->wbins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(double));
  else
    s->bins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(uint32_t));

  if (!s->bins && !s->wbins)
    return SIF_ERR_ALLOC;

  if (survey) {
    s->rbins = sif_malloc_aligned((size_t)MAX_RADIAL_BINS * sizeof(double));
    if (!s->rbins)
      return SIF_ERR_ALLOC;
  }

  return refine_reserve(s, REFINE_INITIAL_CAPACITY, weighted || survey);
}

/* --- Mesh traversal template --- */

/* Cell classification relative to the [rmin, r_search] annulus. */
#define CELL_FULLY_CORE   1 /* entirely inside rmin: count, never store */
#define CELL_STRADDLES    2 /* spans the rmin boundary: test each particle */
#define CELL_PARTIAL_EDGE 3 /* spans the r_search boundary: test each */
#define CELL_FULLY_SHELL  4 /* entirely inside the annulus: store all        */

void sif__exodus_template_free(mesh_template_t* tpl) {
  sif_free_aligned(tpl->dx);
  sif_free_aligned(tpl->dy);
  sif_free_aligned(tpl->dz);
  sif_free_aligned(tpl->type);
  sif_free_aligned(tpl->min_d2);
  sif_free_aligned(tpl->max_d2);
  memset(tpl, 0, sizeof(*tpl));
}

int sif__exodus_template_build(mesh_template_t* tpl, uint32_t mesh_n_cells,
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
   * The bound is cell_radius < n_cells rather than <= : the centre cell is
   * (int32_t)(cx / cell_length), which reaches n_cells for a cx that rounds up
   * to the box length, and that extra cell is what the strict inequality
   * covers.
   *
   * It only became reachable when the mesh started coming from the caller. A
   * coarse mesh and a large radius are each reasonable alone.
   */
  if (cell_radius >= (int32_t)mesh_n_cells) {
    SIF_LOG_ERROR(TAG,
      "search sphere of %g (%d mesh cells): wider than the mesh (%u cells); "
      "use a finer mesh or smaller radii",
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
    SIF_LOG_ERROR(
      TAG, "out of memory: mesh template of %" PRIu64 " cells", max_cells);
    sif__exodus_template_free(tpl);
    return SIF_ERR_ALLOC;
  }

  const sif_real r_search2 = r_search * r_search;
  const sif_real r_core2 = rmin * rmin;

  for (int32_t dx = -cell_radius; dx <= cell_radius; dx++) {
    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        /* Nearest and farthest a point in this cell can be from the centre. */
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
 * Resolves template entry `i` to its particle range, and to the periodic image
 * of the query centre that range has to be measured against. Shifting the
 * centre instead of the particles keeps the wrap at one addition per axis.
 * `flat` is the cell's index in the mesh, for anything kept per cell.
 *
 * @return 0 if the cell is empty
 */
static inline int query_cell(const mesh_query_t* q, uint32_t i,
  uint64_t* p_start, uint64_t* p_end, uint64_t* flat_out, sif_real* ex,
  sif_real* ey, sif_real* ez) {

  const int32_t N = q->n_cells;
  const sif_real box_L = q->box_length;

  int32_t ix = q->centre_ix + q->tpl->dx[i];
  sif_real cx_eff = q->cx;
  if (ix < 0) {
    ix += N;
    cx_eff += box_L;
  } else if (ix >= N) {
    ix -= N;
    cx_eff -= box_L;
  }

  int32_t iy = q->centre_iy + q->tpl->dy[i];
  sif_real cy_eff = q->cy;
  if (iy < 0) {
    iy += N;
    cy_eff += box_L;
  } else if (iy >= N) {
    iy -= N;
    cy_eff -= box_L;
  }

  int32_t iz = q->centre_iz + q->tpl->dz[i];
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

static const char* const RESCALE_REASON_LABEL[RESCALE_N_REASONS] = {
  "converged",
  "search sphere empty",
  "underdense out to r_search",
  "smaller than the rung",
  "degenerate search range",
  "refinement alloc failed",
  "outside the footprint",
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
static const sif_real RESCALE_BUCKET_EDGE[RESCALE_N_BUCKETS - 1] = {
  1.05f, 1.10f, 1.25f, 1.50f, 2.00f};

static const char* const RESCALE_BUCKET_LABEL[RESCALE_N_BUCKETS] = {
  "1.00-1.05", "1.05-1.10", "1.10-1.25", "1.25-1.50", "1.50-2.00", ">=2.00"};

/*
 * Record one successful rescaling, as the ratio of the radius it returned to
 * the rung that found the candidate. Every success is counted, including the
 * ones the overlap re-checks go on to reject: what is being described here is
 * the rescaling, not the catalogue.
 */
void sif__exodus_rescale_report_add(rescale_report_t* rep, sif_real ratio) {
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
void sif__exodus_rescale_report_log(const rescale_report_t* rep) {
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
 * sif__exodus_find_radius_weighted(). `vol_factor` stays a count either way,
 * since the one thing it still sizes -- the histogram, from the tracers
 * expected per bin -- is about how many tracers there are, not what they weigh.
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
    .centre_ix = (int32_t)(cx * inv_l),
    .centre_iy = (int32_t)(cy * inv_l),
    .centre_iz = (int32_t)(cz * inv_l),
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

    const sif_real within = weighted ? (sif_real)w_within : (sif_real)n_within;
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
       * the distance itself. Tiling keeps that part vectorised despite the
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
    const sif_real n_in_min = weighted ? (sif_real)(total_W - above_w - count_w)
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
 * The two specialisations of find_exact_radius_impl(). The flag is a literal
 * in each, so every branch on it folds away and the unweighted rescaling
 * compiles to the same code it did before weights existed: integer bins, no
 * weight loads, no double accumulators.
 */
SIF_HOT_LOOP sif_real sif__exodus_find_radius(const sif_chain_mesh_t* mesh,
  sif_real cx, sif_real cy, sif_real cz, sif_real r_search, sif_real threshold,
  sif_real vol_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason) {

  return find_exact_radius_impl(mesh, cx, cy, cz, r_search, threshold,
    vol_factor, vol_factor, scratch, rmin, tpl, window, reason, 0);
}

SIF_HOT_LOOP sif_real sif__exodus_find_radius_weighted(
  const sif_chain_mesh_t* mesh, sif_real cx, sif_real cy, sif_real cz,
  sif_real r_search, sif_real threshold, sif_real vol_factor,
  sif_real mass_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason) {

  return find_exact_radius_impl(mesh, cx, cy, cz, r_search, threshold,
    vol_factor, mass_factor, scratch, rmin, tpl, window, reason, 1);
}

/* --- Crossing witness --- */

/*
 * Whether sif__exodus_find_radius() would certainly return a radius above
 * sqrt(rho2), answered without running it. The driver asks this of a
 * candidate that would collide with the catalogue at any radius past rho2:
 * a yes refuses it before the rescaling is paid for, with the decision the
 * rescaling would have led to (see collision_radius() in exodus_finder.c).
 *
 * The rescaling returns the outermost particle distance d2 in the annulus at
 * which the n_in particles inside satisfy n_in^2 <= K2 * d2^3. Take n, the
 * particles at or inside rho2, and the nearest particle beyond it. When that
 * particle is walked, n_in is n; its d2 is larger than rho2, so if n already
 * passes at rho2 it passes there too, and the walk, coming from outside,
 * stops there or further out. Two counts settle it -- n, and whether any
 * particle lies in (rho2, r_search2] -- and only the cells the sphere through
 * rho2 crosses need a distance: the count is O(surface), like the rescaling's
 * own early refusal, where the histogram it replaces is O(volume).
 *
 * The counts follow the kernel cell by cell, so they are the counts the walk
 * would see: a core cell is inside whatever its distances say, a shell cell
 * is in the annulus whatever they say, and every other particle goes by
 * dist2() against the same periodic image. A whole cell is taken without
 * distances only with WITNESS_SLACK to spare on its bound. The test itself
 * keeps the same slack below the kernel's, so no rounding in either can turn
 * a yes here into a no there.
 *
 * `closed` asks in addition that the search sphere is not underdense out to
 * r_search, which the minima need: there a sphere that is gets measured again
 * with the reach extended, and the radius that comes back has no upper bound.
 * Under the cell seeds the sphere is simply refused, which is a refusal
 * either way.
 */
#define WITNESS_SLACK 1e-5f

int sif__exodus_witness(const sif_chain_mesh_t* mesh, sif_real cx, sif_real cy,
  sif_real cz, sif_real rho2, sif_real r_search, sif_real threshold,
  sif_real vol_factor, const mesh_template_t* tpl, int closed) {

  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);

  const sif_real inv_l = 1.0f / mesh->cell_length;
  const mesh_query_t q = {
    .mesh = mesh,
    .tpl = tpl,
    .cx = cx,
    .cy = cy,
    .cz = cz,
    .centre_ix = (int32_t)(cx * inv_l),
    .centre_iy = (int32_t)(cy * inv_l),
    .centre_iz = (int32_t)(cz * inv_l),
    .n_cells = (int32_t)mesh->n_cells,
    .box_length = mesh->box_length,
  };

  const sif_real r_search2 = r_search * r_search;
  if (!(rho2 < r_search2))
    return 0;

  const sif_real K = (threshold + 1.0f) * vol_factor;
  const sif_real K2 = K * K;
  const sif_real bound = K2 * (rho2 * rho2 * rho2) * (1.0f - WITNESS_SLACK);

  /* Past this many inside, the test is already lost. */
  const double n_cap = sqrt((double)bound) + 2.0;

  const sif_real whole_in = rho2 * (1.0f - WITNESS_SLACK);
  const sif_real whole_out = rho2 * (1.0f + WITNESS_SLACK);

  uint64_t n = 0;      /* at or inside rho2 */
  uint64_t within = 0; /* inside r_search, counted as the kernel counts it */
  int beyond = 0;      /* a particle in (rho2, r_search2] */

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    /* A cell wholly beyond rho2 only has to say it is not empty, and once one
     * has, the rest add nothing unless the sphere has to be closed. */
    if (!closed && beyond && tpl->min_d2[i] > whole_out)
      continue;

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(&q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    const uint64_t p_count = p_end - p_start;

    if (type == CELL_FULLY_CORE || tpl->max_d2[i] <= whole_in) {
      n += p_count;
      within += p_count;
    } else if (type == CELL_FULLY_SHELL && tpl->min_d2[i] > whole_out) {
      within += p_count;
      beyond = 1;
    } else {
      for (uint64_t p = p_start; p < p_end; p++) {
        const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);
        if (d2 <= rho2) {
          n++;
          within++;
        } else if (type == CELL_FULLY_SHELL || d2 <= r_search2) {
          within++;
          beyond = 1;
        }
      }
    }

    if ((double)n > n_cap)
      return 0;
  }

  const sif_real rn = (sif_real)n;
  if (!beyond || !(rn * rn <= bound))
    return 0;

  if (closed) {
    /* The kernel's refusal is (within / expected) - 1 <= threshold. */
    const double expected =
      (double)vol_factor * (double)r_search * (double)r_search *
      (double)r_search;
    if (!((double)within >
          (1.0 + (double)threshold) * expected * (1.0 + WITNESS_SLACK)))
      return 0;
  }

  return 1;
}

/* --- Survey rescaling --- */

/*
 * The survey form of the same condition. There is no box mean to measure a
 * sphere against: the expected content of a sphere is what the randoms put in
 * it, scaled by alpha = W_data / W_random, which carries the footprint, its
 * holes and the radial selection all at once. So the test at a radius d is
 *
 *   D(<d) / (alpha * R(<d)) - 1 <= threshold  <=>  D(<d) <= K * R(<d),
 *   K = (1 + threshold) * alpha,
 *
 * with D and R the data and random weight enclosed. Both are plain sums, and
 * a mesh without weights sums ones, so the kernels below read every tracer
 * through survey_w() and every whole cell through survey_cell_w() and do not
 * care which kind of mesh they were given.
 *
 * The answer is still a data distance. Between two neighbouring data tracers D
 * is constant and R can only grow with the radius, so the test only gets
 * easier outwards and the outermost radius that passes is always where a data
 * tracer sits -- with that tracer, as in the box, not counted as inside.
 *
 * The randoms are binned on the same histogram as the data but never resolved
 * tracer by tracer: they only have to say how much is expected inside a radius,
 * which is a smooth quantity, so within a bin R is interpolated linearly in d^2
 * from the bin's own total. That keeps the exact pass -- the sort -- to the
 * data alone. The error it makes is confined to one bin, far thinner than any
 * scale the randoms resolve, and it cannot break the skip test: the
 * interpolated R never exceeds what the bin's upper edge encloses, which is
 * what that test assumes.
 *
 * A sphere with no randoms in it has no expected content and so no density;
 * it passes nothing, whatever the data say.
 */

/* Weight of tracer p, or 1 on a mesh without weights. */
static inline double survey_w(const sif_real* w, uint64_t p) {
  return w ? (double)w[p] : 1.0;
}

/* Weight of a whole cell, from the mesh's table or its tracer count. */
static inline double survey_cell_w(
  const sif_real* cw, uint64_t flat, uint64_t p_start, uint64_t p_end) {
  return cw ? (double)cw[flat] : (double)(p_end - p_start);
}

/*
 * Weight inside r_search, and the tracer count with it. The same three cases
 * as the box's early refusal: whole cells from the table, only the cells the
 * outer edge cuts read tracer by tracer.
 */
static void survey_sphere_total(const mesh_query_t* q, sif_real r_search2,
  double* w_within, uint64_t* n_within) {

  const sif_chain_mesh_t* mesh = q->mesh;
  const mesh_template_t* tpl = q->tpl;
  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mw = mesh->weights;
  const sif_real* cw = mesh->cell_weights;

  double w = 0.0;
  uint64_t n = 0;

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    if (type == CELL_FULLY_CORE || type == CELL_FULLY_SHELL) {
      w += survey_cell_w(cw, flat, p_start, p_end);
      n += p_end - p_start;
      continue;
    }

    for (uint64_t p = p_start; p < p_end; p++) {
      if (dist2(mx[p], my[p], mz[p], ex, ey, ez) <= r_search2) {
        w += survey_w(mw, p);
        n++;
      }
    }
  }

  *w_within = w;
  *n_within = n;
}

/*
 * One mesh's histogram of the annulus, in weight, and its weight in the core
 * and in the annulus. The data and the randoms each go through this once, on
 * the same bins.
 */
static void survey_histogram(const mesh_query_t* q, double* wbins,
  uint32_t n_bins, sif_real r_core2, sif_real r_search2, sif_real inv_bin_w,
  double* w_core_out, double* w_shell_out) {

  const sif_chain_mesh_t* mesh = q->mesh;
  const mesh_template_t* tpl = q->tpl;
  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mw = mesh->weights;
  const sif_real* cw = mesh->cell_weights;

  double w_core = 0.0;
  double w_shell = 0.0;

  memset(wbins, 0, (size_t)n_bins * sizeof(double));

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(q, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    if (type == CELL_FULLY_CORE) {
      w_core += survey_cell_w(cw, flat, p_start, p_end);
    } else if (type == CELL_FULLY_SHELL) {
      sif_real tile[DIST_TILE];

      for (uint64_t base = p_start; base < p_end; base += DIST_TILE) {
        const uint32_t m = (uint32_t)((p_end - base < (uint64_t)DIST_TILE)
                                        ? (p_end - base)
                                        : (uint64_t)DIST_TILE);
#pragma omp simd
        for (uint32_t t = 0; t < m; t++) {
          tile[t] = dist2(mx[base + t], my[base + t], mz[base + t], ex, ey, ez);
        }

        for (uint32_t t = 0; t < m; t++) {
          const double w = survey_w(mw, base + t);
          wbins[bin_of(tile[t], r_core2, inv_bin_w, n_bins)] += w;
          w_shell += w;
        }
      }
    } else {
      for (uint64_t p = p_start; p < p_end; p++) {
        const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);
        const double w = survey_w(mw, p);

        if (d2 <= r_core2) {
          w_core += w;
        } else if (d2 <= r_search2) {
          wbins[bin_of(d2, r_core2, inv_bin_w, n_bins)] += w;
          w_shell += w;
        }
      }
    }
  }

  *w_core_out = w_core;
  *w_shell_out = w_shell;
}

/*
 * The exact pass over bins [bin_lo, bin_hi]: every data tracer that lands in
 * them, sorted and walked outside-in, against the random weight interpolated
 * at its own distance. `d_top` and `r_top` are the data and random weight at
 * or below the top of bin_hi.
 *
 * @return BIN_FOUND with *out set, BIN_EXHAUSTED, or SIF_ERR_ALLOC.
 */
static int survey_resolve(const mesh_query_t* dq, radial_scratch_t* scratch,
  uint32_t bin_lo, uint32_t bin_hi, sif_real lo, sif_real hi, sif_real r_core2,
  sif_real r_search2, sif_real inv_bin_w, sif_real bin_w, uint32_t n_bins,
  double d_top, double r_top, const double* rbins, sif_real K, sif_real* out) {

  const sif_chain_mesh_t* mesh = dq->mesh;
  const mesh_template_t* tpl = dq->tpl;
  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mw = mesh->weights;

  scratch->refine_count = 0;

  for (uint32_t i = 0; i < tpl->count; i++) {
    const uint8_t type = tpl->type[i];

    if (type == CELL_FULLY_CORE || tpl->max_d2[i] < lo || tpl->min_d2[i] > hi)
      continue;

    uint64_t p_start, p_end, flat;
    sif_real ex, ey, ez;
    if (!query_cell(dq, i, &p_start, &p_end, &flat, &ex, &ey, &ez))
      continue;

    if (refine_reserve(scratch,
          scratch->refine_count + (uint32_t)(p_end - p_start), 1) != SIF_OK)
      return SIF_ERR_ALLOC;

    for (uint64_t p = p_start; p < p_end; p++) {
      const sif_real d2 = dist2(mx[p], my[p], mz[p], ex, ey, ez);

      if (type != CELL_FULLY_SHELL && !(d2 > r_core2 && d2 <= r_search2))
        continue;

      const uint32_t bi = bin_of(d2, r_core2, inv_bin_w, n_bins);
      if (bi < bin_lo || bi > bin_hi)
        continue;

      scratch->pairs[scratch->refine_count++] =
        (refine_pair_t){d2, (sif_real)survey_w(mw, p)};
    }
  }

  if (scratch->refine_count > 1)
    sort_pairs_asc(scratch->pairs, scratch->refine_count);

  /* The random weight below the lower edge of the bin under the walk. The walk
   * only ever moves inwards, so this only ever steps down a bin at a time. */
  uint32_t cur_bin = bin_hi;
  double r_below = r_top - rbins[bin_hi];
  double d_current = d_top;

  for (int32_t k = (int32_t)scratch->refine_count - 1; k >= 0; k--) {
    const sif_real d2_test = scratch->pairs[k].d2;
    const double w_test = (double)scratch->pairs[k].w;

    if (d2_test == 0.0f) {
      d_current -= w_test;
      continue;
    }

    const uint32_t bi = bin_of(d2_test, r_core2, inv_bin_w, n_bins);
    while (cur_bin > bi) {
      cur_bin--;
      r_below -= rbins[cur_bin];
    }

    sif_real frac =
      (d2_test - (r_core2 + (sif_real)cur_bin * bin_w)) * inv_bin_w;
    frac = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);

    const sif_real r_in = (sif_real)(r_below + rbins[cur_bin] * (double)frac);
    const sif_real d_in = (sif_real)(d_current - w_test);

    if (r_in > 0.0f && d_in <= K * r_in) {
      *out = SIF_REAL_SQRT(d2_test);
      return BIN_FOUND;
    }

    d_current -= w_test;
  }

  return BIN_EXHAUSTED;
}

/*
 * The survey rescaling: the largest data distance at which D(<d) <= K R(<d).
 * The same three stages as find_exact_radius_impl(), run on two meshes.
 *
 * `dq` and `rq` share the centre and the annulus but not the template, since
 * the two meshes are free to use different cells.
 */
SIF_HOT_LOOP sif_real sif__exodus_find_radius_survey(const mesh_query_t* dq,
  const mesh_query_t* rq, sif_real r_search, sif_real K,
  radial_scratch_t* scratch, sif_real rmin, uint32_t window, uint8_t* reason) {

  *reason = RESCALE_OK;

  const sif_real r_search2 = r_search * r_search;
  const sif_real r_core2 = rmin * rmin;
  const sif_real span = r_search2 - r_core2;

  if (!(span > 0.0f)) {
    *reason = RESCALE_DEGENERATE;
    return -1.0f;
  }

  /* The cheap refusal, as in the box: the whole search sphere first, O(surface)
   * on both meshes, before anything is binned. */
  double d_within, r_within;
  uint64_t n_d_within, n_r_within;
  survey_sphere_total(dq, r_search2, &d_within, &n_d_within);
  survey_sphere_total(rq, r_search2, &r_within, &n_r_within);

  if (!(r_within > 0.0)) {
    *reason = RESCALE_NO_RANDOMS;
    return -1.0f;
  }

  if ((sif_real)d_within <= K * (sif_real)r_within) {
    *reason = RESCALE_BEYOND_SEARCH;
    return -1.0f;
  }

  /* Sized by the data, which is what the exact pass sorts; the randoms only
   * have to fill whatever bins that gives them. */
  uint32_t n_bins = MAX_RADIAL_BINS;
  {
    const sif_real core_fraction =
      (rmin * rmin * rmin) / (r_search * r_search * r_search);
    const sif_real want =
      (sif_real)n_d_within * (1.0f - core_fraction) / PARTICLES_PER_BIN;

    if (!(want >= (sif_real)MIN_RADIAL_BINS))
      n_bins = MIN_RADIAL_BINS;
    else if (want < (sif_real)MAX_RADIAL_BINS)
      n_bins = (uint32_t)want;
  }

  const sif_real inv_bin_w = (sif_real)n_bins / span;
  const sif_real bin_w = span / (sif_real)n_bins;

  double* dbins = scratch->wbins;
  double* rbins = scratch->rbins;

  double d_core, d_shell, r_core, r_shell;
  survey_histogram(
    dq, dbins, n_bins, r_core2, r_search2, inv_bin_w, &d_core, &d_shell);
  survey_histogram(
    rq, rbins, n_bins, r_core2, r_search2, inv_bin_w, &r_core, &r_shell);

  const double d_total = d_core + d_shell;
  const double r_total = r_core + r_shell;

  /* Backstops, as in the box: the totals were assembled in a different order
   * from the ones above and can differ from them in the last bit. */
  if (!(r_total > 0.0)) {
    *reason = RESCALE_NO_RANDOMS;
    return -1.0f;
  }
  if ((sif_real)d_total <= K * (sif_real)r_total) {
    *reason = RESCALE_BEYOND_SEARCH;
    return -1.0f;
  }

  double d_above = 0.0; /* data weight beyond the bin under examination */
  double r_above = 0.0; /* random weight beyond it */
  int32_t b_first = -1;

  for (int32_t b = (int32_t)n_bins - 1; b >= 0; b--) {
    const double bd = dbins[b];
    const double br = rbins[b];

    /* No data weight, no candidate radius -- but its randoms are still
     * beyond everything further in. */
    if (!(bd > 0.0)) {
      r_above += br;
      continue;
    }

    sif_real hi = r_core2 + (sif_real)(b + 1) * bin_w;
    if (b == (int32_t)n_bins - 1)
      hi = r_search2 * (1.0f + 1e-6f);

    /* The skip test: the least data a tracer in this bin could have inside it
     * against the most randoms it could have -- everything up to the bin's
     * upper edge. If even that fails, the whole bin does. */
    const sif_real d_in_min = (sif_real)(d_total - d_above - bd);
    const sif_real r_in_max = (sif_real)(r_total - r_above);

    if (!(r_in_max > 0.0f) || d_in_min > K * r_in_max) {
      d_above += bd;
      r_above += br;
      continue;
    }

    if (b_first < 0)
      b_first = b;

    int32_t b_lo = b - (int32_t)(window - 1);
    if (b_lo < 0)
      b_lo = 0;

    double window_d = 0.0, window_r = 0.0;
    for (int32_t bb = b_lo; bb <= b; bb++) {
      window_d += dbins[bb];
      window_r += rbins[bb];
    }

    const sif_real win_lo = r_core2 + (sif_real)b_lo * bin_w;

    sif_real radius = -1.0f;
    const int status = survey_resolve(dq, scratch, (uint32_t)b_lo, (uint32_t)b,
      win_lo, hi, r_core2, r_search2, inv_bin_w, bin_w, n_bins,
      d_total - d_above, r_total - r_above, rbins, K, &radius);

    if (status == BIN_FOUND) {
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

    d_above += window_d;
    r_above += window_r;
    b = b_lo;
  }

  *reason = RESCALE_BELOW_RUNG;
  return -1.0f;
}
