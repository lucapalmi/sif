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
#include <string.h>

/* Logical partitioning of the candidate scan. Decoupled from the thread count
 * so the two passes agree on chunk boundaries regardless of the team size. */
#define FINDER_SCAN_CHUNKS 256

SIF_DEFINE_QUICKSORT(sort_candidates, sif_candidate_t, a.delta < b.delta)

/* --- 1. Base Utilities --- */

SIF_PURE_FUNCTION inline uint64_t sif__flat_index(
  uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz) {
  return (uint64_t)ix * n * n + (uint64_t)iy * n + (uint64_t)iz;
}

int sif__finder_check_args(const char* tag, const sif_real* radii,
  uint32_t n_radii, sif_real threshold, sif_real overlap_fraction) {

  if (!radii || n_radii == 0) {
    SIF_LOG_ERROR(tag, "radii: none given");
    return SIF_ERR_INVALID;
  }
  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f) || !isfinite(radii[i])) {
      SIF_LOG_ERROR(
        tag, "radii[%u] = %g: not a positive radius", i, (double)radii[i]);
      return SIF_ERR_INVALID;
    }
  }
  if (!(threshold > -1.0f && threshold < 0.0f)) {
    SIF_LOG_ERROR(tag, "threshold %g: not a density contrast in (-1, 0)",
      (double)threshold);
    return SIF_ERR_INVALID;
  }
  if (!(overlap_fraction >= 0.0f && overlap_fraction <= 1.0f)) {
    SIF_LOG_ERROR(
      tag, "overlap_fraction %g: not in [0, 1]", (double)overlap_fraction);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* --- 2. Field preparation --- */

/*
 * Below this many cells per radius the top-hat no longer suppresses the modes
 * the deconvolution amplifies. At two cells the net gain at the Nyquist corner
 * is about 1.1; at one it is several, and the finder would be hunting minima
 * in noise the deconvolution put there itself.
 */
#define FINDER_MIN_CELLS_PER_RADIUS 2.0f

int sif__finder_deconvolve_cic(const char* tag, sif_fft_workspace_t* ws,
  const sif_grid_t* grid, sif_real min_radius, sif_option opt) {

  if (opt & SIF_FINDER_KEEP_CIC_WINDOW) {
    SIF_LOG_TRACE(tag, "keeping the CIC window, as requested");
    return SIF_OK;
  }

  if (min_radius < FINDER_MIN_CELLS_PER_RADIUS * grid->cell_length) {
    SIF_LOG_WARNING(tag,
      "the smallest radius is %g, under %.0f cells of %g; deconvolving the "
      "CIC window amplifies the smallest scales more than the top-hat there "
      "suppresses them. Refine the grid, drop that radius, or pass "
      "SIF_FINDER_KEEP_CIC_WINDOW",
      (double)min_radius, (double)FINDER_MIN_CELLS_PER_RADIUS,
      (double)grid->cell_length);
  }

  const int status = sif__fft_set_cic_correction(ws, 1);
  if (status != SIF_OK)
    SIF_LOG_ERROR(tag, "failed to set up the CIC window correction");

  return status;
}

/* --- 3. Candidate scanning --- */

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
        candidates[w].refined = 0;
        w++;
      }
    }
  }

  sort_candidates(candidates, n_candidates);
  buf->count = n_candidates;

  return SIF_OK;
}

/* --- 3b. Minimum scanning --- */

/*
 * The Gaussian the minima are found on: sigma of one cell, truncated two cells
 * either side, exp(-o^2 / 2) normalised over o = -2..2. Written out rather
 * than computed, so a catalogue does not depend on how a platform's expf()
 * rounds.
 *
 * Why it is there at all: a top-hat's transform falls only as (kR)^-2, and the
 * CIC deconvolution lifts what is left near the Nyquist corner, so the
 * smoothed field carries ripple a cell across. The grid samples that ripple
 * but cannot resolve it, and a sub-cell position fitted to it follows wherever
 * the nodes happened to fall: on lognormal mocks, shifting every tracer by a
 * fraction of a cell moved the fitted centres by 0.3-0.5 cells more than the
 * shift. One cell of Gaussian removes the ripple and brings that to 0.07-0.15,
 * while leaving the scale the rung asked for alone -- it widens a window of
 * radius R by a fraction of order (cell / R)^2. Half a cell did not help.
 *
 * Minima are found on the smoothed field too, not only fitted on it. Ripple
 * makes minima of its own, which the Gaussian then erases, and a fit at one of
 * those has no minimum to find.
 */
#define GAUSS_W0 0.40261995f
#define GAUSS_W1 0.24420134f
#define GAUSS_W2 0.05448868f

/* Columns smoothed together on the two strided axes, sized to a cache line
 * of floats so a gathered row is one line rather than sixteen. */
#define GAUSS_TILE 16

/* How far the Newton step may move a centre off its node, in cells per axis.
 * A discrete minimum is not always within half a cell of the continuous one:
 * in a basin elongated off the axes it can be most of a cell away, and at
 * half a cell a quarter of minima were being sent back to their node. Past
 * one cell the quadratic is being trusted further than the stencil it came
 * from. */
#define MINIMUM_MAX_STEP 1.0

static SIF_ALWAYS_INLINE sif_real gauss5(
  sif_real m2, sif_real m1, sif_real c, sif_real p1, sif_real p2) {

  return GAUSS_W0 * c + GAUSS_W1 * (m1 + p1) + GAUSS_W2 * (m2 + p2);
}

/* The same, renormalised over the observed taps. An unobserved cell stays
 * unobserved; an observed one always has at least its own tap. */
static SIF_ALWAYS_INLINE sif_real gauss5_observed(
  sif_real m2, sif_real m1, sif_real c, sif_real p1, sif_real p2) {

  if (!(c < SIF__FINDER_UNOBSERVED))
    return c;

  sif_real acc = GAUSS_W0 * c;
  sif_real w = GAUSS_W0;
  if (m1 < SIF__FINDER_UNOBSERVED) {
    acc += GAUSS_W1 * m1;
    w += GAUSS_W1;
  }
  if (p1 < SIF__FINDER_UNOBSERVED) {
    acc += GAUSS_W1 * p1;
    w += GAUSS_W1;
  }
  if (m2 < SIF__FINDER_UNOBSERVED) {
    acc += GAUSS_W2 * m2;
    w += GAUSS_W2;
  }
  if (p2 < SIF__FINDER_UNOBSERVED) {
    acc += GAUSS_W2 * p2;
    w += GAUSS_W2;
  }
  return acc / w;
}

/* Smooths one contiguous line of n values in place, through a copy padded
 * with its periodic images. */
static void gauss_line(sif_real* line, sif_real* pad, uint32_t n, int survey) {
  for (uint32_t j = 0; j < n + 4; j++)
    pad[j] = line[(j + 2 * n - 2) % n];

  if (survey) {
    for (uint32_t k = 0; k < n; k++)
      line[k] =
        gauss5_observed(pad[k], pad[k + 1], pad[k + 2], pad[k + 3], pad[k + 4]);
  } else {
    for (uint32_t k = 0; k < n; k++)
      line[k] = gauss5(pad[k], pad[k + 1], pad[k + 2], pad[k + 3], pad[k + 4]);
  }
}

/*
 * Smooths `width` adjacent columns of n rows in place, along the rows. Row r
 * starts at base + r * stride. The rows are gathered whole into the tile
 * first, so the writes back never feed a later read.
 */
static void gauss_tile(sif_real* base, uint64_t stride, uint32_t n,
  uint32_t width, sif_real* tile, int survey) {

  for (uint32_t r = 0; r < n; r++)
    memcpy(tile + (uint64_t)r * GAUSS_TILE, base + r * stride,
      width * sizeof(sif_real));

  for (uint32_t r = 0; r < n; r++) {
    const sif_real* m2 = tile + (uint64_t)((r + 2 * n - 2) % n) * GAUSS_TILE;
    const sif_real* m1 = tile + (uint64_t)((r + n - 1) % n) * GAUSS_TILE;
    const sif_real* c = tile + (uint64_t)r * GAUSS_TILE;
    const sif_real* p1 = tile + (uint64_t)((r + 1) % n) * GAUSS_TILE;
    const sif_real* p2 = tile + (uint64_t)((r + 2) % n) * GAUSS_TILE;
    sif_real* out = base + r * stride;

    if (survey) {
      for (uint32_t b = 0; b < width; b++)
        out[b] = gauss5_observed(m2[b], m1[b], c[b], p1[b], p2[b]);
    } else {
      for (uint32_t b = 0; b < width; b++)
        out[b] = gauss5(m2[b], m1[b], c[b], p1[b], p2[b]);
    }
  }
}

/*
 * The Gaussian, in place and one axis at a time. A pass only ever mixes
 * values along its own axis, so every line is independent and needs nothing
 * larger than itself: no second grid, which at 2250^3 would be 45 GB.
 */
static int gauss_smooth(sif_real* f, uint32_t n, int survey) {
  const uint64_t n2 = (uint64_t)n * n;
  const uint32_t n_blocks = (n + GAUSS_TILE - 1) / GAUSS_TILE;
  int failed = 0;

#pragma omp parallel
  {
    sif_real* pad = malloc(((uint64_t)n + 4) * sizeof(sif_real));
    sif_real* tile = malloc((uint64_t)n * GAUSS_TILE * sizeof(sif_real));
    int ok = (pad && tile);
    if (!ok) {
#pragma omp atomic write
      failed = 1;
    }

    /* Every thread reaches every barrier below, with or without its buffers:
     * a thread that returned early would leave the others waiting. */
#pragma omp barrier

    if (ok && !failed) {
#pragma omp for schedule(static)
      for (uint64_t l = 0; l < n2; l++)
        gauss_line(f + l * n, pad, n, survey);

#pragma omp for schedule(static)
      for (uint64_t t = 0; t < (uint64_t)n * n_blocks; t++) {
        const uint32_t ix = (uint32_t)(t / n_blocks);
        const uint32_t z0 = (uint32_t)(t % n_blocks) * GAUSS_TILE;
        const uint32_t width = (n - z0 < GAUSS_TILE ? n - z0 : GAUSS_TILE);
        gauss_tile(f + ix * n2 + z0, n, n, width, tile, survey);
      }

#pragma omp for schedule(static)
      for (uint64_t t = 0; t < (uint64_t)n * n_blocks; t++) {
        const uint32_t iy = (uint32_t)(t / n_blocks);
        const uint32_t z0 = (uint32_t)(t % n_blocks) * GAUSS_TILE;
        const uint32_t width = (n - z0 < GAUSS_TILE ? n - z0 : GAUSS_TILE);
        gauss_tile(f + (uint64_t)iy * n + z0, n2, n, width, tile, survey);
      }
    }

    free(pad);
    free(tile);
  }

  if (failed) {
    SIF_LOG_ERROR(
      "finder", "failed to allocate smoothing buffers for %u cells", n);
    return SIF_ERR_ALLOC;
  }
  return SIF_OK;
}

/* Gathers the 27 values around node (ix, iy, iz), s[(a+1)*9 + (b+1)*3 + c+1]
 * holding the one at offset (a, b, c), and the flat indices alongside. */
static SIF_ALWAYS_INLINE void load_stencil(const sif_real* f, uint32_t n,
  uint32_t ix, uint32_t iy, uint32_t iz, sif_real s[27], uint64_t idx[27]) {

  const uint32_t xs[3] = {(ix + n - 1) % n, ix, (ix + 1) % n};
  const uint32_t ys[3] = {(iy + n - 1) % n, iy, (iy + 1) % n};
  const uint32_t zs[3] = {(iz + n - 1) % n, iz, (iz + 1) % n};

  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      for (int c = 0; c < 3; c++) {
        const uint64_t i = ((uint64_t)xs[a] * n + ys[b]) * n + zs[c];
        idx[a * 9 + b * 3 + c] = i;
        s[a * 9 + b * 3 + c] = f[i];
      }
}

/* Strictly below every neighbour, ties broken by flat index: a neighbour at a
 * lower index wins an equal value, so of any run of equal values exactly the
 * lowest-indexed is a minimum. */
static SIF_ALWAYS_INLINE int is_minimum(
  const sif_real s[27], const uint64_t idx[27]) {

  const sif_real v = s[13];
  const uint64_t c = idx[13];
  for (int k = 0; k < 27; k++) {
    if (k == 13)
      continue;
    if (idx[k] < c ? !(v < s[k]) : !(v <= s[k]))
      return 0;
  }
  return 1;
}

/*
 * One Newton step from the node: gradient and Hessian by central differences
 * on the stencil, cross terms included, then d = -H^-1 g by cofactors. Exact
 * for a quadratic, which a basin smoothed well past the cell scale is to
 * within its cubic term. In double, because at a large rung the second
 * differences are small against the values they are taken from.
 *
 * @return 1 with the step in `d` and the depth at its end in `depth`, or 0 if
 * the stencil is not convex or the step goes further than MINIMUM_MAX_STEP.
 */
static SIF_ALWAYS_INLINE int newton_step(
  const sif_real s[27], double d[3], double* depth) {

#define S(a, b, c) ((double)s[((a) + 1) * 9 + ((b) + 1) * 3 + ((c) + 1)])
  const double c0 = S(0, 0, 0);
  const double g0 = 0.5 * (S(1, 0, 0) - S(-1, 0, 0));
  const double g1 = 0.5 * (S(0, 1, 0) - S(0, -1, 0));
  const double g2 = 0.5 * (S(0, 0, 1) - S(0, 0, -1));

  const double h00 = S(1, 0, 0) - 2.0 * c0 + S(-1, 0, 0);
  const double h11 = S(0, 1, 0) - 2.0 * c0 + S(0, -1, 0);
  const double h22 = S(0, 0, 1) - 2.0 * c0 + S(0, 0, -1);
  const double h01 =
    0.25 * (S(1, 1, 0) - S(1, -1, 0) - S(-1, 1, 0) + S(-1, -1, 0));
  const double h02 =
    0.25 * (S(1, 0, 1) - S(1, 0, -1) - S(-1, 0, 1) + S(-1, 0, -1));
  const double h12 =
    0.25 * (S(0, 1, 1) - S(0, 1, -1) - S(0, -1, 1) + S(0, -1, -1));
#undef S

  /* Cofactors of the symmetric H; the determinant expands along row 0. */
  const double k00 = h11 * h22 - h12 * h12;
  const double k01 = h02 * h12 - h01 * h22;
  const double k02 = h01 * h12 - h02 * h11;
  const double k11 = h00 * h22 - h02 * h02;
  const double k12 = h01 * h02 - h00 * h12;
  const double k22 = h00 * h11 - h01 * h01;
  const double det = h00 * k00 + h01 * k01 + h02 * k02;

  /* Positive definite, by Sylvester's leading minors. */
  if (!(h00 > 0.0 && k22 > 0.0 && det > 0.0))
    return 0;

  d[0] = -(k00 * g0 + k01 * g1 + k02 * g2) / det;
  d[1] = -(k01 * g0 + k11 * g1 + k12 * g2) / det;
  d[2] = -(k02 * g0 + k12 * g1 + k22 * g2) / det;

  if (!(fabs(d[0]) <= MINIMUM_MAX_STEP && fabs(d[1]) <= MINIMUM_MAX_STEP &&
        fabs(d[2]) <= MINIMUM_MAX_STEP))
    return 0;

  *depth = c0 + 0.5 * (g0 * d[0] + g1 * d[1] + g2 * d[2]);
  return 1;
}

/* The whole-cell part of a step, rounding half away from nothing: a step of
 * exactly half a cell stays on its node. */
static SIF_ALWAYS_INLINE int32_t step_cells(double d) {
  return d > 0.5 ? 1 : (d < -0.5 ? -1 : 0);
}

/*
 * The candidate for the cell at node (ix, iy, iz), from the smoothed stencil
 * around it, which is a minimum when `minimum` is set. A cell that is not, or
 * is not to be refined, stays on its node, at its own depth. A minimum to be
 * refined moves to the node nearest its refined centre, at the depth there,
 * with its offset in `off` -- unless the step fails, or lands on a cell that
 * is not itself a candidate: the centre has to sit on one, or a survey void
 * could be placed on unobserved space.
 *
 * @return 1 for a minimum that moved, -1 for one that stayed, 0 otherwise.
 */
static SIF_ALWAYS_INLINE int place_candidate(const sif_real s[27],
  const uint64_t idx[27], int minimum, int refine, const sif_bitmask_t* below,
  uint32_t n, uint32_t ix, uint32_t iy, uint32_t iz, sif_candidate_t* cand,
  sif_offset_t* off) {

  cand->delta = s[13];
  cand->refined = 0;
  cand->flat_idx = idx[13];

  if (!minimum || !refine)
    return 0;

  double d[3], depth;
  if (!newton_step(s, d, &depth))
    return -1;

  const int32_t sx = step_cells(d[0]);
  const int32_t sy = step_cells(d[1]);
  const int32_t sz = step_cells(d[2]);
  const uint32_t jx = (uint32_t)(((int64_t)ix + sx + n) % n);
  const uint32_t jy = (uint32_t)(((int64_t)iy + sy + n) % n);
  const uint32_t jz = (uint32_t)(((int64_t)iz + sz + n) % n);
  const uint64_t flat = ((uint64_t)jx * n + jy) * n + jz;

  if (!sif_bitmask_get(below, flat))
    return -1;

  cand->delta = (sif_real)depth;
  cand->flat_idx = flat;
  off->dx = (sif_real)(d[0] - sx);
  off->dy = (sif_real)(d[1] - sy);
  off->dz = (sif_real)(d[2] - sz);
  return 1;
}

/* First word of chunk c, clamped: with fewer words than chunks the tail
 * chunks are empty. */
static inline uint64_t chunk_start(int c, uint64_t chunk, uint64_t n_words) {
  const uint64_t w = (uint64_t)c * chunk;
  return w < n_words ? w : n_words;
}

/* Grows a buffer of `size`-byte items to hold `needed`, discarding what it
 * held: every caller rewrites it whole. */
static int buffer_reserve(
  void** items, uint64_t* capacity, uint64_t needed, size_t size) {
  if (needed <= *capacity)
    return SIF_OK;

  sif_free_aligned(*items);
  *items = NULL;
  *capacity = 0;

  const uint64_t new_capacity = (uint64_t)((double)needed * 1.2) + 1;
  *items = sif_malloc_aligned(new_capacity * size);
  if (!*items) {
    SIF_LOG_ERROR("finder",
      "failed to allocate %" PRIu64 " entries of %zu bytes", new_capacity,
      size);
    return SIF_ERR_ALLOC;
  }
  *capacity = new_capacity;
  return SIF_OK;
}

void sif__offset_buffer_free(sif_offset_buffer_t* buf) {
  if (!buf)
    return;
  sif_free_aligned(buf->items);
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

int sif__finder_scan_refined(sif_grid_t* grid, const sif_bitmask_t* mask,
  sif_bitmask_t* below, sif_real threshold, int survey, int refine,
  int minima_only, sif_candidate_buffer_t* candidates,
  sif_offset_buffer_t* offsets, sif_refine_stats_t* stats) {

  const uint32_t n = grid->n_cells;
  const uint64_t total_cells = grid->total_cells;
  sif_real* f = grid->values;

  candidates->count = 0;
  offsets->count = 0;
  stats->n_minima = 0;
  stats->n_fallback = 0;

  /* The candidates, by the plain scan's predicate on the top-hat, before the
   * Gaussian touches the field: a word at a time, so no write needs to be
   * atomic. Bits past the last cell stay clear. */
  const uint64_t n_words = below->n_words;

#pragma omp parallel for schedule(static)
  for (uint64_t w = 0; w < n_words; w++) {
    const uint64_t first = w * 64;
    const uint64_t last = first + 64 < total_cells ? first + 64 : total_cells;
    uint64_t word = 0;
    for (uint64_t i = first; i < last; i++)
      word |= (uint64_t)(f[i] <= threshold) << (i - first);
    below->words[w] = word & ~mask->words[w];
  }

  /* Candidates and minima per chunk of words, the minima on the field the
   * centres are placed on -- which is why the Gaussian goes first. */
  uint64_t cand_offsets[FINDER_SCAN_CHUNKS] = {0};
  uint64_t min_offsets[FINDER_SCAN_CHUNKS] = {0};
  const uint64_t chunk =
    (n_words + FINDER_SCAN_CHUNKS - 1) / FINDER_SCAN_CHUNKS;

  uint64_t n_below = 0;
#pragma omp parallel for schedule(static) reduction(+ : n_below)
  for (uint64_t w = 0; w < n_words; w++)
    n_below += SIF_POPCOUNT_U64(below->words[w]);

  if (n_below == 0)
    return SIF_OK;

  int status = gauss_smooth(f, n, survey);
  if (status != SIF_OK)
    return status;

#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t w_end = chunk_start(c + 1, chunk, n_words);
    sif_real s[27];
    uint64_t idx[27];
    uint64_t n_cand = 0, n_min = 0;

    for (uint64_t w = chunk_start(c, chunk, n_words); w < w_end; w++) {
      for (uint64_t b = below->words[w]; b; b &= b - 1) {
        uint32_t ix, iy, iz;
        sif__unflatten_index(grid, w * 64 + SIF_CTZ_U64(b), &ix, &iy, &iz);
        load_stencil(f, n, ix, iy, iz, s, idx);
        n_min += is_minimum(s, idx);
        n_cand++;
      }
    }
    cand_offsets[c] = n_cand;
    min_offsets[c] = n_min;
  }

  /* Under minima_only the minima are the candidates, and each chunk writes
   * as many of one as of the other. */
  uint64_t n_minima = 0;
  n_below = 0;
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t nm = min_offsets[c];
    const uint64_t nc = minima_only ? nm : cand_offsets[c];
    cand_offsets[c] = n_below;
    min_offsets[c] = n_minima;
    n_below += nc;
    n_minima += nm;
  }

  /* The index rides in 32 bits next to the depth. Minima are a small
   * fraction of the candidates, and 2^32 of them is past any grid that fits
   * in memory -- but a wrapped index would point at another void's offset. */
  if (n_minima >= UINT32_MAX) {
    SIF_LOG_ERROR("finder", "%" PRIu64 " minima: more than %u can be refined",
      n_minima, UINT32_MAX - 1);
    return SIF_ERR_RANGE;
  }

  status = buffer_reserve((void**)&candidates->items, &candidates->capacity,
    n_below, sizeof(sif_candidate_t));
  if (status == SIF_OK && refine)
    status = buffer_reserve((void**)&offsets->items, &offsets->capacity,
      n_minima, sizeof(sif_offset_t));
  if (status != SIF_OK)
    return status;

  /* Every minimum takes the next slot of its chunk's span whether or not it
   * moves, so where a slot lands never depends on how its neighbours fared;
   * the slot of one that stays on its node is simply never pointed at. */
  sif_candidate_t* cand = candidates->items;
  sif_offset_t* off = offsets->items;
  uint64_t n_fallback = 0;

#pragma omp parallel for schedule(static, 1) reduction(+ : n_fallback)
  for (int c = 0; c < FINDER_SCAN_CHUNKS; c++) {
    const uint64_t w_end = chunk_start(c + 1, chunk, n_words);
    sif_real s[27];
    uint64_t idx[27];
    uint64_t k = cand_offsets[c];
    uint64_t slot = min_offsets[c];

    for (uint64_t w = chunk_start(c, chunk, n_words); w < w_end; w++) {
      for (uint64_t b = below->words[w]; b; b &= b - 1) {
        uint32_t ix, iy, iz;
        sif__unflatten_index(grid, w * 64 + SIF_CTZ_U64(b), &ix, &iy, &iz);
        load_stencil(f, n, ix, iy, iz, s, idx);
        const int minimum = is_minimum(s, idx);
        if (minima_only && !minimum)
          continue;

        const int placed = place_candidate(s, idx, minimum, refine, below, n,
          ix, iy, iz, &cand[k], refine ? &off[slot] : NULL);
        if (placed == 1)
          cand[k].refined = (uint32_t)(slot + 1);
        n_fallback += placed == -1;
        slot += minimum;
        k++;
      }
    }
  }

  sort_candidates(cand, n_below);
  candidates->count = n_below;
  offsets->count = refine ? n_minima : 0;
  stats->n_minima = n_minima;
  stats->n_fallback = n_fallback;

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
    tag, "exact overlap exclusions:  %7" PRIu64, stats->rejected_overlap);
  SIF_LOG_TRACE(
    tag, "rescaling failures:        %7" PRIu64, stats->rejected_rescale);
  SIF_LOG_TRACE(
    tag, "proxy re-check exclusions: %7" PRIu64, stats->rejected_proxy_recheck);
  SIF_LOG_TRACE(
    tag, "exact re-check exclusions: %7" PRIu64, stats->rejected_exact);

  SIF_LOG_INFO(tag,
    "r = %6.2f | candidates: %7" PRIu64 " | accepted: %7" PRIu64
    " | total voids: %7" PRIu64 " | time: %6.3fs",
    (double)radius, stats->n_candidates, stats->accepted, total_voids,
    elapsed_s);

  SIF_LOG_FLUSH();
}

/* --- 4. Geometry --- */

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

SIF_HOT_LOOP uint8_t sif__overlap_exact(const sif_catalogue_t* cat, sif_real cx,
  sif_real cy, sif_real cz, sif_real r, sif_real max_r, sif_real box_length,
  const sif_cell_linked_list_t* cll, uint32_t p2_mask,
  sif_real overlap_fraction) {

  (void)p2_mask; /* the list has its own grid, see cll_mask below */

  const sif_real* cat_cx = SIF_ASSUME_ALIGNED(cat->cx);
  const sif_real* cat_cy = SIF_ASSUME_ALIGNED(cat->cy);
  const sif_real* cat_cz = SIF_ASSUME_ALIGNED(cat->cz);
  const sif_real* cat_r = SIF_ASSUME_ALIGNED(cat->radii);

  sif_real half_box = box_length * 0.5f;

  /* The furthest a colliding void can be: r + r_i - f * min(r, r_i) grows
   * with r_i, so it is largest at r_i = max_r. max_r + r * (1 - f), which
   * this used to be, is that only while r <= max_r -- and a rescaled radius
   * is often past every one accepted so far. At f = 1 a candidate of 60 then
   * missed a void of 10 centred 55 inside it. */
  sif_real search_rad = r + max_r - overlap_fraction * (r < max_r ? r : max_r);

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

/* --- 5. Marking --- */

SIF_HOT_LOOP void sif__mark_sphere(sif_bitmask_t* mask, sif_real cx,
  sif_real cy, sif_real cz, sif_real r_true, uint32_t n_cells, uint32_t p2_mask,
  sif_real cell_length) {

  const sif_real r2 = r_true * r_true;
  const sif_real inv_cell_length = 1.0f / cell_length;
  const sif_real box_length = (sif_real)n_cells * cell_length;
  const sif_real half_box = box_length * 0.5f;

  const int32_t cell_radius = (int32_t)(r_true * inv_cell_length) + 1;
  const int32_t centre_ix = (int32_t)(cx * inv_cell_length);
  const int32_t centre_iy = (int32_t)(cy * inv_cell_length);
  const int32_t centre_iz = (int32_t)(cz * inv_cell_length);

  const int32_t span = 2 * cell_radius + 1;

  /* Threads write bits that share 64-bit words, so the atomic setter is
   * mandatory here. Setting a bit is idempotent, so the result does not depend
   * on the schedule. */
#pragma omp parallel for schedule(static) if (span > 24)
  for (int32_t dx = -cell_radius; dx <= cell_radius; dx++) {
    const int32_t global_x = centre_ix + dx;
    const sif_real px = (sif_real)global_x * cell_length;
    sif_real dist_x = SIF_REAL_ABS(px - cx);
    dist_x = (dist_x > half_box) ? box_length - dist_x : dist_x;
    const sif_real dx2 = dist_x * dist_x;

    if (dx2 > r2)
      continue;

    const uint32_t wrap_x = sif__wrap_pbc(global_x, n_cells, p2_mask);

    for (int32_t dy = -cell_radius; dy <= cell_radius; dy++) {
      const int32_t global_y = centre_iy + dy;
      const sif_real py = (sif_real)global_y * cell_length;
      sif_real dist_y = SIF_REAL_ABS(py - cy);
      dist_y = (dist_y > half_box) ? box_length - dist_y : dist_y;
      const sif_real dxy2 = dx2 + dist_y * dist_y;

      if (dxy2 > r2)
        continue;

      const uint32_t wrap_y = sif__wrap_pbc(global_y, n_cells, p2_mask);

      for (int32_t dz = -cell_radius; dz <= cell_radius; dz++) {
        const int32_t global_z = centre_iz + dz;
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
