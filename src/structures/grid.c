/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/grid.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/settings_internal.h"
#include "core/system_internal.h"
#include "io/internal.h"
#include "math/fft.h"
#include "sif/core/settings.h"
#include "sif/io/grid_io.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/field_internal.h"
#include "utils/logger_internal.h"

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* Buckets the CIC counting sort splits a tile into. Fixed rather than taken
 * from the thread count, so the deposit order -- and with it the last bits of
 * every cell -- is a property of the input alone; see
 * grid_cic_deposit_tile(). */
#define GRID_CIC_CHUNKS 256

/*
 * Planes per CIC slab, and why the slab count is not the thread count.
 *
 * A cell on the first plane of a slab is summed in two parts: its own slab's
 * deposits, in particle order, and then the previous slab's ghost plane, added
 * whole by grid_cic_reduce_ghost_cells(). A cell anywhere else gets the same
 * contributions interleaved in particle order. Float addition does not
 * associate, so the two orders differ in the last bit -- and where the slab
 * boundaries fall is therefore part of the answer.
 *
 * Tied to the thread count, that made the grid differ bitwise between, say, 56
 * and 72 cores; the smoothed field then differed at threshold crossings and
 * the near-ties in the candidate sort, and the finders' greedy acceptance
 * turned that into a few hundred different voids. Fixing the width makes the
 * boundaries a property of the grid size alone.
 *
 * 16 planes keeps the ghost planes at 1/16 of the grid in extra memory, and
 * leaves 64 slabs at 1024^3 for the deposit to share among threads.
 */
#define GRID_CIC_SLAB_WIDTH 16u

static inline uint64_t grid_flat_index(
  const sif_grid_t* grid, uint32_t ix, uint32_t iy, uint32_t iz) {
  return (uint64_t)ix * grid->n_cells * grid->n_cells +
         (uint64_t)iy * grid->n_cells + (uint64_t)iz;
}

sif_grid_t* sif_grid_alloc(uint32_t n_cells, sif_real box_length) {

  /* Both are load-bearing before anything is allocated. n_cells == 0 makes
   * `n & (n - 1)` read as a power of two, so p2_mask becomes UINT32_MAX and
   * the shift below asks for the count of trailing zeros of zero, which is
   * undefined. A box_length of zero is quieter and worse: every allocation
   * succeeds and cell_length is 0, so the grid looks usable right up to the
   * first division by it. */
  if (n_cells == 0) {
    SIF_LOG_ERROR("grid", "n_cells: 0 (expected > 0)");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (!(box_length > (sif_real)0.0)) {
    SIF_LOG_ERROR("grid", "box_length: %g (expected > 0)", (double)box_length);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  sif_grid_t* grid = malloc(sizeof(sif_grid_t));
  if (!grid) {
    SIF_LOG_ERROR(
      "grid", "grid: cannot allocate (%zu bytes)", sizeof(sif_grid_t));
    sif__error_status_set(SIF_ERR_ALLOC);
    return NULL;
  }

  grid->n_cells = n_cells;
  grid->total_cells = (uint64_t)n_cells * n_cells * n_cells;
  grid->box_length = box_length;
  grid->cell_length = box_length / n_cells;

  grid->p2_mask = ((n_cells & (n_cells - 1)) == 0) ? (n_cells - 1) : 0;
  grid->p2_shift = grid->p2_mask ? SIF_CTZ_U32(n_cells) : 0;

  grid->content = SIF_GRID_EMPTY;
  grid->assignment = SIF_ASSIGNMENT_UNKNOWN;
  grid->interlaced = 0;
  grid->values = sif_calloc_aligned(grid->total_cells, sizeof(sif_real));
  if (!grid->values) {
    SIF_LOG_ERROR("grid", "grid: cannot allocate %u^3 cells (%.2f GiB)",
      n_cells, SIF__GIB(grid->total_cells * sizeof(sif_real)));
    sif__error_status_set(SIF_ERR_ALLOC);
    free(grid);
    return NULL;
  }

  SIF_LOG_TRACE("grid", "cubic grid initialised");
  return grid;
}

void sif_grid_free(sif_grid_t* grid) {
  if (!grid)
    return;

  /* values may legitimately be NULL: a finder can have taken the buffer. The
   * struct still has to be released. */
  sif_free_aligned(grid->values);
  free(grid);

  SIF_LOG_TRACE("grid", "cubic grid destroyed");
}

/* --- CIC construction --- */

/*
 * A CIC deposit touches the eight cells around a particle, so two threads
 * working on nearby particles write the same cells. The three usual answers
 * are atomics on every write, a lock per cell, or giving each thread private
 * memory; this file takes the third.
 *
 * Space is cut into slabs along x, one per thread, and every particle is
 * routed to the slab its cell falls in. A thread then owns its slab outright
 * and writes it with no synchronisation at all. What crosses a boundary is
 * the +1 cell of a deposit sitting in the last plane, which is why each slab
 * carries one extra plane -- the ghost -- that is folded into the neighbour
 * afterwards.
 *
 * The cost is memory: the slabs together hold the whole grid again, plus one
 * extra plane per slab. At 2048^3 that is a second 34 GB alive during the
 * assignment, which is the reason the slab count follows the thread count
 * rather than being made large for its own sake.
 */
typedef struct {
  sif_real* values;
  uint32_t ix_start;
  uint32_t ix_end;
  /**
   * Planes this slab owns. values holds lo + width + hi: lo ghost planes
   * below the owned ones and hi above, the reach of the assignment kernel
   * either side of a particle's base cell -- 0 and 1 for CIC, 1 and 1 for
   * TSC, 1 and 2 for PCS.
   */
  uint32_t width;
  uint32_t lo;
  uint32_t hi;
  uint32_t n_cells;
} grid_slab_t;

SIF_PURE_FUNCTION static inline uint64_t grid_slab_flat_index(
  const grid_slab_t* slab, uint32_t local_ix, uint32_t iy, uint32_t iz) {

  return (uint64_t)local_ix * slab->n_cells * slab->n_cells +
         (uint64_t)iy * slab->n_cells + (uint64_t)iz;
}

/*
 * One arena for every slab, sized in a first pass and carved up in a second.
 *
 * Two loops over the same arithmetic because the total has to be known before
 * the arena exists, and the offsets cannot be handed out before it does. One
 * allocation rather than n_threads of them is what keeps the slabs adjacent in
 * memory and the teardown a single free -- and each slab's run is padded to a
 * whole cache line so that two threads depositing into neighbouring slabs
 * never share the line at their boundary.
 */
static grid_slab_t* grid_slabs_alloc(
  uint32_t n_cells, int n_threads, uint32_t lo, uint32_t hi) {
  grid_slab_t* slabs = malloc(n_threads * sizeof(grid_slab_t));
  if (!slabs) {
    SIF_LOG_ERROR("grid_cic", "grid: cannot allocate %d CIC slab headers",
      n_threads);
    return NULL;
  }

  uint64_t total_slab_cells = 0;
  uint64_t align_elements = SIF_CACHE_LINE / sizeof(sif_real);

  for (int t = 0; t < n_threads; t++) {
    uint32_t ix_start = (uint32_t)t * (n_cells / n_threads);
    uint32_t ix_end = (t == n_threads - 1)
                        ? n_cells
                        : (uint32_t)(t + 1) * (n_cells / n_threads);
    uint32_t width = ix_end - ix_start;

    uint64_t slab_cells = (uint64_t)(lo + width + hi) * n_cells * n_cells;

    slab_cells = (slab_cells + align_elements - 1) & ~(align_elements - 1);
    total_slab_cells += slab_cells;
  }

  /* Zeroed, because the deposit loop accumulates into it. The size is the
   * grid again plus a plane per slab, which is the number that surprises: it
   * is why a grid that fits can still fail to be assigned. */
  sif_real* arena = sif_calloc_aligned(total_slab_cells, sizeof(sif_real));
  if (!arena) {
    SIF_LOG_ERROR("grid_cic",
      "grid: cannot allocate the CIC slabs (%d slabs over %u^3 cells, %.2f "
      "GiB on top of the grid itself)",
      n_threads, n_cells, SIF__GIB(total_slab_cells * sizeof(sif_real)));
    free(slabs);
    return NULL;
  }

  /* Second pass: the same arithmetic again, now assigning offsets. */
  uint64_t current_offset = 0;
  for (int t = 0; t < n_threads; t++) {
    slabs[t].n_cells = n_cells;
    slabs[t].ix_start = (uint32_t)t * (n_cells / n_threads);
    slabs[t].ix_end = (t == n_threads - 1)
                        ? n_cells
                        : (uint32_t)(t + 1) * (n_cells / n_threads);
    slabs[t].width = slabs[t].ix_end - slabs[t].ix_start;
    slabs[t].lo = lo;
    slabs[t].hi = hi;

    uint64_t slab_cells =
      (uint64_t)(lo + slabs[t].width + hi) * n_cells * n_cells;
    slab_cells = (slab_cells + align_elements - 1) & ~(align_elements - 1);

    slabs[t].values = arena + current_offset;
    current_offset += slab_cells;
  }

  return slabs;
}

static void grid_slabs_free(grid_slab_t* slabs, int n_threads) {
  if (slabs) {
    /* One arena behind all the slabs, and slab 0 sits at its base -- so this
     * frees the whole thing exactly once. The other slabs point into the
     * middle of it and must not be freed. */
    if (n_threads > 0 && slabs[0].values) {
      sif_free_aligned(slabs[0].values);
    }
    free(slabs);
  }
}

/*
 * Folds each slab's ghost plane into the neighbour that actually owns it.
 *
 * A deposit at the last plane of a slab writes its +1 cell into the ghost, so
 * after the assignment every slab holds a plane of weight belonging to the slab
 * to its right. The wrap at the end -- slab n-1's ghost folding into slab 0 --
 * is what makes the x axis periodic, and it is why the deposit loop can leave
 * the x index unwrapped while y and z are wrapped explicitly.
 *
 * Serial on purpose rather than by oversight: it is n_slabs planes of N^2, a
 * rounding error against the assignment, and the write pattern (each iteration
 * writing a different slab's plane 0) is easy to get subtly wrong in parallel
 * for nothing.
 */
static void grid_cic_reduce_ghost_cells(grid_slab_t* slabs, int n_threads) {
  /* Upper ghosts first, into the next slab's first planes, then lower ghosts
   * into the previous slab's last ones; for CIC, which has only the one upper
   * plane, this is exactly the fold it always was. Serial, so the order the
   * planes are added in is fixed. */
  for (int t = 0; t < n_threads; t++) {
    grid_slab_t* slab = &slabs[t];
    grid_slab_t* slab_next = &slabs[(t + 1) % n_threads];
    const uint32_t N = slab->n_cells;

    for (uint32_t g = 0; g < slab->hi; g++) {
      for (uint32_t iy = 0; iy < N; iy++) {
        for (uint32_t iz = 0; iz < N; iz++) {
          uint64_t ghost_idx =
            grid_slab_flat_index(slab, slab->lo + slab->width + g, iy, iz);
          uint64_t target_idx =
            grid_slab_flat_index(slab_next, slab_next->lo + g, iy, iz);
          slab_next->values[target_idx] += slab->values[ghost_idx];
        }
      }
    }
  }

  for (int t = 0; t < n_threads; t++) {
    grid_slab_t* slab = &slabs[t];
    grid_slab_t* slab_prev = &slabs[(t + n_threads - 1) % n_threads];
    const uint32_t N = slab->n_cells;

    for (uint32_t g = 0; g < slab->lo; g++) {
      for (uint32_t iy = 0; iy < N; iy++) {
        for (uint32_t iz = 0; iz < N; iz++) {
          uint64_t ghost_idx = grid_slab_flat_index(slab, g, iy, iz);
          uint64_t target_idx = grid_slab_flat_index(slab_prev,
            slab_prev->lo + slab_prev->width - slab->lo + g, iy, iz);
          slab_prev->values[target_idx] += slab->values[ghost_idx];
        }
      }
    }
  }
}

/*
 * Copies the owned planes of every slab into the grid, converting weight to
 * density on the way.
 *
 * The division by the cell volume happens here, once per cell, rather than in
 * the deposit loop where it would run eight times per particle. It is also
 * where SIF_GRID_DENSITY acquires its actual meaning: weight per unit volume,
 * not weight per cell.
 *
 * Ghost planes are not copied -- they were folded into their owners already,
 * and copying them would double-count every boundary cell.
 */
static void grid_cic_merge_slabs_to_grid(sif_grid_t* grid,
  const grid_slab_t* slabs, int n_threads, sif_real inv_cell_volume) {

#pragma omp parallel for schedule(static)
  for (int t = 0; t < n_threads; t++) {
    const grid_slab_t* slab = &slabs[t];
    const uint32_t N = grid->n_cells;

    for (uint32_t local_ix = 0; local_ix < slab->width; local_ix++) {
      uint32_t global_ix = slab->ix_start + local_ix;
      for (uint32_t iy = 0; iy < N; iy++) {
        for (uint32_t iz = 0; iz < N; iz++) {
          uint64_t src = grid_slab_flat_index(slab, slab->lo + local_ix, iy, iz);
          uint64_t dst = grid_flat_index(grid, global_ix, iy, iz);
          grid->values[dst] = slab->values[src] * inv_cell_volume;
        }
      }
    }
  }
}

/*
 * Cell index along one axis. Shared by the counting, scatter and deposit
 * passes so the three can never disagree about which slab owns a particle.
 * The >= N branch only fires on the exact upper edge after rounding, and
 * wraps it periodically to cell 0.
 */
static inline uint32_t cic_cell_x(sif_real x, sif_real inv_cell, uint32_t N) {
  uint32_t ix = (uint32_t)(x * inv_cell);
  return (ix >= N) ? 0u : ix;
}

/*
 * The base cell a scheme's kernel is anchored on: the cell below the particle
 * for the even-order kernels (CIC, PCS), the nearest node for the odd ones
 * (NGP, TSC), which are centred on it. Like cic_cell_x(), the one derivation
 * every pass shares, wrapped into [0, N).
 */
static inline uint32_t base_cell(
  sif_real u, sif_assignment_t scheme, uint32_t N) {
  const uint32_t i = (scheme == SIF_ASSIGNMENT_NGP ||
                       scheme == SIF_ASSIGNMENT_TSC)
                       ? (uint32_t)(u + 0.5f)
                       : (uint32_t)u;
  return (i >= N) ? i - N : i;
}

/* The kernel's reach below and above the base cell. */
static inline uint32_t kernel_lo(sif_assignment_t scheme) {
  return (scheme == SIF_ASSIGNMENT_TSC || scheme == SIF_ASSIGNMENT_PCS) ? 1u
                                                                        : 0u;
}
static inline uint32_t kernel_hi(sif_assignment_t scheme) {
  switch (scheme) {
  case SIF_ASSIGNMENT_NGP:
    return 0u;
  case SIF_ASSIGNMENT_PCS:
    return 2u;
  default:
    return 1u;
  }
}

/*
 * The kernel's weights along one axis, for the particle at u (in cells) whose
 * base cell is `base`, lowest first: w[j] goes to cell base - lo + j. The
 * fractional distances are taken from the unwrapped base, so a particle that
 * rounded up to the top face has the distance it really has.
 *
 *   NGP  1
 *   TSC  (1/2 - d)^2 / 2, 3/4 - d^2, (1/2 + d)^2 / 2   with d = u - nearest
 *   PCS  (1 - t)^3 / 6, (4 - 6t^2 + 3t^3) / 6,
 *        (4 - 6(1-t)^2 + 3(1-t)^3) / 6, t^3 / 6        with t = u - floor(u)
 *
 * Each set sums to one.
 */
static inline void kernel_weights(
  sif_real u, sif_assignment_t scheme, sif_real w[4]) {
  switch (scheme) {
  case SIF_ASSIGNMENT_NGP:
    w[0] = 1.0f;
    break;
  case SIF_ASSIGNMENT_TSC: {
    const sif_real d = u - (sif_real)(uint32_t)(u + 0.5f);
    const sif_real a = 0.5f - d, b = 0.5f + d;
    w[0] = 0.5f * a * a;
    w[1] = 0.75f - d * d;
    w[2] = 0.5f * b * b;
    break;
  }
  case SIF_ASSIGNMENT_PCS: {
    const sif_real t = u - (sif_real)(uint32_t)u;
    const sif_real r = 1.0f - t;
    const sif_real t2 = t * t, r2 = r * r;
    w[0] = r2 * r / 6.0f;
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t2 * t) / 6.0f;
    w[2] = (4.0f - 6.0f * r2 + 3.0f * r2 * r) / 6.0f;
    w[3] = t2 * t / 6.0f;
    break;
  }
  default: {
    const sif_real t = u - (sif_real)(uint32_t)u;
    w[0] = 1.0f - t;
    w[1] = t;
    break;
  }
  }
}

/* A coordinate as the passes read it: moved by `shift` -- half a cell for the
 * second grid of an interlaced deposit, otherwise none -- and folded into the
 * box. With no shift the coordinate is read untouched. */
static inline sif_real deposit_coord(sif_real v, sif_real shift, sif_real box) {
  return sif__box_fold(shift != 0.0f ? v + shift : v, box);
}

/*
 * Bins one run of particles by destination slab and deposits it.
 *
 * The counting sort is what lets the deposit hand each thread one contiguous
 * run of particles that all land in the slab it owns -- without it, a thread
 * would have to test every particle against its own range and skip most of
 * them. `counts` and `sorted` are scratch owned by the caller so that a run
 * over many tiles allocates them once; both are indexed from zero within the
 * tile, while the values stored in `sorted` are indices into the whole field.
 *
 * The chunk count is fixed rather than taken from the thread count, so the
 * bucket layout -- and therefore the order weight is accumulated in -- is a
 * property of the input alone. A grid that changed in its last bits with the
 * number of cores would make the cache keyed on it worthless, and would put a
 * thread count into every reproducibility claim downstream.
 */
static void grid_cic_deposit_tile(grid_slab_t* slabs, int n_slabs,
  const sif_real* xs, const sif_real* ys, const sif_real* zs,
  const sif_real* weights, uint64_t tile_lo, uint64_t tile_hi, uint64_t* counts,
  uint64_t* sorted, uint32_t N, sif_real idx, uint32_t base_width,
  sif_real box, sif_assignment_t scheme, sif_real shift) {

  const int n_chunks = GRID_CIC_CHUNKS;
  const uint64_t tile_n = tile_hi - tile_lo;
  const uint64_t chunk_size = (tile_n + n_chunks - 1) / n_chunks;

  memset(counts, 0, (size_t)n_chunks * n_slabs * sizeof(uint64_t));

/* pass 1: how many particles each (chunk, slab) pair owns. Each chunk writes
 * its own row, so the counts need no synchronisation. */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < n_chunks; c++) {
    uint64_t start = tile_lo + (uint64_t)c * chunk_size;
    uint64_t end =
      (start + chunk_size > tile_hi) ? tile_hi : start + chunk_size;

    for (uint64_t p = start; p < end; p++) {
      uint32_t ix =
        base_cell(deposit_coord(xs[p], shift, box) * idx, scheme, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));
      counts[(uint64_t)c * n_slabs + target_slab]++;
    }
  }

  /* pass 2: turn the counts into write offsets, slab-major -- every chunk's
   * share of slab s is laid out consecutively, so slab s ends up owning one
   * unbroken run of the sorted array. That is what pass 4 slices on. Serial
   * because a prefix sum is, and it is n_chunks * n_slabs entries. */

  uint64_t current_offset = 0;

  for (int32_t slab = 0; slab < n_slabs; slab++) {
    for (int c = 0; c < n_chunks; c++) {
      uint64_t flat_idx = (uint64_t)c * n_slabs + slab;
      uint64_t count = counts[flat_idx];
      counts[flat_idx] = current_offset;
      current_offset += count;
    }
  }

/* pass 3: scatter the particle indices into their slab's run. Each chunk
 * advances only its own offsets, so again no synchronisation -- and the order
 * within a slab is the order the chunks were laid out in, not the order the
 * threads happened to finish.
 *
 * Indices rather than copies of the coordinates: 8 bytes per particle instead
 * of 24 (or 32 with weights), at the price of the indirect read in pass 4. */
#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < n_chunks; c++) {
    uint64_t start = tile_lo + (uint64_t)c * chunk_size;
    uint64_t end =
      (start + chunk_size > tile_hi) ? tile_hi : start + chunk_size;

    for (uint64_t p = start; p < end; p++) {
      uint32_t ix =
        base_cell(deposit_coord(xs[p], shift, box) * idx, scheme, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));

      uint64_t flat_idx = (uint64_t)c * n_slabs + target_slab;
      sorted[counts[flat_idx]] = p;
      counts[flat_idx]++;
    }
  }

/* pass 4: deposit. One thread per slab, each over its own contiguous run,
 * writing only into memory it owns. */
#pragma omp parallel for schedule(static, 1)
  for (int32_t slab_idx = 0; slab_idx < n_slabs; slab_idx++) {

    /* Pass 3 left every offset advanced to the end of its (chunk, slab) run,
     * so the last chunk's entry for a slab is that slab's end -- and the
     * previous slab's end is this one's start, the runs being contiguous. */
    uint64_t slab_start =
      (slab_idx == 0)
        ? 0
        : counts[(uint64_t)(n_chunks - 1) * n_slabs + (slab_idx - 1)];
    uint64_t slab_end = counts[(uint64_t)(n_chunks - 1) * n_slabs + slab_idx];

    grid_slab_t* slab = &slabs[slab_idx];
    const uint32_t x0 = slab->ix_start;

    sif_real* s_values = SIF_ASSUME_ALIGNED(slab->values);

    if (scheme != SIF_ASSIGNMENT_CIC) {
      /* The general kernel: p weights per axis, p^3 writes. The x index runs
       * from the slab's lower ghost planes up into its upper ones, never past
       * them, since a particle is routed by its base cell; y and z wrap. */
      const uint32_t lo = kernel_lo(scheme);
      const uint32_t width = lo + 1 + kernel_hi(scheme);

      for (uint64_t p = slab_start; p < slab_end; p++) {
        const uint64_t orig_p = sorted[p];
        const sif_real ux = deposit_coord(xs[orig_p], shift, box) * idx;
        const sif_real uy = deposit_coord(ys[orig_p], shift, box) * idx;
        const sif_real uz = deposit_coord(zs[orig_p], shift, box) * idx;

        sif_real wx[4], wy[4], wz[4];
        kernel_weights(ux, scheme, wx);
        kernel_weights(uy, scheme, wy);
        kernel_weights(uz, scheme, wz);

        const uint32_t lix = base_cell(ux, scheme, N) - x0;
        const uint32_t by = base_cell(uy, scheme, N);
        const uint32_t bz = base_cell(uz, scheme, N);

        uint32_t jy[4], jz[4];
        for (uint32_t j = 0; j < width; j++) {
          jy[j] = (by + N + j - lo) % N;
          jz[j] = (bz + N + j - lo) % N;
        }

        const sif_real particle_weight = weights ? weights[orig_p] : 1.0f;

        for (uint32_t a = 0; a < width; a++) {
          const sif_real wa = particle_weight * wx[a];
          for (uint32_t b = 0; b < width; b++) {
            const sif_real wab = wa * wy[b];
            const uint64_t row = grid_slab_flat_index(slab, lix + a, jy[b], 0);
            for (uint32_t c = 0; c < width; c++)
              s_values[row + jz[c]] += wab * wz[c];
          }
        }
      }
      continue;
    }

    for (uint64_t p = slab_start; p < slab_end; p++) {
      /* The indirection costs less than it looks: consecutive entries in a
       * slab's run come from the same chunk, so the reads walk the coordinate
       * arrays in roughly increasing order rather than jumping about. */
      uint64_t orig_p = sorted[p];

      /* Folded, as in passes 1 and 3: a coordinate within rounding of a face
       * was accepted by the validator, and all three passes have to see it at
       * the same place. A no-op for every coordinate already in the box. */
      const sif_real cx = deposit_coord(xs[orig_p], shift, box) * idx;
      const sif_real cy = deposit_coord(ys[orig_p], shift, box) * idx;
      const sif_real cz = deposit_coord(zs[orig_p], shift, box) * idx;
      /* The same >= N fallback cic_cell_x applies, spelled out here because
       * the fractional part is needed anyway. The two must agree exactly: this
       * derives the cell a second time, and a particle that lands in a
       * different slab than pass 1 sent it to would write outside this
       * thread's memory. */
      uint32_t raw_cx = (uint32_t)cx, raw_cy = (uint32_t)cy,
               raw_cz = (uint32_t)cz;

      uint32_t ix = (raw_cx >= N) ? 0 : raw_cx;
      uint32_t iy = (raw_cy >= N) ? 0 : raw_cy;
      uint32_t iz = (raw_cz >= N) ? 0 : raw_cz;

      /* y and z wrap here; x does not, and must not. Its +1 cell is allowed to
       * land in the ghost plane, which the reduction afterwards folds into the
       * next slab -- wrapping it now would write into another thread's slab. */
      uint32_t iy1 = (iy + 1 == N) ? 0 : iy + 1;
      uint32_t iz1 = (iz + 1 == N) ? 0 : iz + 1;

      const uint32_t lix = ix - x0;
      const uint32_t lix1 = lix + 1;

      const sif_real tx = cx - (sif_real)raw_cx;
      const sif_real ty = cy - (sif_real)raw_cy;
      const sif_real tz = cz - (sif_real)raw_cz;

      const sif_real wx0 = 1.0f - tx, wx1 = tx;
      const sif_real wy0 = 1.0f - ty, wy1 = ty;
      const sif_real wz0 = 1.0f - tz, wz1 = tz;

      /* The eight trilinear weights factor as (x,y) * z, so the four xy
       * products and the two z terms below cover all eight corners with one
       * multiply each at the point of use -- 14 multiplications rather than
       * the 24 the fully written-out form costs. */
      const sif_real w00 = wx0 * wy0;
      const sif_real w01 = wx0 * wy1;
      const sif_real w10 = wx1 * wy0;
      const sif_real w11 = wx1 * wy1;

      /* The particle weight, or 1 for an unweighted field. */
      const sif_real particle_weight = weights ? weights[orig_p] : 1.0f;

      const sif_real mz0 = particle_weight * wz0;
      const sif_real mz1 = particle_weight * wz1;

      /* 8 writes, exactly 1 multiplication each */
      s_values[grid_slab_flat_index(slab, lix, iy, iz)] += w00 * mz0;
      s_values[grid_slab_flat_index(slab, lix, iy, iz1)] += w00 * mz1;
      s_values[grid_slab_flat_index(slab, lix, iy1, iz)] += w01 * mz0;
      s_values[grid_slab_flat_index(slab, lix, iy1, iz1)] += w01 * mz1;
      s_values[grid_slab_flat_index(slab, lix1, iy, iz)] += w10 * mz0;
      s_values[grid_slab_flat_index(slab, lix1, iy, iz1)] += w10 * mz1;
      s_values[grid_slab_flat_index(slab, lix1, iy1, iz)] += w11 * mz0;
      s_values[grid_slab_flat_index(slab, lix1, iy1, iz1)] += w11 * mz1;
    }
  }
}

/*
 * Particles per tile, and why the sort is tiled at all.
 *
 * The index array is 8 bytes per particle. Sorting the whole field at once
 * makes that 27 GB at 3.4e9 tracers and 69 GB at 8.6e9 -- for a scratch buffer
 * that never leaves this function. Tiling caps it at this many entries
 * regardless of how many particles there are.
 *
 * It is also faster, which is not the trade one expects from a memory fix. The
 * deposit reads coordinates through the sorted indices, and over a whole field
 * those reads range over every position; confined to a tile they stay in a
 * slice of it, and the array itself is faulted in once and then reused. What
 * tiling costs is one extra pass over the slabs per tile, so the useful range
 * is wide but not unbounded -- measured, anything from a few tiles to a few
 * dozen beats the untiled sort, and the gain flattens well before the tiles
 * get small enough to matter.
 */
static uint64_t grid_cic_tile_size(void) {
  return (uint64_t)sif__setting_number("cic_tile_particles");
}

/*
 * @return SIF_OK, or SIF_ERR_ALLOC with the grid untouched: nothing is written
 * to it until the slabs are merged at the very end.
 */
static int grid_compute_cic(sif_grid_t* grid, const sif_real* xs,
  const sif_real* ys, const sif_real* zs, const sif_real* weights,
  uint64_t n_particles, sif_assignment_t scheme, sif_real shift) {

  const uint32_t N = grid->n_cells;
  const sif_real idx = (sif_real)N / (sif_real)grid->box_length;

  /* From the grid size alone, never the thread count: see
   * GRID_CIC_SLAB_WIDTH. The last slab takes the remainder. */
  int32_t n_slabs = (int32_t)(N / GRID_CIC_SLAB_WIDTH);
  if (n_slabs < 1)
    n_slabs = 1;

  grid_slab_t* slabs =
    grid_slabs_alloc(N, n_slabs, kernel_lo(scheme), kernel_hi(scheme));
  if (!slabs)
    return SIF_ERR_ALLOC;

  uint64_t tile_n = grid_cic_tile_size();
  if (tile_n > n_particles)
    tile_n = n_particles;
  if (tile_n == 0)
    tile_n = 1;

  uint64_t* counts =
    malloc((size_t)GRID_CIC_CHUNKS * n_slabs * sizeof(uint64_t));
  uint64_t* sorted_indices = malloc(tile_n * sizeof(uint64_t));

  if (!counts || !sorted_indices) {
    SIF_LOG_ERROR("grid_cic",
      "grid: cannot allocate the CIC sort scratch (%" PRIu64
      " particles per tile, %.2f GiB); lower the cic_tile_particles setting",
      tile_n, SIF__GIB(tile_n * sizeof(uint64_t)));
    free(counts);
    free(sorted_indices);
    grid_slabs_free(slabs, n_slabs);
    return SIF_ERR_ALLOC;
  }

  const uint32_t base_width = N / n_slabs;

  SIF_LOG_TRACE("grid_cic",
    "binning %" PRIu64 " particles in %" PRIu64 " tile(s) of %" PRIu64
    " (%.2f GiB of sort scratch)",
    n_particles, (n_particles + tile_n - 1) / tile_n, tile_n,
    (double)(tile_n * sizeof(uint64_t)) / (1024.0 * 1024.0 * 1024.0));

  for (uint64_t lo = 0; lo < n_particles; lo += tile_n) {
    const uint64_t hi = (lo + tile_n > n_particles) ? n_particles : lo + tile_n;

    grid_cic_deposit_tile(slabs, n_slabs, xs, ys, zs, weights, lo, hi, counts,
      sorted_indices, N, idx, base_width, grid->box_length, scheme, shift);
  }

  grid_cic_reduce_ghost_cells(slabs, n_slabs);

  const sif_real inv_cell_vol =
    1.0f /
    (sif_real)(grid->cell_length * grid->cell_length * grid->cell_length);
  grid_cic_merge_slabs_to_grid(grid, slabs, n_slabs, inv_cell_vol);

  free(sorted_indices);
  free(counts);
  grid_slabs_free(slabs, n_slabs);

  SIF_LOG_TRACE("grid", "assignment completed");
  return SIF_OK;
}

/*
 * Whether the on-disk .xgrid cache is active. Off by default: a single 2048^3
 * grid is ~34 GB, and the cache is never evicted, so opting in has to be a
 * deliberate act.
 *
 *   sif_setting_set("grid_cache_enabled", "1");
 */
static int grid_cache_enabled(void) {
  return sif__setting_flag("grid_cache_enabled");
}

/*
 * Version of the CIC deposition itself.
 *
 * Bumped to 2 when the counting sort became tiled: the same particles are
 * deposited, but a tile at a time, so weight accumulates into a cell in a
 * different order and the sums move in their last bits.
 *
 * Folded into every cache key, so that changing how weight is deposited -- a
 * boundary fix, a different rounding, a change of accumulation order -- makes
 * every existing entry a miss instead of leaving it silently valid. Without
 * this the cache would keep serving the previous implementation's answer to
 * anyone who had run before the change, which is the one cache failure that
 * survives a rebuild and looks like a physics result.
 *
 * Bump it whenever grid_compute_cic() stops producing bit-identical output.
 */
#define GRID_CIC_VERSION 3u

/* Blocks the content hash splits each array into. Fixed rather than derived
 * from the thread count, so the key is a property of the data alone: a grid
 * cached on an 8-core machine has to be found again on a 64-core one. */
#define GRID_HASH_BLOCKS 64

/*
 * The key is two 64-bit lanes rather than one.
 *
 * Both are advanced in the same pass, since the pass over memory is the entire
 * cost and a second accumulator rides along in a register. 128 bits puts the
 * collision probability over any plausible number of cached grids far below
 * the chance of the disk lying about the bytes -- and unlike the 64-bit case,
 * far enough below that the provenance check in the reader is a backstop
 * rather than the thing actually doing the work.
 */
typedef struct {
  uint64_t a;
  uint64_t b;
} grid_key_t;

/* One round of the splitmix64 finaliser, which is what utils/random.h already
 * uses to decorrelate a seed. Cheap, and good enough to avalanche a float. */
static inline uint64_t grid_mix(uint64_t h, uint64_t v) {
  h ^= v;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 29;
  return h;
}

static inline void key_mix(grid_key_t* k, uint64_t v) {
  k->a = grid_mix(k->a, v);
  /* The lanes differ only by a constant offset on the input, which is enough
   * to make them independent for collision purposes without a second pass. */
  k->b = grid_mix(k->b, v + 0x9e3779b97f4a7c15ULL);
}

/*
 * The bits of one sif_real, through memcpy rather than a pointer cast: reading
 * a float through a uint32_t* is undefined, and every compiler turns this into
 * a register move.
 *
 * Bitwise on purpose. -0.0 and +0.0 compare equal but deposit identically,
 * while two values differing in the last bit deposit differently -- so bits,
 * not value, are what decide whether a cached grid applies.
 */
static inline uint64_t grid_real_bits(sif_real v) {
#ifdef SIF_USE_DOUBLE
  uint64_t bits;
#else
  uint32_t bits;
#endif
  memcpy(&bits, &v, sizeof(bits));
  return (uint64_t)bits;
}

/*
 * Folds one array of n values into the key.
 *
 * Split into a fixed number of blocks hashed in parallel and combined in index
 * order, the same shape as the reductions in utils/array.c and for the same
 * reason: the answer must not depend on how many threads ran. Each block is
 * seeded with its own index, so moving a run of values from one block to
 * another changes the key even if the multiset of values does not.
 *
 * Exactly n elements are read, never the allocation's padding -- the field's
 * arrays are padded to a cache line and that padding is uninitialised, so
 * hashing it would make the key differ between two runs on identical data.
 */
static void key_fold_array(grid_key_t* key, const sif_real* a, uint64_t n) {
  if (!a || n == 0) {
    key_mix(key, 0);
    return;
  }

  grid_key_t partial[GRID_HASH_BLOCKS];

  const uint64_t chunk = n / GRID_HASH_BLOCKS;
  const uint64_t rem = n % GRID_HASH_BLOCKS;

#pragma omp parallel for schedule(static)
  for (int b = 0; b < GRID_HASH_BLOCKS; b++) {
    const uint64_t ub = (uint64_t)b;
    const uint64_t lo = ub * chunk + (ub < rem ? ub : rem);
    const uint64_t hi = lo + chunk + (ub < rem ? 1 : 0);

    grid_key_t local = {key->a ^ ub, key->b ^ (ub * 0x9e3779b97f4a7c15ULL)};
    for (uint64_t i = lo; i < hi; i++)
      key_mix(&local, grid_real_bits(a[i]));

    partial[b] = local;
  }

  for (int b = 0; b < GRID_HASH_BLOCKS; b++) {
    key_mix(key, partial[b].a);
    key_mix(key, partial[b].b);
  }
}

/*
 * The cache key: everything that can change the grid this call produces.
 *
 * Shape, the deposition version, and the full contents of every array that is
 * read -- positions always, weights when present. Hashing the weight *flag*
 * alone was a defect rather than an approximation: two runs over one set of
 * positions with different weights produced the same key, and the second
 * silently received the first one's grid.
 */
static grid_key_t grid_cache_key(const sif_grid_t* grid,
  const sif_field_t* field, sif_assignment_t scheme, int interlace) {

  grid_key_t key = {0xcbf29ce484222325ULL, 0x9e3779b97f4a7c15ULL};

  key_mix(&key, GRID_CIC_VERSION);

  /* Mixed in only when it is not plain CIC, so every key cached before the
   * schemes existed -- all of them CIC -- still names the same grid. */
  if (scheme != SIF_ASSIGNMENT_CIC || interlace) {
    key_mix(&key, (uint64_t)scheme);
    key_mix(&key, interlace ? 0x1a7e41aceULL : 0u);
  }
  key_mix(&key, grid->n_cells);
  key_mix(&key, grid_real_bits(grid->box_length));
  key_mix(&key, field->n_particles);
  key_mix(&key, field->weights ? 1u : 0u);

  /* Folded one after another rather than interleaved, so the key depends on
   * which array a value came from and not merely on the set of values. */
  key_fold_array(&key, field->x, field->n_particles);
  key_fold_array(&key, field->y, field->n_particles);
  key_fold_array(&key, field->z, field->n_particles);

  if (field->weights)
    key_fold_array(&key, field->weights, field->n_particles);

  return key;
}

/*
 * The interlaced deposit, into grid->values: the plain grid and the one with
 * every particle moved by half a cell, combined in Fourier space as
 *
 *   delta(k) = [ delta_0(k) + delta_s(k) exp(i k . s) ] / 2,   s = (l/2)(1,1,1)
 *
 * The phase undoes the shift for the true modes, while an aliased image at
 * k + 2 pi n / l picks up exp(i pi (n_x + n_y + n_z)) on top -- +1 or -1 --
 * so the images with an odd index sum cancel between the two grids.
 *
 * Memory: the shifted spectrum is kept while the plain grid is deposited and
 * transformed, and the combination needs the workspace's backward buffer, so
 * three spectrum-sized buffers are alive at the peak. The real-space cells are
 * released before the last of them is allocated and replaced by the result.
 */
static int grid_compute_interlaced(sif_grid_t* grid, const sif_field_t* field,
  sif_assignment_t scheme) {

  sif_system_state_t* state = sif__system_state();
  if (!state)
    return SIF_ERR_UNINITIALISED;

  const uint32_t N = grid->n_cells;
  const sif_real half_cell = 0.5f * grid->cell_length;

  sif_fft_workspace_t* ws = sif__fft_workspace_alloc(state->fft_mgr, N);
  if (!ws)
    return SIF_ERR_ALLOC;

  const uint64_t n_modes = (uint64_t)N * N * (N / 2 + 1);
  sif_real_complex* shifted = NULL;
  int status = grid_compute_cic(grid, field->x, field->y, field->z,
    field->weights, field->n_particles, scheme, half_cell);
  if (status == SIF_OK)
    status = sif__fft_grid_forward(ws, grid);
  if (status == SIF_OK) {
    shifted = sif_malloc_aligned(n_modes * sizeof(sif_real_complex));
    if (!shifted) {
      SIF_LOG_ERROR("grid", "interlacing: cannot allocate a %u^3 spectrum "
        "(%.2f GiB)", N, SIF__GIB(n_modes * sizeof(sif_real_complex)));
      status = SIF_ERR_ALLOC;
    } else {
      memcpy(shifted, ws->delta_k, n_modes * sizeof(sif_real_complex));
    }
  }
  if (status == SIF_OK)
    status = grid_compute_cic(grid, field->x, field->y, field->z,
      field->weights, field->n_particles, scheme, 0.0f);
  if (status == SIF_OK)
    status = sif__fft_grid_forward(ws, grid);

  /* From here the cells are not needed; letting them go before the backward
   * buffer is allocated is what keeps the peak at three spectra. */
  if (status == SIF_OK) {
    sif_free_aligned(grid->values);
    grid->values = NULL;
    status = sif__fft_workspace_init_backward(ws, state->fft_mgr);
  }

  if (status == SIF_OK) {
    const uint32_t half = N / 2;
    sif_real_complex* out = ws->delta_k_cpy;
    const sif_real_complex* plain = ws->delta_k;

#pragma omp parallel for schedule(static)
    for (uint32_t ix = 0; ix < N; ix++) {
      const int32_t kx = (ix > half) ? (int32_t)ix - (int32_t)N : (int32_t)ix;
      for (uint32_t iy = 0; iy < N; iy++) {
        const int32_t ky =
          (iy > half) ? (int32_t)iy - (int32_t)N : (int32_t)iy;
        for (uint32_t iz = 0; iz <= half; iz++) {
          const double phase =
            SIF_PI * (double)(kx + ky + (int32_t)iz) / (double)N;
          const double c = cos(phase), sn = sin(phase);
          const uint64_t m = ((uint64_t)ix * N + iy) * (half + 1) + iz;
          const double re = shifted[m][0] * c - shifted[m][1] * sn;
          const double im = shifted[m][0] * sn + shifted[m][1] * c;
          out[m][0] = (sif_real)(0.5 * (plain[m][0] + re));
          out[m][1] = (sif_real)(0.5 * (plain[m][1] + im));
        }
      }
    }
    sif_free_aligned(shifted);
    shifted = NULL;

    if (!sif__fft_grid_backward(ws)) {
      status = SIF_ERR_ALLOC;
    } else {
      grid->values = sif__fft_workspace_take_real_buffer(ws);
    }
  }

  sif_free_aligned(shifted);
  sif__fft_workspace_free(ws);

  /* A failure after the cells were released leaves an empty grid rather than
   * a dangling one, as the header promises. */
  if (!grid->values) {
    grid->values = sif_calloc_aligned(grid->total_cells, sizeof(sif_real));
    grid->content = SIF_GRID_EMPTY;
    grid->assignment = SIF_ASSIGNMENT_UNKNOWN;
    grid->interlaced = 0;
    if (status == SIF_OK)
      status = SIF_ERR_ALLOC;
  }

  if (status == SIF_OK)
    SIF_LOG_TRACE("grid", "interlaced assignment completed");
  return status;
}

static const char* scheme_name(sif_assignment_t scheme) {
  switch (scheme) {
  case SIF_ASSIGNMENT_NGP:
    return "NGP";
  case SIF_ASSIGNMENT_CIC:
    return "CIC";
  case SIF_ASSIGNMENT_TSC:
    return "TSC";
  case SIF_ASSIGNMENT_PCS:
    return "PCS";
  default:
    return "unknown";
  }
}

int sif_grid_assign(sif_grid_t* grid, const sif_field_t* field,
  sif_assignment_t scheme, sif_option opt) {
  if (!grid || !grid->values) {
    SIF_LOG_ERROR("grid_cic", "grid: %s", !grid ? "NULL" : "no cell values");
    return SIF_ERR_INVALID;
  }
  if (scheme < SIF_ASSIGNMENT_NGP || scheme > SIF_ASSIGNMENT_PCS) {
    SIF_LOG_ERROR("grid", "scheme: %d (expected NGP, CIC, TSC or PCS)",
      (int)scheme);
    return SIF_ERR_INVALID;
  }
  /* A PCS kernel reaches four cells; on fewer it would deposit a particle
   * onto the same cell twice through the wrap, and the slabs' ghost planes
   * would overrun the slab they fold into. */
  if (grid->n_cells < 4) {
    SIF_LOG_ERROR("grid", "n_cells: %u (expected at least 4 to assign onto)",
      grid->n_cells);
    return SIF_ERR_INVALID;
  }
  const int interlace = (opt & SIF_GRID_INTERLACE) != 0;
  /* An empty field would deposit nothing and come back a grid of zeros, which
   * no later step can tell from a real one until a mean of zero divides
   * something. A field emptied by sif_chain_mesh_alloc_consume() is the usual
   * way to get here, and the check says so. */
  const int checked =
    sif__field_check(field, SIF__FIELD_NEED_CARTESIAN, "grid_cic", "field");
  if (checked != SIF_OK)
    return checked;

  /* The CIC scatter derives a particle's slab twice, once when counting and
   * once when depositing, and the two derivations only agree for coordinates
   * inside the box -- so anything else is refused here rather than letting the
   * deposit pass underflow its local slab index. Coordinates within rounding
   * of a face are let through and folded as the deposit reads them. */
  const int valid = sif__field_check_in_box(field->x, field->y, field->z,
    field->n_particles, grid->box_length, "grid_cic", "field", NULL);
  if (valid != SIF_OK)
    return valid;

  const int use_cache = grid_cache_enabled();

  char final_path[1024] = {0};
  grid_key_t key = {0, 0};

  if (use_cache) {
    /* One streaming pass over the field. It is paid only when the cache is
     * enabled, and against the assignment below -- which reads the same arrays
     * and then scatters eight read-modify-writes per particle across a grid
     * far larger than cache -- it is a small fraction of what a hit saves. */
    key = grid_cache_key(grid, field, scheme, interlace);

    /* Created here rather than at start-up, since the cache is off by
     * default. A directory that cannot be used has been reported, and the grid
     * is computed without the cache. */
    const char* cache_dir = NULL;
    if (sif__setting_directory("cache_directory", &cache_dir) == SIF_OK) {
      snprintf(final_path, sizeof(final_path), "%s/grid_%016llx%016llx.xgrid",
        cache_dir, (unsigned long long)key.b, (unsigned long long)key.a);

      /* Keyed, so a file found by name still has to say it came from this
       * field before its contents are believed. */
      const uint64_t expect[2] = {key.a, key.b};
      if (sif__grid_read_into_keyed(grid, final_path, expect) == SIF_OK) {
        SIF_LOG_INFO("grid_cic", "loaded cached .xgrid from %s", final_path);
        return SIF_OK;
      }

      SIF_LOG_TRACE("grid_cic", "grid is not cached. starting assignment");
    }
  }

  const int status =
    interlace ? grid_compute_interlaced(grid, field, scheme)
              : grid_compute_cic(grid, field->x, field->y, field->z,
                  field->weights, field->n_particles, scheme, 0.0f);
  if (status != SIF_OK)
    return status;
  grid->content = SIF_GRID_DENSITY;
  grid->assignment = scheme;
  grid->interlaced = (uint32_t)interlace;
  SIF_LOG_TRACE("grid", "%s%s assignment onto %u^3", interlace ? "interlaced " : "",
    scheme_name(scheme), grid->n_cells);

  /* final_path stays empty when the cache is off, or on when the directory
   * could not be resolved -- both mean there is nothing to write. */
  if (!use_cache || final_path[0] == '\0') {
    SIF_LOG_FLUSH();
    return SIF_OK;
  }

  /* Through a temporary and a rename: a job array runs many ranks against one
   * cache, and rename is what stops a reader from finding a half-written file
   * that would then pass its checksum only by accident. */
  char tmp_path[1100];
  snprintf(
    tmp_path, sizeof(tmp_path), "%s.tmp.%ld", final_path, (long)getpid());

  const uint64_t source_key[2] = {key.a, key.b};

  if (sif__grid_write_keyed(tmp_path, grid, source_key) == SIF_OK) {
    if (rename(tmp_path, final_path) == 0) {
      SIF_LOG_TRACE("grid_cic", "%s: grid cached", final_path);
    } else {
      SIF_LOG_WARNING("grid_cic",
        "%s: grid computed but not cached: cannot rename %s into place: %s",
        final_path, tmp_path, strerror(errno));
      remove(tmp_path);
    }
  } else {
    SIF_LOG_WARNING("grid_cic",
      "%s: grid computed but not cached: the write failed (see the error "
      "above)",
      tmp_path);
    remove(tmp_path);
  }

  /* A cache that could not be written is not the caller's problem: the grid
   * they asked for is in their hands either way. */
  SIF_LOG_FLUSH();
  return SIF_OK;
}

int sif_grid_assign_cic(sif_grid_t* grid, const sif_field_t* field) {
  return sif_grid_assign(grid, field, SIF_ASSIGNMENT_CIC, 0);
}

int sif_grid_to_density_contrast(sif_grid_t* grid) {
  if (!grid || !grid->values) {
    SIF_LOG_ERROR("grid", "grid: %s", !grid ? "NULL" : "no cell values");
    return SIF_ERR_INVALID;
  }

  /* Converting an already-converted grid computes (delta + 1) / mean - 1 over
   * a mean that is now about zero. It does not fail, it just returns a field
   * made of noise, so it is refused here rather than diagnosed later.
   * SIF_GRID_EMPTY is left alone: that is a grid the caller filled itself, and
   * only the caller knows what is in it. */
  if (grid->content == SIF_GRID_DENSITY_CONTRAST) {
    SIF_LOG_ERROR("grid",
      "grid: already a density contrast (converting again would divide by a "
      "mean of about 0)");
    return SIF_ERR_INVALID;
  }

  const uint64_t total_cells = grid->total_cells;

  /* Summed in fixed blocks and the blocks added serially, rather than with an
   * OpenMP reduction: a reduction's order follows the thread count, and the
   * mean would then differ in its last bit from one machine to the next. */
  double block_mass[GRID_HASH_BLOCKS];
  const uint64_t block = (total_cells + GRID_HASH_BLOCKS - 1) / GRID_HASH_BLOCKS;

#pragma omp parallel for schedule(static, 1)
  for (int b = 0; b < GRID_HASH_BLOCKS; b++) {
    const uint64_t start = (uint64_t)b * block;
    const uint64_t end =
      (start + block < total_cells) ? start + block : total_cells;

    double sum = 0.0;
    for (uint64_t i = start; i < end; i++)
      sum += (double)grid->values[i];
    block_mass[b] = sum;
  }

  double total_mass = 0.0;
  for (int b = 0; b < GRID_HASH_BLOCKS; b++)
    total_mass += block_mass[b];

  const double rho_mean = total_mass / (double)total_cells;

  /* The likely causes differ by sign, so the message names the one that
   * fits: zero is a grid nothing was deposited onto, negative is negative
   * weights, NaN is a non-finite value somewhere in the cells. */
  if (!(rho_mean > 0.0)) {
    SIF_LOG_ERROR("grid",
      "grid: mean density %g (expected > 0)%s", rho_mean,
      rho_mean == 0.0 ? "; nothing has been deposited onto it"
      : rho_mean < 0.0 ? "; negative weights outweigh the positive ones"
                       : "; some cell values are not finite");
    return SIF_ERR_RANGE;
  }

  const sif_real rho_mean_inv = (sif_real)(1.0 / rho_mean);

  /* compute overdensity field */
  sif_real* d_ptr = SIF_ASSUME_ALIGNED(grid->values);
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < total_cells; i++) {
    d_ptr[i] = d_ptr[i] * rho_mean_inv - 1.0f;
  }

  grid->content = SIF_GRID_DENSITY_CONTRAST;
  return SIF_OK;
}

#undef MIN
#undef GRID_CIC_VERSION
#undef GRID_HASH_BLOCKS
