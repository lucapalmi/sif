/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/grid.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/system_internal.h"
#include "io/internal.h"
#include "sif/core/settings.h"
#include "sif/io/grid_io.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
  if (n_cells == 0 || !(box_length > (sif_real)0.0)) {
    SIF_LOG_ERROR("grid",
      "invalid geometry (n_cells=%u, box_length=%g); both must be positive",
      n_cells, (double)box_length);
    return NULL;
  }

  sif_grid_t* grid = malloc(sizeof(sif_grid_t));
  if (!grid) {
    SIF_LOG_ERROR(
      "grid", "failed to allocate grid struct (%zu bytes)", sizeof(sif_grid_t));
    return NULL;
  }

  grid->n_cells = n_cells;
  grid->total_cells = (uint64_t)n_cells * n_cells * n_cells;
  grid->box_length = box_length;
  grid->cell_length = box_length / n_cells;

  grid->p2_mask = ((n_cells & (n_cells - 1)) == 0) ? (n_cells - 1) : 0;
  grid->p2_shift = grid->p2_mask ? SIF_CTZ_U32(n_cells) : 0;

  grid->content = SIF_GRID_EMPTY;
  grid->values = sif_calloc_aligned(grid->total_cells, sizeof(sif_real));
  if (!grid->values) {
    SIF_LOG_ERROR("grid", "failed to allocate the cell array");
    free(grid);
    return NULL;
  }

  SIF_LOG_TRACE("grid", "cubic grid initialized");
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
 * and writes it with no synchronization at all. What crosses a boundary is
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
  /** Planes this slab owns; values holds width + 1, the last being the ghost.
   */
  uint32_t width;
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
static grid_slab_t* grid_slabs_alloc(uint32_t n_cells, int n_threads) {
  grid_slab_t* slabs = malloc(n_threads * sizeof(grid_slab_t));
  if (!slabs)
    return NULL;

  uint64_t total_slab_cells = 0;
  uint64_t align_elements = SIF_CACHE_LINE / sizeof(sif_real);

  for (int t = 0; t < n_threads; t++) {
    uint32_t ix_start = (uint32_t)t * (n_cells / n_threads);
    uint32_t ix_end = (t == n_threads - 1)
                        ? n_cells
                        : (uint32_t)(t + 1) * (n_cells / n_threads);
    uint32_t width = ix_end - ix_start;

    uint64_t slab_cells = (uint64_t)(width + 1) * n_cells * n_cells;

    slab_cells = (slab_cells + align_elements - 1) & ~(align_elements - 1);
    total_slab_cells += slab_cells;
  }

  /* Zeroed, because the deposit loop accumulates into it. */
  sif_real* arena = sif_calloc_aligned(total_slab_cells, sizeof(sif_real));
  if (!arena) {
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

    uint64_t slab_cells = (uint64_t)(slabs[t].width + 1) * n_cells * n_cells;
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
  for (int t = 0; t < n_threads; t++) {
    grid_slab_t* slab = &slabs[t];
    grid_slab_t* slab_next = &slabs[(t + 1) % n_threads];
    const uint32_t N = slab->n_cells;

    for (uint32_t iy = 0; iy < N; iy++) {
      for (uint32_t iz = 0; iz < N; iz++) {
        uint64_t ghost_idx = grid_slab_flat_index(slab, slab->width, iy, iz);
        uint64_t target_idx = grid_slab_flat_index(slab_next, 0, iy, iz);
        slab_next->values[target_idx] += slab->values[ghost_idx];
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
          uint64_t src = grid_slab_flat_index(slab, local_ix, iy, iz);
          uint64_t dst = grid_flat_index(grid, global_ix, iy, iz);
          grid->values[dst] = slab->values[src] * inv_cell_volume;
        }
      }
    }
  }
}

/*
 * The CIC scatter derives a particle's slab twice, once when counting and once
 * when depositing, and the two derivations must agree. They only do for
 * coordinates inside the box, so reject anything else up front rather than
 * letting the deposit pass underflow its local slab index.
 */
static int grid_validate_positions(const sif_real* xs, const sif_real* ys,
  const sif_real* zs, uint64_t n_particles, sif_real box_length) {

  uint64_t n_bad = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_bad)
  for (uint64_t p = 0; p < n_particles; p++) {
    if (!(xs[p] >= 0.0f && xs[p] < box_length) ||
        !(ys[p] >= 0.0f && ys[p] < box_length) ||
        !(zs[p] >= 0.0f && zs[p] < box_length)) {
      n_bad++;
    }
  }

  if (n_bad > 0) {
    SIF_LOG_ERROR("grid_cic",
      "%" PRIu64 " of %" PRIu64
      " particles lie outside [0, %g) and cannot be assigned; wrap or shift "
      "the field into box-local coordinates first",
      n_bad, n_particles, (double)box_length);
    return SIF_ERR_RANGE;
  }

  return SIF_OK;
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

static void grid_compute_cic(sif_grid_t* grid, const sif_real* xs,
  const sif_real* ys, const sif_real* zs, const sif_real* weights,
  uint64_t n_particles) {

  const uint32_t N = grid->n_cells;
  const sif_real idx = (sif_real)N / (sif_real)grid->box_length;

  int32_t n_slabs = sif__system_max_threads();
  n_slabs = ((int)N < n_slabs) ? (int)N : n_slabs;

  grid_slab_t* slabs = grid_slabs_alloc(N, n_slabs);
  if (!slabs) {
    SIF_LOG_WARNING("grid_cic", "aborting particle assignment");
    return;
  }

  /*
   * The particles are bucketed by destination slab before anything is
   * deposited, in a counting sort across four passes: count, prefix-sum,
   * scatter, deposit. Sorting first is what lets pass 4 hand each thread one
   * contiguous run of particles that all land in the slab it owns -- without
   * it, a thread would have to test every particle against its own range and
   * skip most of them.
   *
   * The chunk count is fixed rather than taken from the thread count, so the
   * bucket layout -- and therefore the order weight is accumulated in -- is a
   * property of the input alone. A grid that changed in its last bits with the
   * number of cores would make the cache keyed on it worthless, and would put
   * a thread count into every reproducibility claim downstream.
   */
  const int n_chunks = 256;

  /* pass 1: how many particles each (chunk, slab) pair owns. Each chunk writes
   * its own row, so the counts need no synchronization. */

  uint64_t* counts = calloc(n_chunks * n_slabs, sizeof(uint64_t));
  if (!counts) {
    SIF_LOG_ERROR("grid_cic", "failed to allocate counts array");
    SIF_LOG_WARNING("grid_cic", "aborting particle assignment");
    grid_slabs_free(slabs, n_slabs);
    return;
  }

  uint32_t base_width = N / n_slabs;
  uint64_t chunk_size = (n_particles + n_chunks - 1) / n_chunks;

#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < n_chunks; c++) {
    uint64_t start = c * chunk_size;
    uint64_t end =
      (start + chunk_size > n_particles) ? n_particles : start + chunk_size;

    for (uint64_t p = start; p < end; p++) {
      uint32_t ix = cic_cell_x(xs[p], idx, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));
      counts[c * n_slabs + target_slab]++;
    }
  }

  /* pass 2: turn the counts into write offsets, slab-major -- every chunk's
   * share of slab s is laid out consecutively, so slab s ends up owning one
   * unbroken run of the sorted array. That is what pass 4 slices on. Serial
   * because a prefix sum is, and it is n_chunks * n_slabs entries. */

  uint64_t current_offset = 0;

  for (int32_t slab = 0; slab < n_slabs; slab++) {
    for (int c = 0; c < n_chunks; c++) {
      uint64_t flat_idx = c * n_slabs + slab;
      uint64_t count = counts[flat_idx];
      counts[flat_idx] = current_offset;
      current_offset += count;
    }
  }

  /* pass 3: scatter the particle indices into their slab's run. Each chunk
   * advances only its own offsets, so again no synchronization -- and the
   * order within a slab is the order the chunks were laid out in, not the
   * order the threads happened to finish.
   *
   * Indices rather than copies of the coordinates: 8 bytes per particle
   * instead of 24 (or 32 with weights), at the price of the indirect read in
   * pass 4. */

  uint64_t* sorted_indices = malloc(n_particles * sizeof(uint64_t));
  if (!sorted_indices) {
    SIF_LOG_ERROR("grid_cic", "failed to allocate index array");
    free(counts);
    grid_slabs_free(slabs, n_slabs);
    return;
  }

#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < n_chunks; c++) {
    uint64_t start = c * chunk_size;
    uint64_t end =
      (start + chunk_size > n_particles) ? n_particles : start + chunk_size;

    for (uint64_t p = start; p < end; p++) {
      uint32_t ix = cic_cell_x(xs[p], idx, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));

      uint64_t flat_idx = c * n_slabs + target_slab;
      uint64_t write_idx = counts[flat_idx];

      sorted_indices[write_idx] = p;

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
      (slab_idx == 0) ? 0 : counts[(n_chunks - 1) * n_slabs + (slab_idx - 1)];
    uint64_t slab_end = counts[(n_chunks - 1) * n_slabs + slab_idx];

    grid_slab_t* slab = &slabs[slab_idx];
    const uint32_t x0 = slab->ix_start;

    sif_real* s_values = SIF_ASSUME_ALIGNED(slab->values);

    for (uint64_t p = slab_start; p < slab_end; p++) {
      /* The indirection costs less than it looks: consecutive entries in a
       * slab's run come from the same chunk, so the reads walk the coordinate
       * arrays in roughly increasing order rather than jumping about. */
      uint64_t orig_p = sorted_indices[p];

      const sif_real cx = xs[orig_p] * idx;
      const sif_real cy = ys[orig_p] * idx;
      const sif_real cz = zs[orig_p] * idx;

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

  grid_cic_reduce_ghost_cells(slabs, n_slabs);

  const sif_real inv_cell_vol =
    1.0f /
    (sif_real)(grid->cell_length * grid->cell_length * grid->cell_length);
  grid_cic_merge_slabs_to_grid(grid, slabs, n_slabs, inv_cell_vol);

  free(sorted_indices);
  free(counts);
  grid_slabs_free(slabs, n_slabs);

  SIF_LOG_TRACE("grid", "cic assignment completed");
}

/*
 * Whether the on-disk .xgrid cache is active. Off by default: a single 2048^3
 * grid is ~34 GB, and the cache is never evicted, so opting in has to be a
 * deliberate act.
 *
 *   sif_setting_set("grid_cache_enabled", "1");
 */
static int grid_cache_enabled(void) {
  const char* v = sif_setting_get("grid_cache_enabled", "0");
  if (!v)
    return 0;
  return (
    v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
}

/*
 * Version of the CIC deposition itself.
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
#define GRID_CIC_VERSION 1u

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

/* One round of the splitmix64 finalizer, which is what utils/random.h already
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
 * arrays are padded to a cache line and that padding is uninitialized, so
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
static grid_key_t grid_cache_key(
  const sif_grid_t* grid, const sif_field_t* field) {

  grid_key_t key = {0xcbf29ce484222325ULL, 0x9e3779b97f4a7c15ULL};

  key_mix(&key, GRID_CIC_VERSION);
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

void sif_grid_assign_cic(sif_grid_t* grid, const sif_field_t* field) {
  if (!grid || !field || !grid->values) {
    SIF_LOG_ERROR("grid_cic", "invalid grid or field");
    return;
  }

  if (grid_validate_positions(field->x, field->y, field->z, field->n_particles,
        grid->box_length) != SIF_OK) {
    return;
  }

  const int use_cache = grid_cache_enabled();

  char final_path[1024] = {0};
  grid_key_t key = {0, 0};

  if (use_cache) {
    /* One streaming pass over the field. It is paid only when the cache is
     * enabled, and against the assignment below -- which reads the same arrays
     * and then scatters eight read-modify-writes per particle across a grid
     * far larger than cache -- it is a small fraction of what a hit saves. */
    key = grid_cache_key(grid, field);

    /* sif_init() puts cache_directory in the settings table and creates it, so
     * this is the one source of truth; re-deriving $HOME/.sif/grid_cache here
     * would be a second copy of that default, free to drift from the first. */
    const char* cache_dir = sif_setting_get("cache_directory", NULL);
    if (!cache_dir) {
      SIF_LOG_WARNING("grid_cic",
        "the grid cache is enabled but cache_directory is unset; computing "
        "without it");
    } else {
      snprintf(final_path, sizeof(final_path), "%s/grid_%016llx%016llx.xgrid",
        cache_dir, (unsigned long long)key.b, (unsigned long long)key.a);

      /* Keyed, so a file found by name still has to say it came from this
       * field before its contents are believed. */
      const uint64_t expect[2] = {key.a, key.b};
      if (sif__grid_read_into_keyed(final_path, grid, expect) == SIF_OK) {
        SIF_LOG_INFO("grid_cic", "loaded cached .xgrid from %s", final_path);
        return;
      }

      SIF_LOG_TRACE("grid_cic", "grid is not cached. starting assignment");
    }
  }

  grid_compute_cic(
    grid, field->x, field->y, field->z, field->weights, field->n_particles);
  grid->content = SIF_GRID_DENSITY;

  /* final_path stays empty when the cache is off, or on when the directory
   * could not be resolved -- both mean there is nothing to write. */
  if (!use_cache || final_path[0] == '\0') {
    SIF_LOG_FLUSH();
    return;
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
      SIF_LOG_TRACE("grid_cic", "cached computed grid to .xgrid file");
    } else {
      remove(tmp_path);
    }
  } else {
    SIF_LOG_WARNING("grid_cic", "could not write .xgrid to cache directory.");
    remove(tmp_path);
  }

  SIF_LOG_FLUSH();
}

void sif_grid_to_density_contrast(sif_grid_t* grid) {
  if (!grid || !grid->values || grid->total_cells == 0) {
    SIF_LOG_ERROR("grid", "invalid or empty grid");
    return;
  }

  /* Converting an already-converted grid computes (delta + 1) / mean - 1 over
   * a mean that is now about zero. It does not fail, it just returns a field
   * made of noise, so it is refused here rather than diagnosed later.
   * SIF_GRID_EMPTY is left alone: that is a grid the caller filled itself, and
   * only the caller knows what is in it. */
  if (grid->content == SIF_GRID_DENSITY_CONTRAST) {
    SIF_LOG_ERROR("grid", "this grid already holds a density contrast");
    return;
  }

  const uint64_t total_cells = grid->total_cells;

  double total_mass = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total_mass)
  for (uint64_t i = 0; i < total_cells; i++) {
    total_mass += (double)grid->values[i];
  }

  const double rho_mean = total_mass / (double)total_cells;

  if (!(rho_mean > 0.0)) {
    SIF_LOG_ERROR("grid",
      "mean density is %g, cannot normalize to an overdensity field", rho_mean);
    return;
  }

  const sif_real rho_mean_inv = (sif_real)(1.0 / rho_mean);

  /* compute overdensity field */
  sif_real* d_ptr = SIF_ASSUME_ALIGNED(grid->values);
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < total_cells; i++) {
    d_ptr[i] = d_ptr[i] * rho_mean_inv - 1.0f;
  }

  grid->content = SIF_GRID_DENSITY_CONTRAST;
}

#undef MIN
#undef GRID_CIC_VERSION
#undef GRID_HASH_BLOCKS
