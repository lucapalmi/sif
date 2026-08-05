#include "sif/structures/grid.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sif/core/settings.h"
#include "core/get_system.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/io/grid_io.h"

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline uint64_t __grid_get_flat_index(
  const sif_grid_t* grid, uint32_t ix, uint32_t iy, uint32_t iz) {
  return (uint64_t)ix * grid->n_cells * grid->n_cells +
         (uint64_t)iy * grid->n_cells + (uint64_t)iz;
}

sif_grid_t* sif_grid_alloc(uint32_t n_cells, real_t box_length) {

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
  grid->p2_shift = grid->p2_mask ? (uint32_t)__builtin_ctz(n_cells) : 0;

  grid->delta = sif_calloc_aligned(grid->total_cells, sizeof(real_t));
  if (!grid->delta) {
    SIF_LOG_ERROR("grid", "failed to allocate delta field");
    free(grid);
    return NULL;
  }

  SIF_LOG_TRACE("grid", "cubic grid initialized");
  return grid;
}

void sif_grid_free(sif_grid_t* grid) {
  if (!grid)
    return;

  /* delta may legitimately be NULL: a finder can have taken the buffer. The
   * struct still has to be released. */
  sif_free_aligned(grid->delta);
  free(grid);

  SIF_LOG_TRACE("grid", "cubic grid destroyed");
}

// CIC CONSTRUCTION

typedef struct {
  real_t* delta;
  uint32_t ix_start;
  uint32_t ix_end;
  uint32_t width;
  uint32_t n_cells;
} grid_slab_t;

PURE_FUNCTION static inline uint64_t __grid_slab_get_flat_index(
  const grid_slab_t* slab, uint32_t local_ix, uint32_t iy, uint32_t iz) {

  return (uint64_t)local_ix * slab->n_cells * slab->n_cells +
         (uint64_t)iy * slab->n_cells + (uint64_t)iz;
}

static grid_slab_t* __grid_slabs_alloc(uint32_t n_cells, int n_threads) {
  grid_slab_t* slabs = malloc(n_threads * sizeof(grid_slab_t));
  if (!slabs)
    return NULL;

  uint64_t total_slab_cells = 0;
  uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);

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

  /* 2. The Arena: One massive contiguous memory block */
  real_t* arena = sif_calloc_aligned(total_slab_cells, sizeof(real_t));
  if (!arena) {
    free(slabs);
    return NULL;
  }

  /* 3. Hand out specific pointers */
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

    slabs[t].delta = arena + current_offset;
    current_offset += slab_cells;
  }

  return slabs;
}

static void __grid_slabs_free(grid_slab_t* slabs, int n_threads) {
  if (slabs) {
    if (n_threads > 0 && slabs[0].delta) {
      sif_free_aligned(slabs[0].delta);
    }
    free(slabs);
  }
}

static void __grid_cic_reduce_ghost_cells(grid_slab_t* slabs, int n_threads) {
  for (int t = 0; t < n_threads; t++) {
    grid_slab_t* slab = &slabs[t];
    grid_slab_t* slab_next = &slabs[(t + 1) % n_threads];
    const uint32_t N = slab->n_cells;

    for (uint32_t iy = 0; iy < N; iy++) {
      for (uint32_t iz = 0; iz < N; iz++) {
        uint64_t ghost_idx =
          __grid_slab_get_flat_index(slab, slab->width, iy, iz);
        uint64_t target_idx = __grid_slab_get_flat_index(slab_next, 0, iy, iz);
        slab_next->delta[target_idx] += slab->delta[ghost_idx];
      }
    }
  }
}

static void __grid_cic_merge_slabs_to_grid(sif_grid_t* grid,
  const grid_slab_t* slabs, int n_threads, real_t inv_cell_volume) {

#pragma omp parallel for schedule(static)
  for (int t = 0; t < n_threads; t++) {
    const grid_slab_t* slab = &slabs[t];
    const uint32_t N = grid->n_cells;

    for (uint32_t local_ix = 0; local_ix < slab->width; local_ix++) {
      uint32_t global_ix = slab->ix_start + local_ix;
      for (uint32_t iy = 0; iy < N; iy++) {
        for (uint32_t iz = 0; iz < N; iz++) {
          uint64_t src = __grid_slab_get_flat_index(slab, local_ix, iy, iz);
          uint64_t dst = __grid_get_flat_index(grid, global_ix, iy, iz);
          grid->delta[dst] = slab->delta[src] * inv_cell_volume;
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
static int __grid_validate_positions(const real_t* xs, const real_t* ys,
  const real_t* zs, uint64_t n_particles, real_t box_length) {

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
static inline uint32_t __cic_cell_x(real_t x, real_t inv_cell, uint32_t N) {
  uint32_t ix = (uint32_t)(x * inv_cell);
  return (ix >= N) ? 0u : ix;
}

static void __grid_compute_cic(sif_grid_t* grid, const real_t* xs,
  const real_t* ys, const real_t* zs, const real_t* masses,
  uint64_t n_particles) {

  const uint32_t N = grid->n_cells;
  const real_t idx = (real_t)N / (real_t)grid->box_length;

  int32_t n_slabs = system_get_max_threads();
  n_slabs = ((int)N < n_slabs) ? (int)N : n_slabs;

  grid_slab_t* slabs = __grid_slabs_alloc(N, n_slabs);
  if (!slabs) {
    SIF_LOG_WARNING("grid_cic", "aborting particle assignment");
    return;
  }

  /* Decouple logic partitioning from actual thread counts to prevent data
   * corruption */
  const int n_chunks = 256;

  /* pass 1: count particles in each bin */

  uint64_t* counts = calloc(n_chunks * n_slabs, sizeof(uint64_t));
  if (!counts) {
    SIF_LOG_ERROR("grid_cic", "failed to allocate counts array");
    SIF_LOG_WARNING("grid_cic", "aborting particle assignment");
    __grid_slabs_free(slabs, n_slabs);
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
      uint32_t ix = __cic_cell_x(xs[p], idx, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));
      counts[c * n_slabs + target_slab]++;
    }
  }

  /* pass 2: transform counts into offsets */

  uint64_t current_offset = 0;

  for (int32_t slab = 0; slab < n_slabs; slab++) {
    for (int c = 0; c < n_chunks; c++) {
      uint64_t flat_idx = c * n_slabs + slab;
      uint64_t count = counts[flat_idx];
      counts[flat_idx] = current_offset;
      current_offset += count;
    }
  }

  /* pass 3: scatter*/

  uint64_t* sorted_indices = malloc(n_particles * sizeof(uint64_t));
  if (!sorted_indices) {
    SIF_LOG_ERROR("grid_cic", "failed to allocate index array");
    free(counts);
    __grid_slabs_free(slabs, n_slabs);
    return;
  }

#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < n_chunks; c++) {
    uint64_t start = c * chunk_size;
    uint64_t end =
      (start + chunk_size > n_particles) ? n_particles : start + chunk_size;

    for (uint64_t p = start; p < end; p++) {
      uint32_t ix = __cic_cell_x(xs[p], idx, N);
      uint32_t target_slab = MIN(ix / base_width, (uint32_t)(n_slabs - 1));

      uint64_t flat_idx = c * n_slabs + target_slab;
      uint64_t write_idx = counts[flat_idx];

      sorted_indices[write_idx] = p;

      counts[flat_idx]++;
    }
  }

  /* pass 4: assignment*/
#pragma omp parallel for schedule(static, 1)
  for (int32_t slab_idx = 0; slab_idx < n_slabs; slab_idx++) {

    uint64_t slab_start =
      (slab_idx == 0) ? 0 : counts[(n_chunks - 1) * n_slabs + (slab_idx - 1)];
    uint64_t slab_end = counts[(n_chunks - 1) * n_slabs + slab_idx];

    grid_slab_t* slab = &slabs[slab_idx];
    const uint32_t x0 = slab->ix_start;

    real_t* s_delta = SIF_ASSUME_ALIGNED(slab->delta);

    for (uint64_t p = slab_start; p < slab_end; p++) {
      /* Indirect read: incredibly fast because xs, ys, zs are Morton-sorted! */
      uint64_t orig_p = sorted_indices[p];

      const real_t cx = xs[orig_p] * idx;
      const real_t cy = ys[orig_p] * idx;
      const real_t cz = zs[orig_p] * idx;

      /* Branchless bounds safety */
      uint32_t raw_cx = (uint32_t)cx, raw_cy = (uint32_t)cy,
               raw_cz = (uint32_t)cz;

      uint32_t ix = (raw_cx >= N) ? 0 : raw_cx;
      uint32_t iy = (raw_cy >= N) ? 0 : raw_cy;
      uint32_t iz = (raw_cz >= N) ? 0 : raw_cz;

      uint32_t iy1 = (iy + 1 == N) ? 0 : iy + 1;
      uint32_t iz1 = (iz + 1 == N) ? 0 : iz + 1;

      const uint32_t lix = ix - x0;
      const uint32_t lix1 = lix + 1;

      const real_t tx = cx - (real_t)raw_cx;
      const real_t ty = cy - (real_t)raw_cy;
      const real_t tz = cz - (real_t)raw_cz;

      const real_t wx0 = 1.0f - tx, wx1 = tx;
      const real_t wy0 = 1.0f - ty, wy1 = ty;
      const real_t wz0 = 1.0f - tz, wz1 = tz;

      /* ALGEBRAIC REDUCTION: Drop from 24 multiplications to 14 */
      const real_t w00 = wx0 * wy0;
      const real_t w01 = wx0 * wy1;
      const real_t w10 = wx1 * wy0;
      const real_t w11 = wx1 * wy1;

      /* Per-particle mass when the field carries one, unit mass otherwise. */
      const real_t particle_mass = masses ? masses[orig_p] : 1.0f;

      const real_t mz0 = particle_mass * wz0;
      const real_t mz1 = particle_mass * wz1;

      /* 8 writes, exactly 1 multiplication each */
      s_delta[__grid_slab_get_flat_index(slab, lix, iy, iz)] += w00 * mz0;
      s_delta[__grid_slab_get_flat_index(slab, lix, iy, iz1)] += w00 * mz1;
      s_delta[__grid_slab_get_flat_index(slab, lix, iy1, iz)] += w01 * mz0;
      s_delta[__grid_slab_get_flat_index(slab, lix, iy1, iz1)] += w01 * mz1;
      s_delta[__grid_slab_get_flat_index(slab, lix1, iy, iz)] += w10 * mz0;
      s_delta[__grid_slab_get_flat_index(slab, lix1, iy, iz1)] += w10 * mz1;
      s_delta[__grid_slab_get_flat_index(slab, lix1, iy1, iz)] += w11 * mz0;
      s_delta[__grid_slab_get_flat_index(slab, lix1, iy1, iz1)] += w11 * mz1;
    }
  }

  __grid_cic_reduce_ghost_cells(slabs, n_slabs);

  const real_t inv_cell_vol =
    1.0f / (real_t)(grid->cell_length * grid->cell_length * grid->cell_length);
  __grid_cic_merge_slabs_to_grid(grid, slabs, n_slabs, inv_cell_vol);

  free(sorted_indices);
  free(counts);
  __grid_slabs_free(slabs, n_slabs);

  SIF_LOG_TRACE("grid", "cic assignment completed");
}

/*
 * Whether the on-disk .xgrid cache is active. Off by default: a single 2048^3
 * grid is ~34 GB, and the cache is never evicted, so opting in has to be a
 * deliberate act.
 *
 *   sif_setting_set("grid_cache_enabled", "1");
 */
static int __grid_cache_enabled(void) {
  const char* v = sif_setting_get("grid_cache_enabled", "0");
  if (!v)
    return 0;
  return (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' ||
          v[0] == 'Y');
}

/*
 * Cache key. NOTE: this samples only 5 particles, so it identifies a field by
 * its shape plus a spot check rather than by its full contents. Two distinct
 * fields agreeing on n_particles, box, mass mode and those 5 positions would
 * collide. Good enough for "same run, same data" reuse, not a content hash.
 */
static uint64_t __generate_field_hash(
  const sif_grid_t* grid, const sif_field_t* field, uint8_t has_masses) {
  uint64_t hash = 0xcbf29ce484222325ULL;   // FNV offset basis
  const uint64_t prime = 0x100000001b3ULL; // FNV prime

#define HASH_VAL(val, type)                                                    \
  do {                                                                         \
    type v = (val);                                                            \
    unsigned char* p = (unsigned char*)&v;                                     \
    for (size_t i = 0; i < sizeof(type); i++) {                                \
      hash ^= p[i];                                                            \
      hash *= prime;                                                           \
    }                                                                          \
  } while (0)

  HASH_VAL(grid->n_cells, uint32_t);
  HASH_VAL(grid->box_length, real_t);
  HASH_VAL(field->n_particles, uint64_t);
  HASH_VAL(has_masses, uint8_t);

  if (field->n_particles > 0) {
    uint64_t indices[5] = {0, field->n_particles / 4, field->n_particles / 2,
      (field->n_particles * 3) / 4, field->n_particles - 1};
    for (int i = 0; i < 5; i++) {
      uint64_t idx = indices[i];
      HASH_VAL(field->x[idx], real_t);
      HASH_VAL(field->y[idx], real_t);
      HASH_VAL(field->z[idx], real_t);
    }
  }
#undef HASH_VAL
  return hash;
}

void sif_grid_assign_cic(sif_grid_t* grid, sif_field_t* field) {
  if (!grid || !field || !grid->delta) {
    SIF_LOG_ERROR("grid_cic", "invalid grid or field");
    return;
  }

  if (__grid_validate_positions(field->x, field->y, field->z,
        field->n_particles, grid->box_length) != SIF_OK) {
    return;
  }

  const int use_cache = __grid_cache_enabled();

  char cache_dir[512] = {0};
  char final_path[1024] = {0};
  uint64_t hash = 0;

  if (use_cache) {
    hash = __generate_field_hash(grid, field, field->masses ? 1u : 0u);

    const char* cache_dir_setting = sif_setting_get("cache_directory", NULL);
    if (cache_dir_setting) {
      snprintf(cache_dir, sizeof(cache_dir), "%s", cache_dir_setting);
    } else {
      const char* home = getenv("HOME");
      if (home) {
        snprintf(cache_dir, sizeof(cache_dir), "%s/.sif/grid_cache", home);
      } else {
        snprintf(cache_dir, sizeof(cache_dir), "/tmp/.sif/grid_cache");
      }
    }

    snprintf(final_path, sizeof(final_path), "%s/grid_%llx.xgrid", cache_dir,
      (unsigned long long)hash);

    if (sif_grid_read_into(final_path, grid) == 0) {
      SIF_LOG_INFO("grid_cic", "loaded cached .xgrid from %s", final_path);
      return;
    }

    SIF_LOG_TRACE("grid_cic", "grid is not cached. starting assignment");
  }

  __grid_compute_cic(
    grid, field->x, field->y, field->z, field->masses, field->n_particles);

  if (!use_cache) {
    SIF_LOG_FLUSH();
    return;
  }

  /* Atomic write: rename() past a concurrent job cannot leave a torn file. */
  char tmp_path[1024];
  snprintf(tmp_path, sizeof(tmp_path), "%s/grid_%llx.tmp.%d", cache_dir,
    (unsigned long long)hash, getpid());

  if (sif_grid_write(tmp_path, grid) == 0) {
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

void sif_grid_compute_overdensity(sif_grid_t* grid) {
  if (!grid || !grid->delta || grid->total_cells == 0) {
    SIF_LOG_ERROR("grid", "invalid or empty grid");
    return;
  }

  const uint64_t total_cells = grid->total_cells;

  double total_mass = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total_mass)
  for (uint64_t i = 0; i < total_cells; i++) {
    total_mass += (double)grid->delta[i];
  }

  const double rho_mean = total_mass / (double)total_cells;

  if (!(rho_mean > 0.0)) {
    SIF_LOG_ERROR("grid",
      "mean density is %g, cannot normalize to an overdensity field",
      rho_mean);
    return;
  }

  const real_t rho_mean_inv = (real_t)(1.0 / rho_mean);

  /* compute overdensity field */
  real_t* d_ptr = SIF_ASSUME_ALIGNED(grid->delta);
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < total_cells; i++) {
    d_ptr[i] = d_ptr[i] * rho_mean_inv - 1.0f;
  }
}
