/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/chain_mesh.h"

#include "structures/chain_mesh_internal.h"

#include "core/system_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int chain_mesh_create(sif_chain_mesh_t* mesh, const sif_field_t* field,
  bool consume, sif_option opt);

static inline int32_t iabs(int32_t v) { return v < 0 ? -v : v; }

/*
 * The mesh is anchored at the origin, so every coordinate has to be inside
 * [0, box_length). Anything else silently folded into a boundary cell would
 * make the binning disagree with the queries, so reject it up front.
 */
static int chain_mesh_validate_field(
  const sif_field_t* field, sif_real box_length) {

  if (!field->x || !field->y || !field->z) {
    SIF_LOG_ERROR("chain_mesh", "field has no positions");
    return SIF_ERR_INVALID;
  }

  uint64_t n_bad = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_bad)
  for (uint64_t i = 0; i < field->n_particles; i++) {
    const sif_real x = field->x[i];
    const sif_real y = field->y[i];
    const sif_real z = field->z[i];

    /* The !(a && b) form also rejects NaN, which would otherwise cast to a
     * garbage cell index. */
    if (!(x >= 0.0f && x < box_length) || !(y >= 0.0f && y < box_length) ||
        !(z >= 0.0f && z < box_length)) {
      n_bad++;
    }
  }

  if (n_bad > 0) {
    SIF_LOG_ERROR("chain_mesh",
      "%" PRIu64 " of %" PRIu64
      " particles lie outside [0, %g) and cannot be binned; wrap or shift the "
      "field into box-local coordinates first",
      n_bad, field->n_particles, (double)box_length);
    return SIF_ERR_RANGE;
  }

  return SIF_OK;
}

/*
 * Everything both constructors need before the payload columns exist: the
 * geometry, the cell table and the permutation the binning writes into. The
 * payload pointers are left NULL for the caller to either allocate or steal.
 */
static sif_chain_mesh_t* chain_mesh_new(
  uint32_t n_cells, sif_real box_length, const sif_field_t* field) {

  if (!field) {
    SIF_LOG_ERROR("chain_mesh", "cannot build a mesh from a NULL field");
    return NULL;
  }

  if (n_cells == 0 || !(box_length > 0.0f)) {
    SIF_LOG_ERROR("chain_mesh", "invalid geometry (n_cells=%u, box_length=%g)",
      n_cells, (double)box_length);
    return NULL;
  }

  if (chain_mesh_validate_field(field, box_length) != SIF_OK)
    return NULL;

  sif_chain_mesh_t* mesh = calloc(1, sizeof(sif_chain_mesh_t));
  if (!mesh)
    return NULL;

  const uint64_t n_particles = field->n_particles;

  mesh->n_cells = n_cells;
  mesh->box_length = box_length;
  mesh->total_cells = (uint64_t)n_cells * n_cells * n_cells;
  mesh->cell_length = box_length / (sif_real)n_cells;
  mesh->n_particles = n_particles;

  /* Never optional at construction. The mesh reorders particles, so an index
   * into it means nothing to a caller holding the field -- the map back is the
   * only way a query result can be used, and answering queries is what the
   * structure is for. More than that, it is what the canonicalization sorts
   * on, so a mesh cannot even be built deterministically without it.
   * SIF_MESH_DROP_INDICES releases it after that sort, not instead of it. */
  mesh->original_indices = sif_malloc_aligned(n_particles * sizeof(uint64_t));

  /* NUMA-friendly malloc (unmapped virtual memory) */
  mesh->cell_offsets =
    sif_malloc_aligned((1 + mesh->total_cells) * sizeof(uint64_t));

  if (!mesh->cell_offsets || !mesh->original_indices) {
    sif_chain_mesh_free(mesh);
    return NULL;
  }

  return mesh;
}

/* Releases the index map once it has served as the canonicalization key; see
 * SIF_MESH_DROP_INDICES. */
static void chain_mesh_apply_options(sif_chain_mesh_t* mesh, sif_option opt) {
  if (!(opt & SIF_MESH_DROP_INDICES))
    return;

  sif_free_aligned(mesh->original_indices);
  mesh->original_indices = NULL;

  SIF_LOG_TRACE("chain_mesh",
    "released the field index map (%.2f GiB); nearest-neighbour queries are "
    "unavailable on this mesh",
    (double)(mesh->n_particles * sizeof(uint64_t)) /
      (1024.0 * 1024.0 * 1024.0));
}

sif_chain_mesh_t* sif_chain_mesh_alloc(uint32_t n_cells, sif_real box_length,
  const sif_field_t* field, sif_option opt) {

  sif_chain_mesh_t* mesh = chain_mesh_new(n_cells, box_length, field);
  if (!mesh)
    return NULL;

  /* Padded so each sub-array of the block starts on a cache line, exactly as
   * the field lays its own out. */
  const uint64_t n_particles = mesh->n_particles;
  const uint64_t padded_n = sif_field_padded_n(n_particles);

  /* 1. Unified Position Block */
  mesh->_position_block = sif_malloc_aligned(3 * padded_n * sizeof(sif_real));
  if (!mesh->_position_block) {
    sif_chain_mesh_free(mesh);
    return NULL;
  }
  mesh->x = mesh->_position_block;
  mesh->y = mesh->_position_block + padded_n;
  mesh->z = mesh->_position_block + (2 * padded_n);

  /* 2. Unified Velocity Block, mirroring the field: there is nothing to copy
   * if the field carries no velocities, and nothing a caller could want the
   * mesh to hold that the field does not have. */
  if (field->vx) {
    mesh->_velocity_block = sif_malloc_aligned(3 * padded_n * sizeof(sif_real));
    if (!mesh->_velocity_block) {
      sif_chain_mesh_free(mesh);
      return NULL;
    }
    mesh->vx = mesh->_velocity_block;
    mesh->vy = mesh->_velocity_block + padded_n;
    mesh->vz = mesh->_velocity_block + (2 * padded_n);
  }

  /* 3. Independent Arrays. Weights mirror the field for the same reason. */
  if (field->weights) {
    mesh->weights = sif_malloc_aligned(n_particles * sizeof(sif_real));
    if (!mesh->weights) {
      sif_chain_mesh_free(mesh);
      return NULL;
    }
  }

  if (chain_mesh_create(mesh, field, false, opt) != SIF_OK) {
    sif_chain_mesh_free(mesh);
    return NULL;
  }

  /* chain_mesh_create() has canonicalized by now, so the sort key has done its
   * job and the mesh is in its final order either way. Dropping it here rather
   * than never allocating it is the whole point: the order this mesh is in is
   * the one the key produced. */
  chain_mesh_apply_options(mesh, opt);

  return mesh;
}

sif_chain_mesh_t* sif_chain_mesh_alloc_consume(
  uint32_t n_cells, sif_real box_length, sif_field_t* field, sif_option opt) {

  sif_chain_mesh_t* mesh = chain_mesh_new(n_cells, box_length, field);
  if (!mesh)
    return NULL;

  /*
   * The handover. Both lay their blocks out through sif_field_padded_n(), so
   * the mesh's views land on the field's sub-arrays exactly; that shared rule
   * is the reason this can be a pointer assignment rather than a copy.
   *
   * The two alias for the duration of the build, deliberately: the binning
   * passes read the particles through the field, and they read them before
   * anything is moved. Only once that is done does the field let go.
   */
  const uint64_t padded_n = sif_field_padded_n(mesh->n_particles);

  mesh->_position_block = field->_position_block;
  mesh->x = mesh->_position_block;
  mesh->y = mesh->_position_block + padded_n;
  mesh->z = mesh->_position_block + (2 * padded_n);

  if (field->vx) {
    mesh->_velocity_block = field->_velocity_block;
    mesh->vx = mesh->_velocity_block;
    mesh->vy = mesh->_velocity_block + padded_n;
    mesh->vz = mesh->_velocity_block + (2 * padded_n);
  }

  mesh->weights = field->weights;

  const int status = chain_mesh_create(mesh, field, true, opt);

  /*
   * Emptied on either outcome. By this point the storage belongs to the mesh
   * and, if the build got as far as permuting, no longer holds what the field
   * said it did -- so a field left pointing into it would be both a double
   * owner and a liar about its own contents.
   *
   * original_indices is the field's own map from a Morton sort it may have
   * had; the mesh built its own and has no use for it.
   */
  sif_free_aligned(field->original_indices);

  field->_position_block = NULL;
  field->_velocity_block = NULL;
  field->x = field->y = field->z = NULL;
  field->vx = field->vy = field->vz = NULL;
  field->weights = NULL;
  field->original_indices = NULL;
  field->n_particles = 0;
  field->state_flags = 0;

  if (status != SIF_OK) {
    sif_chain_mesh_free(mesh);
    return NULL;
  }

  SIF_LOG_TRACE("chain_mesh",
    "took over the field's payload columns (%.2f GiB); the field is now empty",
    (double)(3 * padded_n * sizeof(sif_real)) / (1024.0 * 1024.0 * 1024.0));

  chain_mesh_apply_options(mesh, opt);

  return mesh;
}

void sif_chain_mesh_free(sif_chain_mesh_t* mesh) {
  if (!mesh)
    return;

  sif_free_aligned(mesh->_position_block);
  sif_free_aligned(
    mesh->_velocity_block); /* sif_free_aligned handles NULL safely */
  sif_free_aligned(mesh->weights);
  sif_free_aligned(mesh->original_indices);
  sif_free_aligned(mesh->cell_offsets);

  free(mesh);
}

static inline uint64_t chain_mesh_flat_idx(
  const sif_field_t* field, uint64_t i, sif_real inv_l, uint32_t grid_dim) {
  int32_t ix = (int32_t)(field->x[i] * inv_l);
  int32_t iy = (int32_t)(field->y[i] * inv_l);
  int32_t iz = (int32_t)(field->z[i] * inv_l);

  /* Coordinates were validated to be in [0, box_length), so this clamp is a
   * pure backstop against the float edge case x*inv_l == grid_dim. */
  ix = (ix < 0) ? 0 : ((ix >= (int32_t)grid_dim) ? (int32_t)grid_dim - 1 : ix);
  iy = (iy < 0) ? 0 : ((iy >= (int32_t)grid_dim) ? (int32_t)grid_dim - 1 : iy);
  iz = (iz < 0) ? 0 : ((iz >= (int32_t)grid_dim) ? (int32_t)grid_dim - 1 : iz);

  return (uint64_t)ix * grid_dim * grid_dim + (uint64_t)iy * grid_dim + iz;
}

/* Exchanges two entries of the mesh, payload and all. */
static inline void chain_mesh_swap(
  sif_chain_mesh_t* mesh, uint64_t a, uint64_t b) {

#define SWAP_REAL(arr)                                                         \
  do {                                                                         \
    sif_real t = (arr)[a];                                                     \
    (arr)[a] = (arr)[b];                                                       \
    (arr)[b] = t;                                                              \
  } while (0)

  SWAP_REAL(mesh->x);
  SWAP_REAL(mesh->y);
  SWAP_REAL(mesh->z);

  if (mesh->vx) {
    SWAP_REAL(mesh->vx);
    SWAP_REAL(mesh->vy);
    SWAP_REAL(mesh->vz);
  }

  if (mesh->weights)
    SWAP_REAL(mesh->weights);

#undef SWAP_REAL

  uint64_t t = mesh->original_indices[a];
  mesh->original_indices[a] = mesh->original_indices[b];
  mesh->original_indices[b] = t;
}

/*
 * Puts every cell's particles into the order they appear in the source field.
 *
 * The scatter claims slots with an atomic, so which chunk reaches a given cell
 * first is a race: two identical runs order that cell's particles differently.
 * Everything that walks a cell inherits it -- the stacked profiles sum over one
 * in floating point, so their last bits move between runs, and a
 * nearest-neighbour query with an exact distance tie can answer differently
 * each time, which the Voronoi estimator turns into a different particle owning
 * a voxel.
 *
 * Sorting by the source index replaces that with an order determined by the
 * input alone.
 *
 * WHAT IS SORTED, AND WHY IT IS NOT THE PAYLOAD
 * --------------------------------------------
 * This used to insertion-sort the payload itself, swapping whole rows -- x, y,
 * z, the velocities, the weight and the key -- once per inversion. That is
 * linear on a cell whose slice is already nearly ordered, which is what a
 * field still in its generation order gives: consecutive particles are close
 * in space, so a cell draws from a contiguous stretch of the input and the
 * slice arrives as a few ascending runs.
 *
 * It stops being nearly ordered the moment the field is not. In an evolved
 * snapshot a particle's position no longer tracks its index, so a spatial cell
 * draws from all over the input and the slice arrives scrambled; the inversion
 * count goes from a handful to ~n^2/4, and each of those inversions moved four
 * separate multi-GiB arrays. Measured on a 2048^3 z = 0 field at 64 tracers
 * per cell, this function alone took 69 minutes, against 15 seconds for the
 * same box and mesh at z = 99.
 *
 * So the sort runs over a compact array of (key, slot) records instead, and
 * the permutation is applied to the payload once at the end: n moves per
 * column rather than one move per inversion, over 16 contiguous bytes per
 * record rather than four scattered arrays. The result is bit-identical --
 * every key in a cell is a distinct particle index, so the ascending order is
 * unique and no tie-breaking rule can differ.
 *
 * Cells at or below CHAIN_MESH_CANON_INSERTION_MAX still use insertion sort,
 * now over the key array, because at the occupancy a chain mesh is sized for
 * that is the whole population and qsort's per-comparison indirect call is
 * pure overhead there.
 */
#define CHAIN_MESH_CANON_INSERTION_MAX 32u

typedef struct {
  uint64_t key;  /* the particle's index in the source field */
  uint64_t slot; /* where it currently sits, relative to the cell's start */
} chain_mesh_canon_t;

static int chain_mesh_canon_cmp(const void* a, const void* b) {
  const uint64_t ka = ((const chain_mesh_canon_t*)a)->key;
  const uint64_t kb = ((const chain_mesh_canon_t*)b)->key;
  return (ka > kb) - (ka < kb);
}

static void chain_mesh_canonicalize(sif_chain_mesh_t* mesh) {
  const uint64_t n_c = mesh->total_cells;

#pragma omp parallel
  {
    /* Grown to the largest cell this thread meets and reused for every cell
     * after it, so the allocator is not in the inner loop. */
    chain_mesh_canon_t* order = NULL;
    sif_real* rtmp = NULL;
    uint64_t* utmp = NULL;
    uint64_t cap = 0;

#pragma omp for schedule(dynamic, 64)
    for (uint64_t c = 0; c < n_c; c++) {
      const uint64_t lo = mesh->cell_offsets[c];
      const uint64_t hi = mesh->cell_offsets[c + 1];
      const uint64_t n = hi - lo;

      if (n < 2)
        continue;

      /* Already ascending costs one scan to establish and skips everything
       * below. This is the common case for a field whose order still tracks
       * position -- initial conditions, or anything Morton sorted -- and it is
       * why those meshes were never slow to begin with. */
      uint64_t run = 1;
      while (run < n && mesh->original_indices[lo + run - 1] <
                          mesh->original_indices[lo + run])
        run++;
      if (run == n)
        continue;

      if (n > cap) {
        const uint64_t want = n + (n >> 1) + 8;
        chain_mesh_canon_t* n_order =
          sif_malloc_aligned(want * sizeof *n_order);
        sif_real* n_rtmp = sif_malloc_aligned(want * sizeof *n_rtmp);
        uint64_t* n_utmp = sif_malloc_aligned(want * sizeof *n_utmp);

        if (n_order && n_rtmp && n_utmp) {
          sif_free_aligned(order);
          sif_free_aligned(rtmp);
          sif_free_aligned(utmp);
          order = n_order;
          rtmp = n_rtmp;
          utmp = n_utmp;
          cap = want;
        } else {
          sif_free_aligned(n_order);
          sif_free_aligned(n_rtmp);
          sif_free_aligned(n_utmp);
          /* Fall back to the in-place sort, which needs no scratch at all. It
           * is the slow path this function exists to avoid, but it is correct,
           * and it keeps a mesh that cannot spare 24 bytes per tracer in one
           * cell building rather than silently unsorted. */
          for (uint64_t i = lo + 1; i < hi; i++) {
            for (uint64_t j = i; j > lo && mesh->original_indices[j] <
                                             mesh->original_indices[j - 1];
              j--) {
              chain_mesh_swap(mesh, j, j - 1);
            }
          }
          continue;
        }
      }

      for (uint64_t i = 0; i < n; i++) {
        order[i].key = mesh->original_indices[lo + i];
        order[i].slot = i;
      }

      if (n <= CHAIN_MESH_CANON_INSERTION_MAX) {
        for (uint64_t i = 1; i < n; i++) {
          const chain_mesh_canon_t t = order[i];
          uint64_t j = i;
          while (j > 0 && order[j - 1].key > t.key) {
            order[j] = order[j - 1];
            j--;
          }
          order[j] = t;
        }
      } else {
        qsort(order, (size_t)n, sizeof *order, chain_mesh_canon_cmp);
      }

/* Gather into scratch, then copy back: n reads and 2n writes per column,
 * whatever the permutation looks like. */
#define CHAIN_MESH_CANON_APPLY(arr)                                            \
  do {                                                                         \
    for (uint64_t i = 0; i < n; i++)                                           \
      rtmp[i] = (arr)[lo + order[i].slot];                                     \
    for (uint64_t i = 0; i < n; i++)                                           \
      (arr)[lo + i] = rtmp[i];                                                 \
  } while (0)

      CHAIN_MESH_CANON_APPLY(mesh->x);
      CHAIN_MESH_CANON_APPLY(mesh->y);
      CHAIN_MESH_CANON_APPLY(mesh->z);

      if (mesh->vx) {
        CHAIN_MESH_CANON_APPLY(mesh->vx);
        CHAIN_MESH_CANON_APPLY(mesh->vy);
        CHAIN_MESH_CANON_APPLY(mesh->vz);
      }

      if (mesh->weights)
        CHAIN_MESH_CANON_APPLY(mesh->weights);

#undef CHAIN_MESH_CANON_APPLY

      /* The key is its own sorted order, so it needs no gather -- but it does
       * need writing back, and it is uint64 rather than sif_real. */
      for (uint64_t i = 0; i < n; i++)
        utmp[i] = order[i].key;
      for (uint64_t i = 0; i < n; i++)
        mesh->original_indices[lo + i] = utmp[i];
    }

    sif_free_aligned(order);
    sif_free_aligned(rtmp);
    sif_free_aligned(utmp);
  }
}

/* dst[i] = src[perm[i]]. Random reads, sequential writes -- the opposite of
 * the scatter, and the better half of the trade when the two arrays overlap. */
static void chain_mesh_gather(
  sif_real* dst, const sif_real* src, const uint64_t* perm, uint64_t n) {

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++)
    dst[i] = src[perm[i]];
}

/*
 * Reorders one payload column in place, through a scratch column.
 *
 * The alternative is following the permutation's cycles, which needs no
 * scratch at all -- but a cycle walk is inherently serial per cycle and jumps
 * randomly on both sides, where this is one parallel gather and one parallel
 * copy, both of them streaming on the write side. At these particle counts the
 * bandwidth matters more than the 4 bytes per particle.
 */
static void chain_mesh_permute_column(
  sif_real* arr, sif_real* scratch, const uint64_t* perm, uint64_t n) {

  chain_mesh_gather(scratch, arr, perm, n);

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++)
    arr[i] = scratch[i];
}

/*
 * Puts every payload column into mesh order, for a mesh whose columns *are*
 * the field's.
 *
 * PASS 3 has already left original_indices holding, for each destination slot,
 * the particle that belongs there -- which is exactly the gather permutation.
 * One scratch column serves every array in turn, so the extra memory is 4
 * bytes per particle no matter how many payloads the field carries.
 */
static int chain_mesh_permute_payloads(sif_chain_mesh_t* mesh) {
  const uint64_t n_p = mesh->n_particles;

  sif_real* scratch = sif_malloc_aligned(n_p * sizeof(sif_real));
  if (!scratch) {
    SIF_LOG_ERROR("chain_mesh",
      "failed to allocate the %.2f GiB reorder column",
      (double)(n_p * sizeof(sif_real)) / (1024.0 * 1024.0 * 1024.0));
    return SIF_ERR_ALLOC;
  }

  const uint64_t* perm = mesh->original_indices;

  chain_mesh_permute_column(mesh->x, scratch, perm, n_p);
  chain_mesh_permute_column(mesh->y, scratch, perm, n_p);
  chain_mesh_permute_column(mesh->z, scratch, perm, n_p);

  if (mesh->vx) {
    chain_mesh_permute_column(mesh->vx, scratch, perm, n_p);
    chain_mesh_permute_column(mesh->vy, scratch, perm, n_p);
    chain_mesh_permute_column(mesh->vz, scratch, perm, n_p);
  }

  if (mesh->weights)
    chain_mesh_permute_column(mesh->weights, scratch, perm, n_p);

  sif_free_aligned(scratch);
  return SIF_OK;
}

/*
 * @param consume Whether the mesh's payload columns are the field's own. When
 * they are, PASS 3 cannot scatter into them -- it would overwrite particles it
 * has not read yet -- so it records the permutation only and
 * chain_mesh_permute_payloads() moves the data afterwards.
 */
/*
 * Total weight the mesh holds, summed once here so that everything downstream
 * that needs a mean density -- which is every measurement normalized to the
 * box -- gets it for free rather than walking the weights again per call.
 *
 * Accumulated in double however sif_real is configured. At float precision a
 * running sum over a few billion tracers stops moving long before it reaches
 * the end, and a total that divides a whole measurement would carry that error
 * into an amplitude that looks physical.
 */
static void chain_mesh_sum_weights(sif_chain_mesh_t* mesh) {
  if (!mesh->weights) {
    mesh->total_weight = (double)mesh->n_particles;
    return;
  }

  double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
  for (uint64_t p = 0; p < mesh->n_particles; p++)
    total += (double)mesh->weights[p];

  mesh->total_weight = total;
}

static int chain_mesh_create(sif_chain_mesh_t* mesh, const sif_field_t* field,
  bool consume, sif_option opt) {

  if (!mesh || !field)
    return SIF_ERR_INVALID;

  uint64_t n_p = field->n_particles;
  uint64_t n_c = mesh->total_cells;
  sif_real inv_l = 1.0f / mesh->cell_length;
  uint32_t grid_dim = mesh->n_cells;

  SIF_LOG_TRACE("chain_mesh", "starting mesh construction");

/* PASS 0: NUMA-Aware First Touch */
#pragma omp parallel for schedule(static)
  for (uint64_t c = 0; c <= n_c; c++) {
    mesh->cell_offsets[c] = 0;
  }

/* PASS 1: Count particles per cell (Run-Length Encoded Atomics) */
#pragma omp parallel
  {
    int tid = sif__system_thread_num();
    int num_threads = sif__system_num_threads();
    uint64_t chunk_size = n_p / num_threads;
    uint64_t start_i = tid * chunk_size;
    uint64_t end_i = (tid == num_threads - 1) ? n_p : start_i + chunk_size;

    uint64_t i = start_i;
    while (i < end_i) {
      uint64_t flat_idx = chain_mesh_flat_idx(field, i, inv_l, grid_dim);
      uint64_t run_length = 1;

      /* Look ahead to find contiguous particles belonging to the same cell */
      while (i + run_length < end_i) {
        uint64_t next_idx =
          chain_mesh_flat_idx(field, i + run_length, inv_l, grid_dim);
        if (next_idx != flat_idx)
          break;
        run_length++;
      }

/* Execute a single atomic lock for the entire block */
#pragma omp atomic
      mesh->cell_offsets[flat_idx] += run_length;

      i += run_length;
    }
  }

  /* PASS 2: Prefix Sum (Sequential, extremely fast) */
  uint64_t cumulative = 0;
  for (uint64_t c = 0; c < n_c; c++) {
    uint64_t count = mesh->cell_offsets[c];
    mesh->cell_offsets[c] = cumulative;
    cumulative += count;
  }
  mesh->cell_offsets[n_c] = cumulative;

  /* PASS 3: Parallel Scatter (RLE Block Claiming) */
  uint64_t* write_pos = sif_malloc_aligned(n_c * sizeof(uint64_t));
  if (!write_pos) {
    SIF_LOG_ERROR("chain_mesh", "OOM allocating atomic write pointers");
    return SIF_ERR_ALLOC;
  }

#pragma omp parallel for schedule(static)
  for (uint64_t c = 0; c < n_c; c++) {
    write_pos[c] = mesh->cell_offsets[c];
  }

#pragma omp parallel
  {
    int tid = sif__system_thread_num();
    int num_threads = sif__system_num_threads();
    uint64_t chunk_size = n_p / num_threads;
    uint64_t start_i = tid * chunk_size;
    uint64_t end_i = (tid == num_threads - 1) ? n_p : start_i + chunk_size;

    uint64_t i = start_i;
    while (i < end_i) {
      uint64_t flat_idx = chain_mesh_flat_idx(field, i, inv_l, grid_dim);
      uint64_t run_length = 1;

      while (i + run_length < end_i) {
        uint64_t next_idx =
          chain_mesh_flat_idx(field, i + run_length, inv_l, grid_dim);
        if (next_idx != flat_idx)
          break;
        run_length++;
      }

      uint64_t base_pos;
/* Atomically claim the entire block of write positions in one hit */
#pragma omp atomic capture
      {
        base_pos = write_pos[flat_idx];
        write_pos[flat_idx] += run_length;
      }

      /* Scatter Payload WITHOUT any atomic locks.
       *
       * The `consume` test is loop-invariant, so it costs nothing the branch
       * predictor does not already know; when it is set the payload move is
       * deferred and only the permutation is recorded here. */
      for (uint64_t j = 0; j < run_length; j++) {
        uint64_t p_idx = i + j;
        uint64_t pos = base_pos + j;

        mesh->original_indices[pos] = p_idx;

        if (consume)
          continue;

        mesh->x[pos] = field->x[p_idx];
        mesh->y[pos] = field->y[p_idx];
        mesh->z[pos] = field->z[p_idx];

        /* The mesh arrays exist exactly when the field's do, so one test
         * covers both and there is no "allocated but nothing to put in it"
         * case to fill with zeros. */
        if (mesh->weights)
          mesh->weights[pos] = field->weights[p_idx];

        if (mesh->vx) {
          mesh->vx[pos] = field->vx[p_idx];
          mesh->vy[pos] = field->vy[p_idx];
          mesh->vz[pos] = field->vz[p_idx];
        }
      }
      i += run_length;
    }
  }

  sif_free_aligned(write_pos);

  /* Deferred until the scatter has finished reading every particle. */
  if (consume && chain_mesh_permute_payloads(mesh) != SIF_OK)
    return SIF_ERR_ALLOC;

  if (opt & SIF_MESH_NO_CANONICAL)
    SIF_LOG_TRACE("chain_mesh",
      "canonical ordering skipped; cell contents are in scatter order and two "
      "identical runs may order a cell differently");
  else
    chain_mesh_canonicalize(mesh);

  chain_mesh_sum_weights(mesh);

  SIF_LOG_TRACE("chain_mesh", "RLE atomic mesh construction complete");

  return SIF_OK;
}

/* ========================================================================== */
/* --- NEAREST NEIGHBOR SPATIAL QUERIES                                       */
/* ========================================================================== */

static inline sif_real dist2_open(sif_real dx, sif_real dy, sif_real dz) {
  return dx * dx + dy * dy + dz * dz;
}

static inline sif_real point_to_cell_dist2_open(sif_real px, sif_real py,
  sif_real pz, int32_t nx, int32_t ny, int32_t nz, sif_real cell_len) {

  sif_real min_x = nx * cell_len;
  sif_real max_x = min_x + cell_len;
  sif_real min_y = ny * cell_len;
  sif_real max_y = min_y + cell_len;
  sif_real min_z = nz * cell_len;
  sif_real max_z = min_z + cell_len;

  sif_real dx = SIF_REAL_MAX(0.0f, SIF_REAL_MAX(min_x - px, px - max_x));
  sif_real dy = SIF_REAL_MAX(0.0f, SIF_REAL_MAX(min_y - py, py - max_y));
  sif_real dz = SIF_REAL_MAX(0.0f, SIF_REAL_MAX(min_z - pz, pz - max_z));

  return dx * dx + dy * dy + dz * dz;
}

static inline sif_real dist2_pbc(
  sif_real dx, sif_real dy, sif_real dz, sif_real box_len) {
  sif_real half_box = box_len * 0.5f;
  dx = SIF_REAL_ABS(dx);
  dx = (dx > half_box) ? (box_len - dx) : dx;
  dy = SIF_REAL_ABS(dy);
  dy = (dy > half_box) ? (box_len - dy) : dy;
  dz = SIF_REAL_ABS(dz);
  dz = (dz > half_box) ? (box_len - dz) : dz;
  return dx * dx + dy * dy + dz * dz;
}

static inline int32_t wrap_idx_pbc(int32_t idx, int32_t n_cells) {
  if (idx < 0)
    return idx + n_cells;
  if (idx >= n_cells)
    return idx - n_cells;
  return idx;
}

static inline sif_real point_to_cell_dist2_pbc(sif_real px, sif_real py,
  sif_real pz, int32_t nx, int32_t ny, int32_t nz, sif_real cell_len,
  sif_real box_len) {

  sif_real min_x = nx * cell_len;
  sif_real max_x = min_x + cell_len;
  sif_real min_y = ny * cell_len;
  sif_real max_y = min_y + cell_len;
  sif_real min_z = nz * cell_len;
  sif_real max_z = min_z + cell_len;

  sif_real closest_x = (px < min_x) ? min_x : ((px > max_x) ? max_x : px);
  sif_real closest_y = (py < min_y) ? min_y : ((py > max_y) ? max_y : py);
  sif_real closest_z = (pz < min_z) ? min_z : ((pz > max_z) ? max_z : pz);

  return dist2_pbc(px - closest_x, py - closest_y, pz - closest_z, box_len);
}

/* Tests every particle of one cell against the running best. */
static inline void scan_cell_open(const sif_chain_mesh_t* mesh, sif_real px,
  sif_real py, sif_real pz, int32_t nx, int32_t ny, int32_t nz,
  sif_real* min_dist2, uint64_t* best_idx) {

  const int32_t n = (int32_t)mesh->n_cells;
  if (nx < 0 || nx >= n || ny < 0 || ny >= n || nz < 0 || nz >= n)
    return;

  sif_real cell_dist2 =
    point_to_cell_dist2_open(px, py, pz, nx, ny, nz, mesh->cell_length);
  if (cell_dist2 >= *min_dist2)
    return;

  uint64_t flat_idx = (uint64_t)nx * mesh->n_cells * mesh->n_cells +
                      (uint64_t)ny * mesh->n_cells + nz;

  uint64_t start_p = mesh->cell_offsets[flat_idx];
  uint64_t end_p = mesh->cell_offsets[flat_idx + 1];

  for (uint64_t p = start_p; p < end_p; p++) {
    sif_real d2 = dist2_open(mesh->x[p] - px, mesh->y[p] - py, mesh->z[p] - pz);

    if (d2 < *min_dist2) {
      *min_dist2 = d2;
      *best_idx = p;
    }
  }
}

static inline void scan_cell_pbc(const sif_chain_mesh_t* mesh, sif_real px,
  sif_real py, sif_real pz, int32_t nx, int32_t ny, int32_t nz,
  sif_real* min_dist2, uint64_t* best_idx) {

  const int32_t n = (int32_t)mesh->n_cells;
  nx = wrap_idx_pbc(nx, n);
  ny = wrap_idx_pbc(ny, n);
  nz = wrap_idx_pbc(nz, n);

  sif_real cell_dist2 = point_to_cell_dist2_pbc(
    px, py, pz, nx, ny, nz, mesh->cell_length, mesh->box_length);
  if (cell_dist2 >= *min_dist2)
    return;

  uint64_t flat_idx = (uint64_t)nx * mesh->n_cells * mesh->n_cells +
                      (uint64_t)ny * mesh->n_cells + nz;

  uint64_t start_p = mesh->cell_offsets[flat_idx];
  uint64_t end_p = mesh->cell_offsets[flat_idx + 1];

  for (uint64_t p = start_p; p < end_p; p++) {
    sif_real d2 = dist2_pbc(
      mesh->x[p] - px, mesh->y[p] - py, mesh->z[p] - pz, mesh->box_length);

    if (d2 < *min_dist2) {
      *min_dist2 = d2;
      *best_idx = p;
    }
  }
}

/*
 * Walks the surface of the cube shell at Chebyshev radius R around
 * (cx, cy, cz). Visiting only the shell keeps this O(R^2) instead of
 * re-walking the whole O(R^3) cube and discarding the interior.
 */
#define FOREACH_SHELL_CELL(R, SCAN)                                            \
  do {                                                                         \
    for (int32_t i = -(R); i <= (R); i++) {                                    \
      for (int32_t j = -(R); j <= (R); j++) {                                  \
        if (iabs(i) == (R) || iabs(j) == (R)) {                                \
          for (int32_t k = -(R); k <= (R); k++) {                              \
            SCAN(i, j, k);                                                     \
          }                                                                    \
        } else {                                                               \
          SCAN(i, j, -(R));                                                    \
          SCAN(i, j, (R));                                                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

uint64_t sif__chain_mesh_find_nearest_slot_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || mesh->n_particles == 0)
    return UINT64_MAX;

  const int32_t n = (int32_t)mesh->n_cells;

  int32_t cx = (int32_t)(px / mesh->cell_length);
  int32_t cy = (int32_t)(py / mesh->cell_length);
  int32_t cz = (int32_t)(pz / mesh->cell_length);

  cx = (cx < 0) ? 0 : ((cx >= n) ? n - 1 : cx);
  cy = (cy < 0) ? 0 : ((cy >= n) ? n - 1 : cy);
  cz = (cz < 0) ? 0 : ((cz >= n) ? n - 1 : cz);

  sif_real min_dist2 = SIF_REAL_MAX_VAL;
  uint64_t best_idx = UINT64_MAX;

  for (int32_t R = 0; R <= n + 2; R++) {

#define SCAN_OPEN(di, dj, dk)                                                  \
  scan_cell_open(                                                              \
    mesh, px, py, pz, cx + (di), cy + (dj), cz + (dk), &min_dist2, &best_idx)

    FOREACH_SHELL_CELL(R, SCAN_OPEN);

#undef SCAN_OPEN

    if (best_idx != UINT64_MAX) {
      sif_real min_dist = SIF_REAL_SQRT(min_dist2);
      sif_real box_min_x = (cx - R) * mesh->cell_length;
      sif_real box_max_x = (cx + R + 1) * mesh->cell_length;
      sif_real box_min_y = (cy - R) * mesh->cell_length;
      sif_real box_max_y = (cy + R + 1) * mesh->cell_length;
      sif_real box_min_z = (cz - R) * mesh->cell_length;
      sif_real box_max_z = (cz + R + 1) * mesh->cell_length;

      if (px - min_dist >= box_min_x && px + min_dist <= box_max_x &&
          py - min_dist >= box_min_y && py + min_dist <= box_max_y &&
          pz - min_dist >= box_min_z && pz + min_dist <= box_max_z) {
        break;
      }
    }
  }

  return best_idx;
}

uint64_t sif__chain_mesh_find_nearest_slot_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || mesh->n_particles == 0)
    return UINT64_MAX;

  const int32_t n = (int32_t)mesh->n_cells;

  int32_t cx_raw = (int32_t)SIF_REAL_FLOOR(px / mesh->cell_length);
  int32_t cy_raw = (int32_t)SIF_REAL_FLOOR(py / mesh->cell_length);
  int32_t cz_raw = (int32_t)SIF_REAL_FLOOR(pz / mesh->cell_length);

  int32_t cx = wrap_idx_pbc(cx_raw, n);
  int32_t cy = wrap_idx_pbc(cy_raw, n);
  int32_t cz = wrap_idx_pbc(cz_raw, n);

  sif_real min_dist2 = SIF_REAL_MAX_VAL;
  uint64_t best_idx = UINT64_MAX;
  const int32_t max_R = n / 2 + 1;

  for (int32_t R = 0; R <= max_R; R++) {

#define SCAN_PBC(di, dj, dk)                                                   \
  scan_cell_pbc(                                                               \
    mesh, px, py, pz, cx + (di), cy + (dj), cz + (dk), &min_dist2, &best_idx)

    FOREACH_SHELL_CELL(R, SCAN_PBC);

#undef SCAN_PBC

    if (best_idx != UINT64_MAX) {
      sif_real min_dist = SIF_REAL_SQRT(min_dist2);

      sif_real offset_x = px - cx_raw * mesh->cell_length;
      sif_real offset_y = py - cy_raw * mesh->cell_length;
      sif_real offset_z = pz - cz_raw * mesh->cell_length;

      offset_x = SIF_REAL_FMOD(
        SIF_REAL_FMOD(offset_x, mesh->cell_length) + mesh->cell_length,
        mesh->cell_length);
      offset_y = SIF_REAL_FMOD(
        SIF_REAL_FMOD(offset_y, mesh->cell_length) + mesh->cell_length,
        mesh->cell_length);
      offset_z = SIF_REAL_FMOD(
        SIF_REAL_FMOD(offset_z, mesh->cell_length) + mesh->cell_length,
        mesh->cell_length);

      if (min_dist <= offset_x + R * mesh->cell_length &&
          min_dist <= (mesh->cell_length - offset_x) + R * mesh->cell_length &&
          min_dist <= offset_y + R * mesh->cell_length &&
          min_dist <= (mesh->cell_length - offset_y) + R * mesh->cell_length &&
          min_dist <= offset_z + R * mesh->cell_length &&
          min_dist <= (mesh->cell_length - offset_z) + R * mesh->cell_length) {
        break;
      }
    }
  }

  return best_idx;
}

/* --- public queries ---
 *
 * The shell walk above answers in mesh order because that is what the mesh can
 * always supply. These translate once, at the end, which is also where the
 * index map is required: without it there is no field index to give back, and
 * handing over a mesh slot that means nothing to the caller would be worse
 * than refusing.
 */

static uint64_t nearest_field_index(
  const sif_chain_mesh_t* mesh, uint64_t slot, const char* who) {

  if (!mesh || !mesh->original_indices) {
    SIF_LOG_ERROR("chain_mesh",
      "%s: this mesh was built with SIF_MESH_DROP_INDICES and cannot name its "
      "particles; rebuild it without that flag to query neighbours",
      who);
    return UINT64_MAX;
  }

  return (slot == UINT64_MAX) ? UINT64_MAX : mesh->original_indices[slot];
}

uint64_t sif_chain_mesh_find_nearest_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || !mesh->original_indices)
    return nearest_field_index(mesh, UINT64_MAX, "find_nearest_open");

  return nearest_field_index(mesh,
    sif__chain_mesh_find_nearest_slot_open(mesh, px, py, pz),
    "find_nearest_open");
}

uint64_t sif_chain_mesh_find_nearest_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || !mesh->original_indices)
    return nearest_field_index(mesh, UINT64_MAX, "find_nearest_pbc");

  return nearest_field_index(mesh,
    sif__chain_mesh_find_nearest_slot_pbc(mesh, px, py, pz),
    "find_nearest_pbc");
}
