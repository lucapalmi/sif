/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/chain_mesh.h"

#include "core/system_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int chain_mesh_create(sif_chain_mesh_t* mesh, const sif_field_t* field);

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

sif_chain_mesh_t* sif_chain_mesh_alloc(uint32_t n_cells, sif_real box_length,
  const sif_field_t* field, bool allocate_masses, bool allocate_velocities,
  bool allocate_original_indices) {

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

  sif_chain_mesh_t* mesh = malloc(sizeof(sif_chain_mesh_t));
  if (!mesh)
    return NULL;

  uint64_t n_particles = field->n_particles;

  mesh->n_cells = n_cells;
  mesh->box_length = box_length;
  mesh->total_cells = (uint64_t)n_cells * n_cells * n_cells;
  mesh->cell_length = box_length / (sif_real)n_cells;
  mesh->n_particles = n_particles;

  /* Padded so each sub-array of the block starts on a cache line, exactly as
   * the field lays its own out. */
  uint64_t padded_n = sif_field_padded_n(n_particles);

  /* 1. Unified Position Block */
  mesh->_position_block = sif_malloc_aligned(3 * padded_n * sizeof(sif_real));
  if (!mesh->_position_block) {
    free(mesh);
    return NULL;
  }
  mesh->x = mesh->_position_block;
  mesh->y = mesh->_position_block + padded_n;
  mesh->z = mesh->_position_block + (2 * padded_n);

  /* 2. Unified Velocity Block (Optional) */
  if (allocate_velocities) {
    mesh->_velocity_block = sif_malloc_aligned(3 * padded_n * sizeof(sif_real));
    if (!mesh->_velocity_block) {
      sif_free_aligned(mesh->_position_block);
      free(mesh);
      return NULL;
    }
    mesh->vx = mesh->_velocity_block;
    mesh->vy = mesh->_velocity_block + padded_n;
    mesh->vz = mesh->_velocity_block + (2 * padded_n);
  } else {
    mesh->_velocity_block = NULL;
    mesh->vx = NULL;
    mesh->vy = NULL;
    mesh->vz = NULL;
  }

  /* 3. Independent Arrays */
  mesh->masses =
    allocate_masses ? sif_malloc_aligned(n_particles * sizeof(sif_real)) : NULL;
  mesh->original_indices =
    allocate_original_indices
      ? sif_malloc_aligned(n_particles * sizeof(uint64_t))
      : NULL;

  /* NUMA-friendly malloc (unmapped virtual memory) */
  mesh->cell_offsets =
    sif_malloc_aligned((1 + mesh->total_cells) * sizeof(uint64_t));

  /* Final safety check for the independent arrays */
  if (!mesh->cell_offsets || (allocate_masses && !mesh->masses) ||
      (allocate_original_indices && !mesh->original_indices)) {
    sif_chain_mesh_free(mesh); /* We can safely call the free function now */
    return NULL;
  }

  if (chain_mesh_create(mesh, field) != SIF_OK) {
    sif_chain_mesh_free(mesh);
    return NULL;
  }

  return mesh;
}

void sif_chain_mesh_free(sif_chain_mesh_t* mesh) {
  if (!mesh)
    return;

  sif_free_aligned(mesh->_position_block);
  sif_free_aligned(
    mesh->_velocity_block); /* sif_free_aligned handles NULL safely */
  sif_free_aligned(mesh->masses);
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

static int chain_mesh_create(sif_chain_mesh_t* mesh, const sif_field_t* field) {

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

      /* Scatter Payload WITHOUT any atomic locks */
      for (uint64_t j = 0; j < run_length; j++) {
        uint64_t p_idx = i + j;
        uint64_t pos = base_pos + j;

        mesh->x[pos] = field->x[p_idx];
        mesh->y[pos] = field->y[p_idx];
        mesh->z[pos] = field->z[p_idx];
        if (mesh->original_indices) {
          mesh->original_indices[pos] = p_idx;
        }

        if (mesh->masses) {
          mesh->masses[pos] = field->masses ? field->masses[p_idx] : 1.0f;
        }

        if (mesh->vx) {
          if (field->vx) {
            mesh->vx[pos] = field->vx[p_idx];
            mesh->vy[pos] = field->vy[p_idx];
            mesh->vz[pos] = field->vz[p_idx];
          } else {
            mesh->vx[pos] = 0.0f;
            mesh->vy[pos] = 0.0f;
            mesh->vz[pos] = 0.0f;
          }
        }
      }
      i += run_length;
    }
  }

  sif_free_aligned(write_pos);
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
      *best_idx = mesh->original_indices[p];
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
      *best_idx = mesh->original_indices[p];
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

uint64_t sif_chain_mesh_find_nearest_open(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || !mesh->original_indices || mesh->n_particles == 0)
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

uint64_t sif_chain_mesh_find_nearest_pbc(
  const sif_chain_mesh_t* mesh, sif_real px, sif_real py, sif_real pz) {

  if (!mesh || !mesh->original_indices || mesh->n_particles == 0)
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
