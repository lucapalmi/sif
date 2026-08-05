#include "sif/structures/tessellation.h"
#include "core/get_system.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* --- Edge Tracking Helpers --- */

typedef struct {
  uint64_t p1;
  uint64_t p2;
} sif_edge_t;

static int __sif_edge_cmp(const void* a, const void* b) {
  const sif_edge_t* ea = (const sif_edge_t*)a;
  const sif_edge_t* eb = (const sif_edge_t*)b;
  if (ea->p1 < eb->p1)
    return -1;
  if (ea->p1 > eb->p1)
    return 1;
  if (ea->p2 < eb->p2)
    return -1;
  if (ea->p2 > eb->p2)
    return 1;
  return 0;
}

static void __sif_compute_voxel_layer(const sif_field_t* field,
  const sif_chain_mesh_t* mesh, uint64_t* layer_out, real_t* volumes,
  int32_t iz, int32_t grid_dim, real_t voxel_len, real_t vol_per_voxel,
  bool is_pbc) {

#pragma omp parallel for schedule(static)
  for (int32_t ix = 0; ix < grid_dim; ix++) {
    for (int32_t iy = 0; iy < grid_dim; iy++) {
      /* The chain mesh is anchored at the origin, so sample in box-local
       * coordinates. */
      real_t px = (ix + 0.5f) * voxel_len;
      real_t py = (iy + 0.5f) * voxel_len;
      real_t pz = (iz + 0.5f) * voxel_len;

      uint64_t nearest_idx =
        is_pbc ? sif_chain_mesh_find_nearest_pbc(mesh, px, py, pz)
               : sif_chain_mesh_find_nearest_open(mesh, px, py, pz);

      layer_out[ix * grid_dim + iy] = nearest_idx;

      if (nearest_idx != UINT64_MAX) {
#pragma omp atomic
        volumes[nearest_idx] += vol_per_voxel;
      }
    }
  }
}

/* ========================================================================== */
/* --- PUBLIC API                                                             */
/* ========================================================================== */

NODISCARD sif_tessellation_t* sif_tessellation_build_approx(
  const sif_field_t* field, uint32_t supersample_factor, sif_option_t opt) {

  if (!field || field->n_particles == 0 || supersample_factor == 0) {
    SIF_LOG_ERROR("tessellation", "Invalid field or zero supersampling.");
    return NULL;
  }

  sif_tessellation_t* tess = malloc(sizeof(sif_tessellation_t));
  if (!tess)
    return NULL;

  tess->num_particles = field->n_particles;
  tess->num_edges = 0;
  tess->neighbor_offsets = NULL;
  tess->neighbor_indices = NULL;

  uint64_t align_elements = __SIF_CACHE_LINE / sizeof(real_t);
  uint64_t padded_n =
    (field->n_particles + align_elements - 1) & ~(align_elements - 1);

  tess->volumes = sif_malloc_aligned(padded_n * sizeof(real_t));
  if (!tess->volumes) {
    free(tess);
    return NULL;
  }

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < padded_n; i++) {
    tess->volumes[i] = 0.0f;
  }

  // Auto-tune grid resolution to roughly 1 cell per particle mean-spacing
  uint32_t mesh_cells =
    (uint32_t)REAL_CEIL(REAL_POW(field->n_particles, 1.0 / 3.0));
  real_t box_len = field->max_p[0] - field->min_p[0];

  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(
    mesh_cells, box_len, field, false, false, true);
  if (!mesh) {
    SIF_LOG_ERROR("tessellation", "Failed to allocate chain mesh.");
    sif_free_aligned(tess->volumes);
    free(tess);
    return NULL;
  }

  /* --- DISPATCH BRANCHES --- */

  if ((opt & __SIF_TESS_METHOD_MASK) == SIF_TESS_METHOD_RANDOM) {
    SIF_LOG_INFO("tessellation", "Building random approximation (Vol only).");

    uint64_t total_samples = field->n_particles * supersample_factor;
    real_t total_volume = box_len * box_len * box_len;
    real_t vol_per_sample = total_volume / (real_t)total_samples;

    uint64_t base_seed = 0x9E3779B97F4A7C15ULL;

#pragma omp parallel
    {
      sif_prng_state_t rng_state;
      sif_prng_init(&rng_state, base_seed ^ (uint64_t)system_get_thread_num());

      bool is_pbc = ((opt & __SIF_PBC_MASK) == SIF_PBC_PERIODIC);

#pragma omp for schedule(guided)
      for (uint64_t i = 0; i < total_samples; i++) {

        /* Box-local: the chain mesh is anchored at the origin. */
        real_t px = sif_prng_next_real(&rng_state) * box_len;
        real_t py = sif_prng_next_real(&rng_state) * box_len;
        real_t pz = sif_prng_next_real(&rng_state) * box_len;

        uint64_t nearest_idx;
        if (is_pbc) {
          nearest_idx = sif_chain_mesh_find_nearest_pbc(mesh, px, py, pz);
        } else {
          nearest_idx = sif_chain_mesh_find_nearest_open(mesh, px, py, pz);
        }

        if (nearest_idx != UINT64_MAX) {
#pragma omp atomic
          tess->volumes[nearest_idx] += vol_per_sample;
        }
      }
    }
  } else if ((opt & __SIF_TESS_METHOD_MASK) == SIF_TESS_METHOD_VOXEL) {
    SIF_LOG_INFO("tessellation", "Building voxel approximation (Vol + Graph).");

    // 1. Grid Dimensions
    uint64_t total_target_voxels = field->n_particles * supersample_factor;
    int32_t grid_dim =
      (int32_t)REAL_CEIL(REAL_POW(total_target_voxels, 1.0 / 3.0));
    real_t voxel_len = box_len / (real_t)grid_dim;
    real_t vol_per_voxel = voxel_len * voxel_len * voxel_len;

    bool is_pbc = ((opt & __SIF_PBC_MASK) == SIF_PBC_PERIODIC);

    // 2. Sliding Window Buffers
    uint64_t elements_per_layer = (uint64_t)grid_dim * grid_dim;
    uint64_t* layer_zero =
      sif_malloc_aligned(elements_per_layer * sizeof(uint64_t));
    uint64_t* layer_current =
      sif_malloc_aligned(elements_per_layer * sizeof(uint64_t));
    uint64_t* layer_next =
      sif_malloc_aligned(elements_per_layer * sizeof(uint64_t));

    // 3. Raw Edge Storage (Bulletproof capacity: 3 edges per actual voxel)
    uint64_t actual_voxels = (uint64_t)grid_dim * grid_dim * grid_dim;
    uint64_t max_raw_edges = actual_voxels * 3;
    sif_edge_t* raw_edges =
      sif_malloc_aligned(max_raw_edges * sizeof(sif_edge_t));
    uint64_t raw_edge_count = 0;

    if (!layer_zero || !layer_current || !layer_next || !raw_edges) {
      SIF_LOG_ERROR(
        "tessellation", "OOM allocating sliding window or edge buffers.");
      sif_free_aligned(layer_zero);
      sif_free_aligned(layer_current);
      sif_free_aligned(layer_next);
      sif_free_aligned(raw_edges);
      sif_chain_mesh_free(mesh);
      sif_free_aligned(tess->volumes);
      free(tess);
      return NULL;
    }

    // Initialize Z=0
    __sif_compute_voxel_layer(field, mesh, layer_zero, tess->volumes, 0,
      grid_dim, voxel_len, vol_per_voxel, is_pbc);
    memcpy(layer_current, layer_zero, elements_per_layer * sizeof(uint64_t));

    // --- PHASE A & B: SLIDING WINDOW SWEEP ---
    for (int32_t iz = 0; iz < grid_dim; iz++) {

      // Prepare the NEXT layer
      if (iz < grid_dim - 1) {
        __sif_compute_voxel_layer(field, mesh, layer_next, tess->volumes,
          iz + 1, grid_dim, voxel_len, vol_per_voxel, is_pbc);
      } else if (is_pbc) {
        // Periodic wrap: DO NOT re-add volumes, just copy the IDs from
        // layer_zero
        memcpy(layer_next, layer_zero, elements_per_layer * sizeof(uint64_t));
      } else {
        // Open wrap: out of bounds
        for (uint64_t i = 0; i < elements_per_layer; i++)
          layer_next[i] = UINT64_MAX;
      }

// Scan edges in the CURRENT layer
#pragma omp parallel for schedule(static)
      for (int32_t ix = 0; ix < grid_dim; ix++) {
        for (int32_t iy = 0; iy < grid_dim; iy++) {
          uint64_t owner = layer_current[ix * grid_dim + iy];
          if (owner == UINT64_MAX)
            continue;

          // Check +X Neighbor
          uint64_t owner_x = UINT64_MAX;
          if (ix < grid_dim - 1)
            owner_x = layer_current[(ix + 1) * grid_dim + iy];
          else if (is_pbc)
            owner_x = layer_current[0 * grid_dim + iy];

          if (owner_x != UINT64_MAX && owner != owner_x) {
            uint64_t idx;
#pragma omp atomic capture
            idx = raw_edge_count++;
            if (idx < max_raw_edges) {
              raw_edges[idx].p1 = (owner < owner_x) ? owner : owner_x;
              raw_edges[idx].p2 = (owner > owner_x) ? owner : owner_x;
            }
          }

          // Check +Y Neighbor
          uint64_t owner_y = UINT64_MAX;
          if (iy < grid_dim - 1)
            owner_y = layer_current[ix * grid_dim + (iy + 1)];
          else if (is_pbc)
            owner_y = layer_current[ix * grid_dim + 0];

          if (owner_y != UINT64_MAX && owner != owner_y) {
            uint64_t idx;
#pragma omp atomic capture
            idx = raw_edge_count++;
            if (idx < max_raw_edges) {
              raw_edges[idx].p1 = (owner < owner_y) ? owner : owner_y;
              raw_edges[idx].p2 = (owner > owner_y) ? owner : owner_y;
            }
          }

          // Check +Z Neighbor (Look ahead to layer_next)
          uint64_t owner_z = layer_next[ix * grid_dim + iy];
          if (owner_z != UINT64_MAX && owner != owner_z) {
            uint64_t idx;
#pragma omp atomic capture
            idx = raw_edge_count++;
            if (idx < max_raw_edges) {
              raw_edges[idx].p1 = (owner < owner_z) ? owner : owner_z;
              raw_edges[idx].p2 = (owner > owner_z) ? owner : owner_z;
            }
          }
        }
      }

      // Swap the layer buffers (Avoid deep copies!)
      uint64_t* temp = layer_current;
      layer_current = layer_next;
      layer_next = temp;
    }

    if (raw_edge_count >= max_raw_edges) {
      SIF_LOG_WARNING("tessellation",
        "Raw edges exceeded max capacity! Graph may be truncated.");
      raw_edge_count = max_raw_edges;
    }

    // --- PHASE C: CSR COMPACTION ---
    SIF_LOG_INFO("tessellation", "Compacting %llu raw edges into CSR graph...",
      raw_edge_count);

    // 1. Sort the raw edges to bring duplicates together
    qsort(raw_edges, raw_edge_count, sizeof(sif_edge_t), __sif_edge_cmp);

    // 2. Count unique undirected edges and allocate offsets
    tess->neighbor_offsets =
      sif_malloc_aligned((tess->num_particles + 1) * sizeof(uint64_t));
    memset(
      tess->neighbor_offsets, 0, (tess->num_particles + 1) * sizeof(uint64_t));

    uint64_t unique_count = 0;
    if (raw_edge_count > 0) {
      tess->neighbor_offsets[raw_edges[0].p1 + 1]++;
      tess->neighbor_offsets[raw_edges[0].p2 + 1]++;
      unique_count++;

      for (uint64_t i = 1; i < raw_edge_count; i++) {
        if (raw_edges[i].p1 != raw_edges[i - 1].p1 ||
            raw_edges[i].p2 != raw_edges[i - 1].p2) {
          tess->neighbor_offsets[raw_edges[i].p1 + 1]++;
          tess->neighbor_offsets[raw_edges[i].p2 + 1]++;
          unique_count++;
        }
      }
    }

    // Directed edges for CSR = Undirected connections * 2
    tess->num_edges = unique_count * 2;

    // 3. Prefix sum the offsets
    for (uint64_t i = 0; i < tess->num_particles; i++) {
      tess->neighbor_offsets[i + 1] += tess->neighbor_offsets[i];
    }

    // 4. Populate indices
    tess->neighbor_indices =
      sif_malloc_aligned(tess->num_edges * sizeof(uint64_t));
    uint64_t* insert_ptrs =
      malloc((tess->num_particles + 1) * sizeof(uint64_t));
    memcpy(insert_ptrs, tess->neighbor_offsets,
      (tess->num_particles + 1) * sizeof(uint64_t));

    if (raw_edge_count > 0) {
      tess->neighbor_indices[insert_ptrs[raw_edges[0].p1]++] = raw_edges[0].p2;
      tess->neighbor_indices[insert_ptrs[raw_edges[0].p2]++] = raw_edges[0].p1;

      for (uint64_t i = 1; i < raw_edge_count; i++) {
        if (raw_edges[i].p1 != raw_edges[i - 1].p1 ||
            raw_edges[i].p2 != raw_edges[i - 1].p2) {
          tess->neighbor_indices[insert_ptrs[raw_edges[i].p1]++] =
            raw_edges[i].p2;
          tess->neighbor_indices[insert_ptrs[raw_edges[i].p2]++] =
            raw_edges[i].p1;
        }
      }
    }

    // Cleanup
    free(insert_ptrs);
    sif_free_aligned(raw_edges);
    sif_free_aligned(layer_zero);
    sif_free_aligned(layer_current);
    sif_free_aligned(layer_next);

    SIF_LOG_INFO(
      "tessellation", "Generated %llu unique CSR edges.", tess->num_edges);

  } else {

    SIF_LOG_INFO("tessellation", "invalid argument");
  }

  sif_chain_mesh_free(mesh);
  return tess;
}

void sif_tessellation_free(sif_tessellation_t* tess) {
  if (!tess)
    return;

  // sif_free_aligned safely handles NULL pointers, so we can pass the CSR
  // arrays directly even if the RANDOM branch left them unallocated.
  sif_free_aligned(tess->volumes);
  sif_free_aligned(tess->neighbor_offsets);
  sif_free_aligned(tess->neighbor_indices);

  free(tess);
}
