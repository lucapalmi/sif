#include "sif/measure/profiles.h"

#include "sif/core/macros.h"
#include "core/get_system.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#define __SIF_WRAP_PBC(val, min_val, box_len)                                  \
  (REAL_FMOD(REAL_FMOD((val) - (min_val), (box_len)) + (box_len), (box_len)) + \
    (min_val))

/* --- Memory Allocation --- */

NODISCARD static sif_density_profiles_t* sif_density_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, real_t ext) {

  sif_density_profiles_t* profs = malloc(sizeof(sif_density_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;

  profs->r_edges = sif_malloc_aligned((n_bins + 1) * sizeof(real_t));
  profs->profiles = sif_calloc_aligned(n_voids * n_bins, sizeof(real_t));

  if (!profs->r_edges || !profs->profiles) {
    sif_density_profiles_free(profs);
    return NULL;
  }
  return profs;
}

NODISCARD static sif_velocity_profiles_t* sif_velocity_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, real_t ext) {

  sif_velocity_profiles_t* profs = malloc(sizeof(sif_velocity_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;

  profs->r_edges = sif_malloc_aligned((n_bins + 1) * sizeof(real_t));
  profs->v_rad = sif_calloc_aligned(n_voids * n_bins, sizeof(real_t));

  if (!profs->r_edges || !profs->v_rad) {
    sif_velocity_profiles_free(profs);
    return NULL;
  }
  return profs;
}

void sif_density_profiles_free(sif_density_profiles_t* profs) {
  if (!profs)
    return;
  sif_free_aligned(profs->r_edges);
  sif_free_aligned(profs->profiles);
  free(profs);
}

void sif_velocity_profiles_free(sif_velocity_profiles_t* profs) {
  if (!profs)
    return;
  sif_free_aligned(profs->r_edges);
  sif_free_aligned(profs->v_rad);
  free(profs);
}

/* --- Master Compute Routine --- */

void sif_profiles_mesh(const sif_catalog_t* cat,
  const sif_field_t* field, real_t box_length, real_t ext, uint32_t n_bins,
  sif_option_t opt, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel) {

  if (!cat || !field || cat->n_voids == 0) {
    SIF_LOG_ERROR("profiles", "Invalid inputs to sif_compute_profiles");
    return;
  }

  /* Determine what needs to be computed */
  int compute_dens = (out_dens != NULL);
  int compute_vel = (out_vel != NULL);

  if (!compute_dens && !compute_vel) {
    SIF_LOG_WARNING("profiles",
      "Both density and velocity outputs are NULL. Nothing to compute.");
    return;
  }

  SIF_LOG_INFO(
    "profiles", "Computing profiles for %" PRIu64 " voids", cat->n_voids);

  /* Conditionally allocate only if the user passed a pointer to a NULL pointer
   */
  if (compute_dens && *out_dens == NULL) {
    *out_dens = sif_density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return;
    }
  }

  if (compute_vel && *out_vel == NULL) {
    *out_vel = sif_velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      if (compute_dens && out_dens) {
        sif_density_profiles_free(*out_dens);
        *out_dens = NULL;
      }
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      return;
    }
  }

  real_t bin_width = ext / (real_t)n_bins;
  real_t* sphere_vols = NULL;

  /* Setup radial edges and full SPHERE volumes */
  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(real_t));
    for (uint32_t j = 0; j < n_bins; j++) {
      real_t r_inner = (real_t)j * bin_width;
      real_t r_outer = (real_t)(j + 1) * bin_width;
      (*out_dens)->r_edges[j] = r_inner;
      /* Calculate the volume of the entire sphere up to r_outer */
      sphere_vols[j] = (4.0 / 3.0) * M_PI * (r_outer * r_outer * r_outer);
    }
    (*out_dens)->r_edges[n_bins] = ext;
  }

  if (compute_vel) {
    for (uint32_t j = 0; j <= n_bins; j++) {
      (*out_vel)->r_edges[j] = (real_t)j * bin_width;
    }
  }

  /* Determine optimal mesh resolution */
  real_t avg_radius = 0.0;
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    avg_radius += cat->radii[i];
  }
  avg_radius /= (real_t)cat->n_voids;

  /* avg_radius or ext can legitimately be zero for a degenerate catalog; the
   * division would then produce inf and an undefined cast. */
  uint32_t n_cells = 1;
  if (avg_radius > 0.0f && ext > 0.0f) {
    real_t target = box_length / (avg_radius * ext);
    if (target >= 256.0f)
      n_cells = 256;
    else if (target >= 1.0f)
      n_cells = (uint32_t)target;
  }

  SIF_LOG_TRACE("profiles", "Building chain mesh with %ux%ux%u cells", n_cells,
    n_cells, n_cells);

  /* Build the mesh */
  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(n_cells, box_length,
    field, compute_dens, compute_vel, true);

  if (!mesh) {
    SIF_LOG_ERROR("profiles",
      "failed to build the chain mesh (are all particles inside the box?)");
    sif_free_aligned(sphere_vols);
    if (compute_dens && *out_dens) {
      sif_density_profiles_free(*out_dens);
      *out_dens = NULL;
    }
    if (compute_vel && *out_vel) {
      sif_velocity_profiles_free(*out_vel);
      *out_vel = NULL;
    }
    return;
  }

  real_t half_box = box_length * 0.5;
  real_t inv_cell_len = 1.0 / mesh->cell_length;

  /* Calculate Mean Density for Overdensity Normalization */
  real_t mean_dens = 1.0;
  if (compute_dens) {
    real_t box_vol = box_length * box_length * box_length;
    if (field->masses) {
      real_t total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
      for (uint64_t p = 0; p < field->n_particles; p++) {
        total_mass += field->masses[p];
      }
      mean_dens = total_mass / box_vol;
    } else {
      mean_dens = (real_t)field->n_particles / box_vol;
    }
  }

  int is_periodic = ((opt & __SIF_PBC_MASK) == SIF_PBC_PERIODIC);

  const real_t* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const real_t* my = SIF_ASSUME_ALIGNED(mesh->y);
  const real_t* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const real_t* mm = mesh->masses ? SIF_ASSUME_ALIGNED(mesh->masses) : NULL;
  const real_t* mvx = mesh->vx ? SIF_ASSUME_ALIGNED(mesh->vx) : NULL;
  const real_t* mvy = mesh->vy ? SIF_ASSUME_ALIGNED(mesh->vy) : NULL;
  const real_t* mvz = mesh->vz ? SIF_ASSUME_ALIGNED(mesh->vz) : NULL;

  /* Per-thread bin scratch, allocated once on the heap. These used to be VLAs
   * inside the parallel region, which overflows an OpenMP thread stack (far
   * smaller than the main stack) for a large caller-supplied n_bins. */
  const int n_threads = sif_system_get_max_threads() > 0 ? sif_system_get_max_threads() : 1;
  real_t* scratch_mass = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(real_t));
  real_t* scratch_vrad = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(real_t));
  uint64_t* scratch_count = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(uint64_t));

  if (!scratch_mass || !scratch_vrad || !scratch_count) {
    SIF_LOG_ERROR("profiles", "OOM allocating the per-thread bin scratch");
    sif_free_aligned(scratch_mass);
    sif_free_aligned(scratch_vrad);
    sif_free_aligned(scratch_count);
    sif_free_aligned(sphere_vols);
    sif_chain_mesh_free(mesh);
    return;
  }

/* --- Core Compute Loop --- */
#pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    real_t cx = cat->cx[i];
    real_t cy = cat->cy[i];
    real_t cz = cat->cz[i];
    real_t rv = cat->radii[i];

    real_t r_max = rv * ext;
    real_t r_max_sq = r_max * r_max;

    const int tid = sif_system_get_thread_num();
    real_t* local_mass = scratch_mass + (size_t)tid * n_bins;
    real_t* local_vrad = scratch_vrad + (size_t)tid * n_bins;
    uint64_t* local_count = scratch_count + (size_t)tid * n_bins;

    if (compute_dens)
      memset(local_mass, 0, n_bins * sizeof(real_t));
    if (compute_vel) {
      memset(local_vrad, 0, n_bins * sizeof(real_t));
      memset(local_count, 0, n_bins * sizeof(uint64_t));
    }

    /* Bounding box indices */
    int32_t ix_min = (int32_t)floor((cx - r_max) * inv_cell_len);
    int32_t ix_max = (int32_t)floor((cx + r_max) * inv_cell_len);
    int32_t iy_min = (int32_t)floor((cy - r_max) * inv_cell_len);
    int32_t iy_max = (int32_t)floor((cy + r_max) * inv_cell_len);
    int32_t iz_min = (int32_t)floor((cz - r_max) * inv_cell_len);
    int32_t iz_max = (int32_t)floor((cz + r_max) * inv_cell_len);

    for (int32_t ix = ix_min; ix <= ix_max; ix++) {
      int32_t w_ix = ix;
      if (is_periodic) {
        w_ix %= (int32_t)n_cells;
        if (w_ix < 0)
          w_ix += n_cells;
      } else if (w_ix < 0 || w_ix >= (int32_t)n_cells) {
        continue; /* Open boundary: prune cells outside the box */
      }

      for (int32_t iy = iy_min; iy <= iy_max; iy++) {
        int32_t w_iy = iy;
        if (is_periodic) {
          w_iy %= (int32_t)n_cells;
          if (w_iy < 0)
            w_iy += n_cells;
        } else if (w_iy < 0 || w_iy >= (int32_t)n_cells) {
          continue; /* Open boundary: prune cells outside the box */
        }

        for (int32_t iz = iz_min; iz <= iz_max; iz++) {
          int32_t w_iz = iz;
          if (is_periodic) {
            w_iz %= (int32_t)n_cells;
            if (w_iz < 0)
              w_iz += n_cells;
          } else if (w_iz < 0 || w_iz >= (int32_t)n_cells) {
            continue; /* Open boundary: prune cells outside the box */
          }

          uint64_t flat_idx = (uint64_t)w_ix * n_cells * n_cells +
                              (uint64_t)w_iy * n_cells + w_iz;

          uint64_t p_start = mesh->cell_offsets[flat_idx];
          uint64_t p_end = mesh->cell_offsets[flat_idx + 1];

          if (is_periodic) {
            /* NO simd pragma here: `bin` below is data-dependent, so the
             * accumulations are a scatter. Asserting the loop is
             * dependence-free let the vectorizer drop updates whenever two
             * lanes landed in the same bin, biasing every profile low by up
             * to ~20% in dense bins. The compiler still vectorizes the
             * distance arithmetic on its own. */
            for (uint64_t p = p_start; p < p_end; p++) {
              real_t dx = mx[p] - cx;
              real_t dy = my[p] - cy;
              real_t dz = mz[p] - cz;

              /* Branchless Periodic Boundary Wrap */
              dx -= box_length * (real_t)(dx > half_box) -
                    box_length * (real_t)(dx < -half_box);
              dy -= box_length * (real_t)(dy > half_box) -
                    box_length * (real_t)(dy < -half_box);
              dz -= box_length * (real_t)(dz > half_box) -
                    box_length * (real_t)(dz < -half_box);

              real_t r2 = dx * dx + dy * dy + dz * dz;

              /* Mask evaluation - compilers handle this efficiently in SIMD */
              if (r2 <= r_max_sq) {
                real_t r = REAL_SQRT(r2);
                uint32_t bin = (uint32_t)(r / (rv * bin_width));

                if (bin < n_bins) {
                  if (compute_dens) {
                    real_t m = mm ? mm[p] : 1.0f;
                    local_mass[bin] += m;
                  }

                  if (compute_vel && mvx) {
                    real_t inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                    real_t v_r =
                      (mvx[p] * dx + mvy[p] * dy + mvz[p] * dz) *
                      inv_r;
                    local_vrad[bin] += v_r;
                    local_count[bin]++;
                  }
                }
              }
            }
          } else {
            /* NO simd pragma here: `bin` below is data-dependent, so the
             * accumulations are a scatter. Asserting the loop is
             * dependence-free let the vectorizer drop updates whenever two
             * lanes landed in the same bin, biasing every profile low by up
             * to ~20% in dense bins. The compiler still vectorizes the
             * distance arithmetic on its own. Open boundary: no wrapping. */
            for (uint64_t p = p_start; p < p_end; p++) {
              real_t dx = mx[p] - cx;
              real_t dy = my[p] - cy;
              real_t dz = mz[p] - cz;

              real_t r2 = dx * dx + dy * dy + dz * dz;

              if (r2 <= r_max_sq) {
                real_t r = REAL_SQRT(r2);
                uint32_t bin = (uint32_t)(r / (rv * bin_width));

                if (bin < n_bins) {
                  if (compute_dens) {
                    real_t m = mm ? mm[p] : 1.0f;
                    local_mass[bin] += m;
                  }

                  if (compute_vel && mvx) {
                    real_t inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                    real_t v_r =
                      (mvx[p] * dx + mvy[p] * dy + mvz[p] * dz) *
                      inv_r;
                    local_vrad[bin] += v_r;
                    local_count[bin]++;
                  }
                }
              }
            }
          }
        }
      }
    }

    /* Finalize the void's profile */
    real_t rv_cubed = rv * rv * rv;
    real_t cumulative_mass = 0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        /* Add the current shell's mass to the running total */
        cumulative_mass += local_mass[j];

        /* Divide total enclosed mass by total spherical volume */
        real_t raw_rho = cumulative_mass / (sphere_vols[j] * rv_cubed);
        (*out_dens)->profiles[global_idx] = (raw_rho / mean_dens) - 1.0;
      }

      if (compute_vel) {
        /* Velocity is typically left as differential (average of the shell) */
        if (local_count[j] > 0) {
          (*out_vel)->v_rad[global_idx] =
            local_vrad[j] / (real_t)local_count[j];
        } else {
          (*out_vel)->v_rad[global_idx] = 0.0;
        }
      }
    }
  }

  sif_free_aligned(scratch_mass);
  sif_free_aligned(scratch_vrad);
  sif_free_aligned(scratch_count);
  sif_free_aligned(sphere_vols);
  sif_chain_mesh_free(mesh);
  SIF_LOG_INFO("profiles", "Profile computation completed");
}

void sif_profiles_voronoi(const sif_catalog_t* cat,
  const sif_field_t* field, const sif_tessellation_t* tess, real_t box_length,
  real_t ext, uint32_t n_bins, sif_option_t opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (!cat || !field || !tess || cat->n_voids == 0) {
    SIF_LOG_ERROR("profiles", "Invalid inputs to sif_profiles_voronoi");
    return;
  }

  int compute_dens = (out_dens != NULL);
  int compute_vel = (out_vel != NULL);

  if (!compute_dens && !compute_vel)
    return;

  SIF_LOG_INFO("profiles",
    "Computing volume-weighted Voronoi profiles for %" PRIu64 " voids",
    cat->n_voids);

  if (compute_dens && *out_dens == NULL) {
    *out_dens = sif_density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return;
    }
  }
  if (compute_vel && *out_vel == NULL) {
    *out_vel = sif_velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      if (compute_dens && *out_dens) {
        sif_density_profiles_free(*out_dens);
        *out_dens = NULL;
      }
      return;
    }
  }

  /* Sphere volumes setup */
  real_t bin_width = ext / (real_t)n_bins;
  real_t* sphere_vols = NULL;

  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(real_t));
    for (uint32_t j = 0; j < n_bins; j++) {
      real_t r_outer = (real_t)(j + 1) * bin_width;
      (*out_dens)->r_edges[j] = (real_t)j * bin_width;
      sphere_vols[j] = (4.0 / 3.0) * M_PI * (r_outer * r_outer * r_outer);
    }
    (*out_dens)->r_edges[n_bins] = ext;
  }

  if (compute_vel) {
    for (uint32_t j = 0; j <= n_bins; j++) {
      (*out_vel)->r_edges[j] = (real_t)j * bin_width;
    }
  }

  /* Chain mesh allocation (Fast Spatial Index) */
  real_t avg_radius = 0.0;
  for (uint64_t i = 0; i < cat->n_voids; i++)
    avg_radius += cat->radii[i];
  avg_radius /= (real_t)cat->n_voids;

  /* avg_radius or ext can legitimately be zero for a degenerate catalog; the
   * division would then produce inf and an undefined cast. */
  uint32_t n_cells = 1;
  if (avg_radius > 0.0f && ext > 0.0f) {
    real_t target = box_length / (avg_radius * ext);
    if (target >= 256.0f)
      n_cells = 256;
    else if (target >= 1.0f)
      n_cells = (uint32_t)target;
  }

  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(
    n_cells, box_length, field, false, false, true);

  if (!mesh) {
    SIF_LOG_ERROR("profiles",
      "failed to build the chain mesh (are all particles inside the box?)");
    sif_free_aligned(sphere_vols);
    if (compute_dens && *out_dens) {
      sif_density_profiles_free(*out_dens);
      *out_dens = NULL;
    }
    if (compute_vel && *out_vel) {
      sif_velocity_profiles_free(*out_vel);
      *out_vel = NULL;
    }
    return;
  }

  /* Calculate Mean Density */
  real_t mean_dens = 1.0;
  if (compute_dens) {
    real_t box_vol = box_length * box_length * box_length;
    if (field->masses) {
      real_t total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
      for (uint64_t p = 0; p < field->n_particles; p++)
        total_mass += field->masses[p];
      mean_dens = total_mass / box_vol;
    } else {
      mean_dens = (real_t)field->n_particles / box_vol;
    }
  }

  int is_periodic = ((opt & __SIF_PBC_MASK) == SIF_PBC_PERIODIC);

  const real_t* fm = field->masses ? SIF_ASSUME_ALIGNED(field->masses) : NULL;
  const real_t* fvx = field->vx ? SIF_ASSUME_ALIGNED(field->vx) : NULL;
  const real_t* fvy = field->vy ? SIF_ASSUME_ALIGNED(field->vy) : NULL;
  const real_t* fvz = field->vz ? SIF_ASSUME_ALIGNED(field->vz) : NULL;

  /* Per-thread bin scratch, see the note in sif_profiles_mesh. */
  const int n_threads = sif_system_get_max_threads() > 0 ? sif_system_get_max_threads() : 1;
  real_t* scratch_mass = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(real_t));
  real_t* scratch_vrad = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(real_t));
  real_t* scratch_vol = sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(real_t));

  if (!scratch_mass || !scratch_vrad || !scratch_vol) {
    SIF_LOG_ERROR("profiles", "OOM allocating the per-thread bin scratch");
    sif_free_aligned(scratch_mass);
    sif_free_aligned(scratch_vrad);
    sif_free_aligned(scratch_vol);
    sif_free_aligned(sphere_vols);
    sif_chain_mesh_free(mesh);
    return;
  }

/* --- Core Sub-Grid Compute Loop --- */
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    real_t cx = cat->cx[i];
    real_t cy = cat->cy[i];
    real_t cz = cat->cz[i];
    real_t rv = cat->radii[i];

    real_t r_max = rv * ext;
    real_t r_max_sq = r_max * r_max;

    /* Local Voxel Grid Definition */
    real_t voxel_len = (2.0f * r_max) / (real_t)__SIF_PROFILE_GRID_DIM;
    real_t vol_per_voxel = voxel_len * voxel_len * voxel_len;

    const int tid = sif_system_get_thread_num();
    real_t* local_mass = scratch_mass + (size_t)tid * n_bins;
    real_t* local_vrad = scratch_vrad + (size_t)tid * n_bins;
    real_t* local_vol = scratch_vol + (size_t)tid * n_bins;

    if (compute_dens)
      memset(local_mass, 0, n_bins * sizeof(real_t));
    if (compute_vel) {
      memset(local_vrad, 0, n_bins * sizeof(real_t));
      memset(local_vol, 0, n_bins * sizeof(real_t));
    }

    /* Sub-Grid Sweep */
    for (uint32_t ix = 0; ix < __SIF_PROFILE_GRID_DIM; ix++) {
      for (uint32_t iy = 0; iy < __SIF_PROFILE_GRID_DIM; iy++) {
        for (uint32_t iz = 0; iz < __SIF_PROFILE_GRID_DIM; iz++) {

          // Physical coordinate of the voxel
          real_t px = (cx - r_max) + (ix + 0.5f) * voxel_len;
          real_t py = (cy - r_max) + (iy + 0.5f) * voxel_len;
          real_t pz = (cz - r_max) + (iz + 0.5f) * voxel_len;

          // Geometric vector from void center to the voxel
          real_t dx = px - cx;
          real_t dy = py - cy;
          real_t dz = pz - cz;
          real_t r2 = dx * dx + dy * dy + dz * dz;

          if (r2 <= r_max_sq) {
            real_t r = REAL_SQRT(r2);
            uint32_t bin = (uint32_t)(r / (rv * bin_width));

            if (bin < n_bins) {

              // Wrap the query point into the box space if it extended beyond
              // the boundaries
              /* The chain mesh is anchored at the origin, so wrap into
               * [0, box_length) rather than into the field's bounding box. */
              real_t qx = px, qy = py, qz = pz;
              if (is_periodic) {
                qx = __SIF_WRAP_PBC(px, 0.0f, box_length);
                qy = __SIF_WRAP_PBC(py, 0.0f, box_length);
                qz = __SIF_WRAP_PBC(pz, 0.0f, box_length);
              }

              // Query the fast spatial index
              uint64_t nearest_idx =
                is_periodic
                  ? sif_chain_mesh_find_nearest_pbc(mesh, qx, qy, qz)
                  : sif_chain_mesh_find_nearest_open(mesh, qx, qy, qz);

              if (nearest_idx != UINT64_MAX) {

                // The exact Voronoi volume intersection
                const real_t cell_vol = tess->volumes[nearest_idx];
                if (!(cell_vol > 0.0f))
                  continue; /* degenerate cell: contributes nothing */

                real_t vol_fraction = vol_per_voxel / cell_vol;

                if (compute_dens) {
                  real_t m = fm ? fm[nearest_idx] : 1.0f;
                  local_mass[bin] += m * vol_fraction;
                }

                if (compute_vel && fvx) {
                  real_t inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                  real_t v_r =
                    (fvx[nearest_idx] * dx + fvy[nearest_idx] * dy +
                      fvz[nearest_idx] * dz) *
                    inv_r;

                  local_vrad[bin] += v_r * vol_per_voxel;
                  local_vol[bin] += vol_per_voxel;
                }
              }
            }
          }
        }
      }
    }

    /* Finalize the void's profile */
    real_t rv_cubed = rv * rv * rv;
    real_t cumulative_mass = 0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        cumulative_mass += local_mass[j];
        real_t raw_rho = cumulative_mass / (sphere_vols[j] * rv_cubed);
        (*out_dens)->profiles[global_idx] = (raw_rho / mean_dens) - 1.0;
      }

      if (compute_vel) {
        if (local_vol[j] > 0.0f) {
          (*out_vel)->v_rad[global_idx] = local_vrad[j] / local_vol[j];
        } else {
          (*out_vel)->v_rad[global_idx] = 0.0f;
        }
      }
    }
  }

  sif_free_aligned(scratch_mass);
  sif_free_aligned(scratch_vrad);
  sif_free_aligned(scratch_vol);
  sif_free_aligned(sphere_vols);
  sif_chain_mesh_free(mesh);

  SIF_LOG_INFO("profiles", "Voronoi profile computation completed");
}
