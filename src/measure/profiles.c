/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/measure/profiles.h"

#include "core/system_internal.h"
#include "sif/core/macros.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Side of the auxiliary voxel grid used by the Voronoi estimator. Private to
 * this file: it is a tuning constant, not part of the library's interface. */
#define PROFILES_GRID_DIM 100

#define WRAP_PBC(val, min_val, box_len)                                        \
  (SIF_REAL_FMOD(                                                              \
     SIF_REAL_FMOD((val) - (min_val), (box_len)) + (box_len), (box_len)) +     \
    (min_val))

/* --- Memory Allocation --- */

SIF_NODISCARD static sif_density_profiles_t* density_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext) {

  sif_density_profiles_t* profs = malloc(sizeof(sif_density_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;

  profs->r_edges = sif_malloc_aligned((n_bins + 1) * sizeof(sif_real));
  profs->profiles = sif_calloc_aligned(n_voids * n_bins, sizeof(sif_real));

  if (!profs->r_edges || !profs->profiles) {
    sif_density_profiles_free(profs);
    return NULL;
  }
  return profs;
}

SIF_NODISCARD static sif_velocity_profiles_t* velocity_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext) {

  sif_velocity_profiles_t* profs = malloc(sizeof(sif_velocity_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;

  profs->r_edges = sif_malloc_aligned((n_bins + 1) * sizeof(sif_real));
  profs->v_rad = sif_calloc_aligned(n_voids * n_bins, sizeof(sif_real));

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

void sif_profiles_mesh(const sif_catalog_t* cat, const sif_field_t* field,
  sif_real box_length, sif_real ext, uint32_t n_bins, sif_option opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (!cat || !field || cat->n_voids == 0) {
    SIF_LOG_ERROR("profiles", "invalid inputs to sif_profiles_mesh");
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
    *out_dens = density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return;
    }
  }

  if (compute_vel && *out_vel == NULL) {
    *out_vel = velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      if (compute_dens && out_dens) {
        sif_density_profiles_free(*out_dens);
        *out_dens = NULL;
      }
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      return;
    }
  }

  sif_real bin_width = ext / (sif_real)n_bins;
  sif_real* sphere_vols = NULL;

  /* Setup radial edges and full SPHERE volumes */
  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(sif_real));
    for (uint32_t j = 0; j < n_bins; j++) {
      sif_real r_inner = (sif_real)j * bin_width;
      sif_real r_outer = (sif_real)(j + 1) * bin_width;
      (*out_dens)->r_edges[j] = r_inner;
      /* Calculate the volume of the entire sphere up to r_outer */
      sphere_vols[j] = (4.0 / 3.0) * SIF_PI * (r_outer * r_outer * r_outer);
    }
    (*out_dens)->r_edges[n_bins] = ext;
  }

  if (compute_vel) {
    for (uint32_t j = 0; j <= n_bins; j++) {
      (*out_vel)->r_edges[j] = (sif_real)j * bin_width;
    }
  }

  /* Determine optimal mesh resolution */
  sif_real avg_radius = 0.0;
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    avg_radius += cat->radii[i];
  }
  avg_radius /= (sif_real)cat->n_voids;

  /* avg_radius or ext can legitimately be zero for a degenerate catalog; the
   * division would then produce inf and an undefined cast. */
  uint32_t n_cells = 1;
  if (avg_radius > 0.0f && ext > 0.0f) {
    sif_real target = box_length / (avg_radius * ext);
    if (target >= 256.0f)
      n_cells = 256;
    else if (target >= 1.0f)
      n_cells = (uint32_t)target;
  }

  SIF_LOG_TRACE("profiles", "Building chain mesh with %ux%ux%u cells", n_cells,
    n_cells, n_cells);

  /* Build the mesh */
  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(
    n_cells, box_length, field, compute_dens, compute_vel, true);

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

  sif_real half_box = box_length * 0.5;
  sif_real inv_cell_len = 1.0 / mesh->cell_length;

  /* Calculate Mean Density for Overdensity Normalization */
  sif_real mean_dens = 1.0;
  if (compute_dens) {
    sif_real box_vol = box_length * box_length * box_length;
    if (field->masses) {
      sif_real total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
      for (uint64_t p = 0; p < field->n_particles; p++) {
        total_mass += field->masses[p];
      }
      mean_dens = total_mass / box_vol;
    } else {
      mean_dens = (sif_real)field->n_particles / box_vol;
    }
  }

  int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mm = mesh->masses ? SIF_ASSUME_ALIGNED(mesh->masses) : NULL;
  const sif_real* mvx = mesh->vx ? SIF_ASSUME_ALIGNED(mesh->vx) : NULL;
  const sif_real* mvy = mesh->vy ? SIF_ASSUME_ALIGNED(mesh->vy) : NULL;
  const sif_real* mvz = mesh->vz ? SIF_ASSUME_ALIGNED(mesh->vz) : NULL;

  /* Per-thread bin scratch, allocated once on the heap. These used to be VLAs
   * inside the parallel region, which overflows an OpenMP thread stack (far
   * smaller than the main stack) for a large caller-supplied n_bins. */
  const int n_threads =
    sif__system_max_threads() > 0 ? sif__system_max_threads() : 1;
  sif_real* scratch_mass =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(sif_real));
  sif_real* scratch_vrad =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(sif_real));
  uint64_t* scratch_count =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(uint64_t));

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
    sif_real cx = cat->cx[i];
    sif_real cy = cat->cy[i];
    sif_real cz = cat->cz[i];
    sif_real rv = cat->radii[i];

    sif_real r_max = rv * ext;
    sif_real r_max_sq = r_max * r_max;

    const int tid = sif__system_thread_num();
    sif_real* local_mass = scratch_mass + (size_t)tid * n_bins;
    sif_real* local_vrad = scratch_vrad + (size_t)tid * n_bins;
    uint64_t* local_count = scratch_count + (size_t)tid * n_bins;

    if (compute_dens)
      memset(local_mass, 0, n_bins * sizeof(sif_real));
    if (compute_vel) {
      memset(local_vrad, 0, n_bins * sizeof(sif_real));
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
              sif_real dx = mx[p] - cx;
              sif_real dy = my[p] - cy;
              sif_real dz = mz[p] - cz;

              /* Branchless Periodic Boundary Wrap */
              dx -= box_length * (sif_real)(dx > half_box) -
                    box_length * (sif_real)(dx < -half_box);
              dy -= box_length * (sif_real)(dy > half_box) -
                    box_length * (sif_real)(dy < -half_box);
              dz -= box_length * (sif_real)(dz > half_box) -
                    box_length * (sif_real)(dz < -half_box);

              sif_real r2 = dx * dx + dy * dy + dz * dz;

              /* Mask evaluation - compilers handle this efficiently in SIMD */
              if (r2 <= r_max_sq) {
                sif_real r = SIF_REAL_SQRT(r2);
                uint32_t bin = (uint32_t)(r / (rv * bin_width));

                if (bin < n_bins) {
                  if (compute_dens) {
                    sif_real m = mm ? mm[p] : 1.0f;
                    local_mass[bin] += m;
                  }

                  if (compute_vel && mvx) {
                    sif_real inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                    sif_real v_r =
                      (mvx[p] * dx + mvy[p] * dy + mvz[p] * dz) * inv_r;
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
              sif_real dx = mx[p] - cx;
              sif_real dy = my[p] - cy;
              sif_real dz = mz[p] - cz;

              sif_real r2 = dx * dx + dy * dy + dz * dz;

              if (r2 <= r_max_sq) {
                sif_real r = SIF_REAL_SQRT(r2);
                uint32_t bin = (uint32_t)(r / (rv * bin_width));

                if (bin < n_bins) {
                  if (compute_dens) {
                    sif_real m = mm ? mm[p] : 1.0f;
                    local_mass[bin] += m;
                  }

                  if (compute_vel && mvx) {
                    sif_real inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                    sif_real v_r =
                      (mvx[p] * dx + mvy[p] * dy + mvz[p] * dz) * inv_r;
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
    sif_real rv_cubed = rv * rv * rv;
    sif_real cumulative_mass = 0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        /* Add the current shell's mass to the running total */
        cumulative_mass += local_mass[j];

        /* Divide total enclosed mass by total spherical volume */
        sif_real raw_rho = cumulative_mass / (sphere_vols[j] * rv_cubed);
        (*out_dens)->profiles[global_idx] = (raw_rho / mean_dens) - 1.0;
      }

      if (compute_vel) {
        /* Velocity is typically left as differential (average of the shell) */
        if (local_count[j] > 0) {
          (*out_vel)->v_rad[global_idx] =
            local_vrad[j] / (sif_real)local_count[j];
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

void sif_profiles_voronoi(const sif_catalog_t* cat, const sif_field_t* field,
  const sif_tessellation_t* tess, sif_real box_length, sif_real ext,
  uint32_t n_bins, sif_option opt, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel) {

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
    *out_dens = density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return;
    }
  }
  if (compute_vel && *out_vel == NULL) {
    *out_vel = velocity_profiles_alloc(cat->n_voids, n_bins, ext);
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
  sif_real bin_width = ext / (sif_real)n_bins;
  sif_real* sphere_vols = NULL;

  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(sif_real));
    for (uint32_t j = 0; j < n_bins; j++) {
      sif_real r_outer = (sif_real)(j + 1) * bin_width;
      (*out_dens)->r_edges[j] = (sif_real)j * bin_width;
      sphere_vols[j] = (4.0 / 3.0) * SIF_PI * (r_outer * r_outer * r_outer);
    }
    (*out_dens)->r_edges[n_bins] = ext;
  }

  if (compute_vel) {
    for (uint32_t j = 0; j <= n_bins; j++) {
      (*out_vel)->r_edges[j] = (sif_real)j * bin_width;
    }
  }

  /* Chain mesh allocation (Fast Spatial Index) */
  sif_real avg_radius = 0.0;
  for (uint64_t i = 0; i < cat->n_voids; i++)
    avg_radius += cat->radii[i];
  avg_radius /= (sif_real)cat->n_voids;

  /* avg_radius or ext can legitimately be zero for a degenerate catalog; the
   * division would then produce inf and an undefined cast. */
  uint32_t n_cells = 1;
  if (avg_radius > 0.0f && ext > 0.0f) {
    sif_real target = box_length / (avg_radius * ext);
    if (target >= 256.0f)
      n_cells = 256;
    else if (target >= 1.0f)
      n_cells = (uint32_t)target;
  }

  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(n_cells, box_length, field, false, false, true);

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
  sif_real mean_dens = 1.0;
  if (compute_dens) {
    sif_real box_vol = box_length * box_length * box_length;
    if (field->masses) {
      sif_real total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
      for (uint64_t p = 0; p < field->n_particles; p++)
        total_mass += field->masses[p];
      mean_dens = total_mass / box_vol;
    } else {
      mean_dens = (sif_real)field->n_particles / box_vol;
    }
  }

  int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const sif_real* fm = field->masses ? SIF_ASSUME_ALIGNED(field->masses) : NULL;
  const sif_real* fvx = field->vx ? SIF_ASSUME_ALIGNED(field->vx) : NULL;
  const sif_real* fvy = field->vy ? SIF_ASSUME_ALIGNED(field->vy) : NULL;
  const sif_real* fvz = field->vz ? SIF_ASSUME_ALIGNED(field->vz) : NULL;

  /* Per-thread bin scratch, see the note in sif_profiles_mesh. */
  const int n_threads =
    sif__system_max_threads() > 0 ? sif__system_max_threads() : 1;
  sif_real* scratch_mass =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(sif_real));
  sif_real* scratch_vrad =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(sif_real));
  sif_real* scratch_vol =
    sif_malloc_aligned((size_t)n_threads * n_bins * sizeof(sif_real));

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
    sif_real cx = cat->cx[i];
    sif_real cy = cat->cy[i];
    sif_real cz = cat->cz[i];
    sif_real rv = cat->radii[i];

    sif_real r_max = rv * ext;
    sif_real r_max_sq = r_max * r_max;

    /* Local Voxel Grid Definition */
    sif_real voxel_len = (2.0f * r_max) / (sif_real)PROFILES_GRID_DIM;
    sif_real vol_per_voxel = voxel_len * voxel_len * voxel_len;

    const int tid = sif__system_thread_num();
    sif_real* local_mass = scratch_mass + (size_t)tid * n_bins;
    sif_real* local_vrad = scratch_vrad + (size_t)tid * n_bins;
    sif_real* local_vol = scratch_vol + (size_t)tid * n_bins;

    if (compute_dens)
      memset(local_mass, 0, n_bins * sizeof(sif_real));
    if (compute_vel) {
      memset(local_vrad, 0, n_bins * sizeof(sif_real));
      memset(local_vol, 0, n_bins * sizeof(sif_real));
    }

    /* Sub-Grid Sweep */
    for (uint32_t ix = 0; ix < PROFILES_GRID_DIM; ix++) {
      for (uint32_t iy = 0; iy < PROFILES_GRID_DIM; iy++) {
        for (uint32_t iz = 0; iz < PROFILES_GRID_DIM; iz++) {

          // Physical coordinate of the voxel
          sif_real px = (cx - r_max) + (ix + 0.5f) * voxel_len;
          sif_real py = (cy - r_max) + (iy + 0.5f) * voxel_len;
          sif_real pz = (cz - r_max) + (iz + 0.5f) * voxel_len;

          // Geometric vector from void center to the voxel
          sif_real dx = px - cx;
          sif_real dy = py - cy;
          sif_real dz = pz - cz;
          sif_real r2 = dx * dx + dy * dy + dz * dz;

          if (r2 <= r_max_sq) {
            sif_real r = SIF_REAL_SQRT(r2);
            uint32_t bin = (uint32_t)(r / (rv * bin_width));

            if (bin < n_bins) {

              // Wrap the query point into the box space if it extended beyond
              // the boundaries
              /* The chain mesh is anchored at the origin, so wrap into
               * [0, box_length) rather than into the field's bounding box. */
              sif_real qx = px, qy = py, qz = pz;
              if (is_periodic) {
                qx = WRAP_PBC(px, 0.0f, box_length);
                qy = WRAP_PBC(py, 0.0f, box_length);
                qz = WRAP_PBC(pz, 0.0f, box_length);
              }

              // Query the fast spatial index
              uint64_t nearest_idx =
                is_periodic
                  ? sif_chain_mesh_find_nearest_pbc(mesh, qx, qy, qz)
                  : sif_chain_mesh_find_nearest_open(mesh, qx, qy, qz);

              if (nearest_idx != UINT64_MAX) {

                // The exact Voronoi volume intersection
                const sif_real cell_vol = tess->volumes[nearest_idx];
                if (!(cell_vol > 0.0f))
                  continue; /* degenerate cell: contributes nothing */

                sif_real vol_fraction = vol_per_voxel / cell_vol;

                if (compute_dens) {
                  sif_real m = fm ? fm[nearest_idx] : 1.0f;
                  local_mass[bin] += m * vol_fraction;
                }

                if (compute_vel && fvx) {
                  sif_real inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;
                  sif_real v_r =
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
    sif_real rv_cubed = rv * rv * rv;
    sif_real cumulative_mass = 0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        cumulative_mass += local_mass[j];
        sif_real raw_rho = cumulative_mass / (sphere_vols[j] * rv_cubed);
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
