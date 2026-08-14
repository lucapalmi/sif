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

/*
 * Elements per per-thread scratch row, rounded up so each thread's row starts
 * on its own cache line.
 */
static size_t pad_row(uint32_t n_bins, size_t elem_size) {
  const size_t per_line = SIF_CACHE_LINE / elem_size;
  return ((n_bins + per_line - 1) / per_line) * per_line;
}

/* --- memory --- */

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

/* --- shared setup --- */

/*
 * Validates what both estimators need before either allocates anything.
 */
static int validate_inputs(const sif_catalog_t* cat, const sif_field_t* field,
  sif_real box_length, sif_real ext, uint32_t n_bins,
  const sif_density_profiles_t* const* out_dens,
  const sif_velocity_profiles_t* const* out_vel, const char* who) {

  if (!cat || !field || cat->n_voids == 0) {
    SIF_LOG_ERROR("profiles", "invalid catalogue or field passed to %s", who);
    return SIF_ERR_INVALID;
  }

  if (!out_dens && !out_vel) {
    SIF_LOG_ERROR("profiles",
      "both outputs are NULL in %s; there is nothing to compute", who);
    return SIF_ERR_INVALID;
  }

  /* n_bins divides ext and ext scales every radius, so a zero in either turns
   * the bin width into 0 or infinity and the bin index into an undefined cast
   * rather than into a wrong answer. */
  if (n_bins == 0 || !(ext > (sif_real)0.0) || !(box_length > (sif_real)0.0)) {
    SIF_LOG_ERROR("profiles",
      "%s needs n_bins > 0, ext > 0 and box_length > 0 (got %u, %g, %g)", who,
      n_bins, (double)ext, (double)box_length);
    return SIF_ERR_INVALID;
  }

  if (out_vel && !field->vx) {
    SIF_LOG_ERROR("profiles",
      "velocity profiles were requested but the field carries no velocities");
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/*
 * Mean tracer density of the box, which normalizes every density profile.
 *
 * Accumulated in double however sif_real is configured, and reduced over a
 * plain OpenMP sum. The weight total runs over every particle in the box: at
 * float precision the running sum stops moving long before it gets there, and
 * because it divides every profile the error would show up as an overall
 * amplitude shift that looks exactly like a physical result.
 */
static double mean_density(const sif_field_t* field, sif_real box_length) {
  const double box_vol =
    (double)box_length * (double)box_length * (double)box_length;

  if (!field->weights)
    return (double)field->n_particles / box_vol;

  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (uint64_t p = 0; p < field->n_particles; p++)
    total_mass += (double)field->weights[p];

  return total_mass / box_vol;
}

/*
 * Chain-mesh resolution: about one cell per search radius.
 *
 * Finer would walk more cells per void than the particles saved are worth;
 * coarser would sweep particles far outside r_max. Capped at 256 because the
 * mesh itself is n^3 cells and a large box with tiny voids would otherwise ask
 * for a mesh bigger than the field.
 */
static uint32_t mesh_resolution(
  const sif_catalog_t* cat, sif_real box_length, sif_real ext) {

  double sum_radius = 0.0;
  for (uint64_t i = 0; i < cat->n_voids; i++)
    sum_radius += (double)cat->radii[i];

  const double avg_radius = sum_radius / (double)cat->n_voids;
  if (!(avg_radius > 0.0))
    return 1;

  const double target = (double)box_length / (avg_radius * (double)ext);
  if (target >= 256.0)
    return 256;
  if (target >= 1.0)
    return (uint32_t)target;

  return 1;
}

/* Fills r_edges and, for densities, the volume of the whole sphere out to each
 * bin's outer edge. Edges are in units of the void radius, so the volumes are
 * too and get scaled by rv^3 per void at the end. */
static void fill_edges(sif_real* r_edges, sif_real* sphere_vols,
  uint32_t n_bins, sif_real ext, sif_real bin_width) {

  for (uint32_t j = 0; j < n_bins; j++) {
    r_edges[j] = (sif_real)j * bin_width;

    if (sphere_vols) {
      const sif_real r_outer = (sif_real)(j + 1) * bin_width;
      sphere_vols[j] =
        (sif_real)(4.0 / 3.0) * SIF_PI * (r_outer * r_outer * r_outer);
    }
  }

  /* Assigned rather than accumulated, so the top edge is exactly the ext the
   * caller asked for. */
  r_edges[n_bins] = ext;
}

/* --- mesh estimator --- */

int sif_profiles_mesh(const sif_catalog_t* cat, const sif_field_t* field,
  sif_real box_length, sif_real ext, uint32_t n_bins, sif_option opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  int status = validate_inputs(cat, field, box_length, ext, n_bins,
    (const sif_density_profiles_t* const*)out_dens,
    (const sif_velocity_profiles_t* const*)out_vel, "sif_profiles_mesh");
  if (status != SIF_OK)
    return status;

  const int compute_dens = (out_dens != NULL);
  const int compute_vel = (out_vel != NULL);

  SIF_LOG_INFO(
    "profiles", "computing profiles for %" PRIu64 " voids", cat->n_voids);

  /* Allocated here only if the caller passed a pointer to NULL; a set it
   * already owns is refilled, which is how a sweep over several catalogues
   * avoids reallocating the same shape every time. */
  if (compute_dens && *out_dens == NULL) {
    *out_dens = density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return SIF_ERR_ALLOC;
    }
  }

  if (compute_vel && *out_vel == NULL) {
    *out_vel = velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
  }

  const sif_real bin_width = ext / (sif_real)n_bins;
  sif_real* sphere_vols = NULL;

  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(sif_real));
    if (!sphere_vols) {
      SIF_LOG_ERROR("profiles", "OOM allocating the shell volumes");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
    fill_edges((*out_dens)->r_edges, sphere_vols, n_bins, ext, bin_width);
  }

  if (compute_vel)
    fill_edges((*out_vel)->r_edges, NULL, n_bins, ext, bin_width);

  const uint32_t n_cells = mesh_resolution(cat, box_length, ext);

  SIF_LOG_TRACE("profiles", "building a %u^3 chain mesh", n_cells);

  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(n_cells, box_length, field);

  if (!mesh) {
    SIF_LOG_ERROR("profiles",
      "failed to build the chain mesh (are all particles inside the box?)");
    sif_free_aligned(sphere_vols);
    status = SIF_ERR_ALLOC;
    goto fail;
  }

  const sif_real half_box = box_length * (sif_real)0.5;
  const sif_real inv_cell_len = (sif_real)1.0 / mesh->cell_length;

  const sif_real mean_dens =
    compute_dens ? (sif_real)mean_density(field, box_length) : (sif_real)1.0;

  const int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const sif_real* mx = SIF_ASSUME_ALIGNED(mesh->x);
  const sif_real* my = SIF_ASSUME_ALIGNED(mesh->y);
  const sif_real* mz = SIF_ASSUME_ALIGNED(mesh->z);
  const sif_real* mm = mesh->weights ? SIF_ASSUME_ALIGNED(mesh->weights) : NULL;
  const sif_real* mvx = mesh->vx ? SIF_ASSUME_ALIGNED(mesh->vx) : NULL;
  const sif_real* mvy = mesh->vy ? SIF_ASSUME_ALIGNED(mesh->vy) : NULL;
  const sif_real* mvz = mesh->vz ? SIF_ASSUME_ALIGNED(mesh->vz) : NULL;

  /* Per-thread bin scratch, allocated once on the heap. These used to be VLAs
   * inside the parallel region, which overflows an OpenMP thread stack (far
   * smaller than the main stack) for a large caller-supplied n_bins.
   *
   * Rows are padded to a whole cache line. Without it two threads binning
   * different voids share the line at their boundary and bounce it between
   * cores on every particle, which is most of the inner loop's work. */
  const int n_threads =
    sif__system_max_threads() > 0 ? sif__system_max_threads() : 1;
  const size_t row_real = pad_row(n_bins, sizeof(sif_real));
  const size_t row_count = pad_row(n_bins, sizeof(uint64_t));

  sif_real* scratch_mass =
    sif_malloc_aligned((size_t)n_threads * row_real * sizeof(sif_real));
  sif_real* scratch_vrad =
    sif_malloc_aligned((size_t)n_threads * row_real * sizeof(sif_real));
  uint64_t* scratch_count =
    sif_malloc_aligned((size_t)n_threads * row_count * sizeof(uint64_t));

  if (!scratch_mass || !scratch_vrad || !scratch_count) {
    SIF_LOG_ERROR("profiles", "OOM allocating the per-thread bin scratch");
    sif_free_aligned(scratch_mass);
    sif_free_aligned(scratch_vrad);
    sif_free_aligned(scratch_count);
    sif_free_aligned(sphere_vols);
    sif_chain_mesh_free(mesh);
    status = SIF_ERR_ALLOC;
    goto fail;
  }

/* --- core loop --- */
#pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    sif_real cx = cat->cx[i];
    sif_real cy = cat->cy[i];
    sif_real cz = cat->cz[i];
    sif_real rv = cat->radii[i];

    /* A void with no radius has no profile: every bin index would be r/0, and
     * the cast of the resulting NaN to uint32_t is undefined. The row stays as
     * the allocator left it, which is zeroed. */
    if (!(rv > (sif_real)0.0))
      continue;

    sif_real r_max = rv * ext;
    sif_real r_max_sq = r_max * r_max;

    const int tid = sif__system_thread_num();
    sif_real* local_mass = scratch_mass + (size_t)tid * row_real;
    sif_real* local_vrad = scratch_vrad + (size_t)tid * row_real;
    uint64_t* local_count = scratch_count + (size_t)tid * row_count;

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

    /* The density profile is cumulative and the velocity profile is not.
     * That is the convention each is used under: an enclosed density contrast
     * is what the spherical-evolution mapping takes, while a radial velocity
     * means the mean infall of a shell. */
    const sif_real rv_cubed = rv * rv * rv;
    sif_real cumulative_mass = (sif_real)0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      const uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        cumulative_mass += local_mass[j];

        /* Enclosed weight over the volume of the whole sphere out to this bin's
         * outer edge -- sphere_vols is in units of the void radius, so rv^3
         * puts it back in physical units. */
        const sif_real raw_rho = cumulative_mass / (sphere_vols[j] * rv_cubed);
        (*out_dens)->profiles[global_idx] =
          (raw_rho / mean_dens) - (sif_real)1.0;
      }

      if (compute_vel) {
        (*out_vel)->v_rad[global_idx] =
          (local_count[j] > 0) ? local_vrad[j] / (sif_real)local_count[j]
                               : (sif_real)0.0;
      }
    }
  }

  sif_free_aligned(scratch_mass);
  sif_free_aligned(scratch_vrad);
  sif_free_aligned(scratch_count);
  sif_free_aligned(sphere_vols);
  sif_chain_mesh_free(mesh);

  SIF_LOG_INFO("profiles", "profile computation completed");
  return SIF_OK;

fail:
  /* Every failure leaves both outputs NULL, including one this call allocated
   * before a later step failed. A half-built set the caller cannot tell from a
   * finished one is worse than no set at all. */
  if (out_dens && *out_dens) {
    sif_density_profiles_free(*out_dens);
    *out_dens = NULL;
  }
  if (out_vel && *out_vel) {
    sif_velocity_profiles_free(*out_vel);
    *out_vel = NULL;
  }
  return status;
}

/* --- Voronoi estimator --- */

int sif_profiles_voronoi(const sif_catalog_t* cat, const sif_field_t* field,
  const sif_tessellation_t* tess, sif_real box_length, sif_real ext,
  uint32_t n_bins, sif_option opt, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel) {

  int status = validate_inputs(cat, field, box_length, ext, n_bins,
    (const sif_density_profiles_t* const*)out_dens,
    (const sif_velocity_profiles_t* const*)out_vel, "sif_profiles_voronoi");
  if (status != SIF_OK)
    return status;

  if (!tess || !tess->volumes) {
    SIF_LOG_ERROR("profiles", "the Voronoi estimator needs a tessellation");
    return SIF_ERR_INVALID;
  }

  const int compute_dens = (out_dens != NULL);
  const int compute_vel = (out_vel != NULL);

  SIF_LOG_INFO("profiles",
    "computing volume-weighted Voronoi profiles for %" PRIu64 " voids",
    cat->n_voids);

  if (compute_dens && *out_dens == NULL) {
    *out_dens = density_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return SIF_ERR_ALLOC;
    }
  }

  if (compute_vel && *out_vel == NULL) {
    *out_vel = velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
  }

  const sif_real bin_width = ext / (sif_real)n_bins;
  sif_real* sphere_vols = NULL;

  if (compute_dens) {
    sphere_vols = sif_malloc_aligned(n_bins * sizeof(sif_real));
    if (!sphere_vols) {
      SIF_LOG_ERROR("profiles", "OOM allocating the shell volumes");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
    fill_edges((*out_dens)->r_edges, sphere_vols, n_bins, ext, bin_width);
  }

  if (compute_vel)
    fill_edges((*out_vel)->r_edges, NULL, n_bins, ext, bin_width);

  /* The mesh carries positions only: this estimator reads weights and
   * velocities from the field, indexed by what find_nearest returns, which is
   * an index into the field rather than into the mesh's own ordering. */
  const uint32_t n_cells = mesh_resolution(cat, box_length, ext);
  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(n_cells, box_length, field);

  if (!mesh) {
    SIF_LOG_ERROR("profiles",
      "failed to build the chain mesh (are all particles inside the box?)");
    sif_free_aligned(sphere_vols);
    status = SIF_ERR_ALLOC;
    goto fail;
  }

  const sif_real mean_dens =
    compute_dens ? (sif_real)mean_density(field, box_length) : (sif_real)1.0;

  const int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const sif_real* fm =
    field->weights ? SIF_ASSUME_ALIGNED(field->weights) : NULL;
  const sif_real* fvx = field->vx ? SIF_ASSUME_ALIGNED(field->vx) : NULL;
  const sif_real* fvy = field->vy ? SIF_ASSUME_ALIGNED(field->vy) : NULL;
  const sif_real* fvz = field->vz ? SIF_ASSUME_ALIGNED(field->vz) : NULL;

  /* Per-thread bin scratch, padded per thread. See the note in
   * sif_profiles_mesh. */
  const int n_threads =
    sif__system_max_threads() > 0 ? sif__system_max_threads() : 1;
  const size_t row_real = pad_row(n_bins, sizeof(sif_real));

  sif_real* scratch_mass =
    sif_malloc_aligned((size_t)n_threads * row_real * sizeof(sif_real));
  sif_real* scratch_vrad =
    sif_malloc_aligned((size_t)n_threads * row_real * sizeof(sif_real));
  sif_real* scratch_vol =
    sif_malloc_aligned((size_t)n_threads * row_real * sizeof(sif_real));

  if (!scratch_mass || !scratch_vrad || !scratch_vol) {
    SIF_LOG_ERROR("profiles", "OOM allocating the per-thread bin scratch");
    sif_free_aligned(scratch_mass);
    sif_free_aligned(scratch_vrad);
    sif_free_aligned(scratch_vol);
    sif_free_aligned(sphere_vols);
    sif_chain_mesh_free(mesh);
    status = SIF_ERR_ALLOC;
    goto fail;
  }

/* --- core loop --- */
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    sif_real cx = cat->cx[i];
    sif_real cy = cat->cy[i];
    sif_real cz = cat->cz[i];
    sif_real rv = cat->radii[i];

    /* A void with no radius has no profile: every bin index would be r/0, and
     * the cast of the resulting NaN to uint32_t is undefined. The row stays as
     * the allocator left it, which is zeroed. */
    if (!(rv > (sif_real)0.0))
      continue;

    sif_real r_max = rv * ext;
    sif_real r_max_sq = r_max * r_max;

    /* Local Voxel Grid Definition */
    sif_real voxel_len = (2.0f * r_max) / (sif_real)PROFILES_GRID_DIM;
    sif_real vol_per_voxel = voxel_len * voxel_len * voxel_len;

    const int tid = sif__system_thread_num();
    sif_real* local_mass = scratch_mass + (size_t)tid * row_real;
    sif_real* local_vrad = scratch_vrad + (size_t)tid * row_real;
    sif_real* local_vol = scratch_vol + (size_t)tid * row_real;

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
  return SIF_OK;

fail:
  if (out_dens && *out_dens) {
    sif_density_profiles_free(*out_dens);
    *out_dens = NULL;
  }
  if (out_vel && *out_vel) {
    sif_velocity_profiles_free(*out_vel);
    *out_vel = NULL;
  }
  return status;
}
