/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/measure/profiles.h"

#include "core/system_internal.h"
#include "measure/profiles_internal.h"
#include "sif/core/macros.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Elements per per-thread scratch row, rounded up so each thread's row starts
 * on its own cache line.
 */
static size_t pad_row(uint32_t n_bins, size_t elem_size) {
  const size_t per_line = SIF_CACHE_LINE / elem_size;
  return ((n_bins + per_line - 1) / per_line) * per_line;
}

/* --- memory --- */

sif_density_profiles_t* sif__density_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext, bool differential) {

  sif_density_profiles_t* profs = malloc(sizeof(sif_density_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;
  profs->differential = differential;
  profs->source_id = 0;

  profs->r_edges = sif_malloc_aligned((n_bins + 1) * sizeof(sif_real));
  profs->profiles = sif_calloc_aligned(n_voids * n_bins, sizeof(sif_real));

  if (!profs->r_edges || !profs->profiles) {
    sif_density_profiles_free(profs);
    return NULL;
  }
  return profs;
}

sif_velocity_profiles_t* sif__velocity_profiles_alloc(
  uint64_t n_voids, uint32_t n_bins, sif_real ext) {

  sif_velocity_profiles_t* profs = malloc(sizeof(sif_velocity_profiles_t));
  if (!profs)
    return NULL;

  profs->n_voids = n_voids;
  profs->n_bins = n_bins;
  profs->ext = ext;
  profs->source_id = 0;

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

/* --- setup --- */

/*
 * Everything that has to hold before anything is allocated.
 */
static int validate_inputs(const sif_catalog_t* cat,
  const sif_chain_mesh_t* mesh, uint32_t n_bins,
  const sif_density_profiles_t* const* out_dens,
  const sif_velocity_profiles_t* const* out_vel) {

  if (!cat || cat->n_voids == 0) {
    SIF_LOG_ERROR("profiles", "no voids to profile");
    return SIF_ERR_INVALID;
  }

  /* The mean density divides every profile, and it is the mesh that supplies
   * both the tracer count and the volume it spreads over. An empty mesh, or
   * one that cannot name its box, has no mean density to normalize by. */
  if (!mesh || mesh->n_particles == 0 || !(mesh->box_length > (sif_real)0.0)) {
    SIF_LOG_ERROR("profiles", "invalid or empty chain mesh");
    return SIF_ERR_INVALID;
  }

  if (!out_dens && !out_vel) {
    SIF_LOG_ERROR(
      "profiles", "both outputs are NULL; there is nothing to compute");
    return SIF_ERR_INVALID;
  }

  /* n_bins divides ext, so a zero turns the bin width into 0 and the bin
   * index into an undefined cast rather than into a wrong answer. */
  if (n_bins == 0) {
    SIF_LOG_ERROR("profiles", "n_bins must be non-zero");
    return SIF_ERR_INVALID;
  }

  if (out_vel && !mesh->vx) {
    SIF_LOG_ERROR("profiles",
      "velocity profiles were requested but the mesh carries no velocities");
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

/*
 * Mean tracer density of the box, which normalizes every density profile.
 *
 * The mesh summed its own weights when it was built, so this is arithmetic
 * rather than a pass over the tracers -- which matters when several catalogues
 * are profiled against one mesh, since each call would otherwise redo it.
 */
static double mean_density(const sif_chain_mesh_t* mesh) {
  const double box_vol = (double)mesh->box_length * (double)mesh->box_length *
                         (double)mesh->box_length;

  return mesh->total_weight / box_vol;
}

/* --- inner kernel --- */

/*
 * Everything the per-cell kernel reads that does not change between voids.
 */
typedef struct {
  const sif_real* x;
  const sif_real* y;
  const sif_real* z;
  const sif_real* w;
  const sif_real* vx;
  const sif_real* vy;
  const sif_real* vz;
  sif_real box_length;
  sif_real half_box;
  sif_real inv_box;
  sif_real cell_length;
  sif_real half_cell;
  uint32_t n_bins;
} bin_ctx_t;

/*
 * Bins one cell's run of tracers around one void.
 *
 * The last four arguments are constants at every call site, so each
 * instantiation the dispatch below asks for drops the tests it does not need
 * -- the weight load, the velocity arm, the periodic wrap -- instead of
 * re-deciding them once per tracer in the innermost loop of the library.
 *
 * @p cx,cy,cz is the void centre already placed in the image this cell belongs
 * to, so the separation is a plain subtraction. Only the fallback path, where
 * a cell is too wide for one image to cover it, still wraps per tracer.
 */
static inline void bin_cell_run(const bin_ctx_t* ctx, uint64_t p_start,
  uint64_t p_end, sif_real cx, sif_real cy, sif_real cz, sif_real r_max_sq,
  sif_real inv_bin, sif_real* local_mass, sif_real* local_vrad,
  uint64_t* local_count, const int do_wrap, const int want_dens,
  const int want_vel, const int have_w) {

  const sif_real* mx = ctx->x;
  const sif_real* my = ctx->y;
  const sif_real* mz = ctx->z;
  const sif_real box_length = ctx->box_length;
  const sif_real half_box = ctx->half_box;
  const uint32_t n_bins = ctx->n_bins;

  /* NO simd pragma here: `bin` below is data-dependent, so the accumulations
   * are a scatter. Asserting the loop is dependence-free let the vectorizer
   * drop updates whenever two lanes landed in the same bin, biasing every
   * profile low by up to ~20% in dense bins. The compiler still vectorizes the
   * distance arithmetic on its own. */
  for (uint64_t p = p_start; p < p_end; p++) {
    sif_real dx = mx[p] - cx;
    sif_real dy = my[p] - cy;
    sif_real dz = mz[p] - cz;

    if (do_wrap) {
      /* Branchless nearest-image wrap. */
      dx -= box_length * (sif_real)(dx > half_box) -
            box_length * (sif_real)(dx < -half_box);
      dy -= box_length * (sif_real)(dy > half_box) -
            box_length * (sif_real)(dy < -half_box);
      dz -= box_length * (sif_real)(dz > half_box) -
            box_length * (sif_real)(dz < -half_box);
    }

    const sif_real r2 = dx * dx + dy * dy + dz * dz;

    if (r2 <= r_max_sq) {
      const sif_real r = SIF_REAL_SQRT(r2);
      const uint32_t bin = (uint32_t)(r * inv_bin);

      if (bin < n_bins) {
        if (want_dens)
          local_mass[bin] += have_w ? ctx->w[p] : (sif_real)1.0;

        if (want_vel) {
          const sif_real inv_r =
            (r > (sif_real)0.0) ? (sif_real)1.0 / r : (sif_real)0.0;
          local_vrad[bin] +=
            (ctx->vx[p] * dx + ctx->vy[p] * dy + ctx->vz[p] * dz) * inv_r;
          local_count[bin]++;
        }
      }
    }
  }
}

/*
 * Places one cell of the walk: which cell of the mesh it is, where the void
 * centre sits relative to it, and how far the cell is from the sphere.
 *
 * The image is chosen per cell rather than per tracer, which is what lets the
 * kernel take the separation as a plain subtraction. The gap it returns -- the
 * distance from the centre to the cell's slab on this axis, zero when the
 * centre is inside it -- is a lower bound on the distance to anything the cell
 * holds, so squaring and summing the three gaps rejects a cell without reading
 * a single tracer.
 *
 * @p idx is off by at most one row, which is what sif_profiles()'s capped
 * range guarantees, so the wrap is an add rather than a division.
 */
static inline uint32_t cell_image(int32_t idx, sif_real centre,
  const bin_ctx_t* ctx, uint32_t n_cells, int is_periodic, sif_real* out_centre,
  sif_real* out_gap) {

  int32_t w = idx;
  if (is_periodic) {
    if (w < 0)
      w += (int32_t)n_cells;
    else if (w >= (int32_t)n_cells)
      w -= (int32_t)n_cells;
  }

  sif_real gap = ((sif_real)w + (sif_real)0.5) * ctx->cell_length - centre;
  sif_real image = centre;

  if (is_periodic) {
    const sif_real k = SIF_REAL_ROUND(gap * ctx->inv_box);
    gap -= k * ctx->box_length;
    image += k * ctx->box_length;
  }

  gap = SIF_REAL_ABS(gap) - ctx->half_cell;

  *out_centre = image;
  *out_gap = (gap > (sif_real)0.0) ? gap : (sif_real)0.0;
  return (uint32_t)w;
}

/*
 * Inclusive cell-index range the search sphere spans on one axis.
 *
 * A sphere that reaches around the box -- a coarse mesh, or a void far larger
 * than the one the resolution was chosen for -- spans more indices than the
 * mesh has cells, and the wrap in the walk would then map two of them onto the
 * same cell and bin its tracers twice. Capping the walk at one full row visits
 * every cell exactly once, which is what the minimum-image separation the
 * estimator takes assumes anyway.
 *
 * Computed in double and clamped before the cast: an r_max a few times the box,
 * which nothing here forbids, overflows an int32 index outright.
 */
static inline void axis_cell_range(sif_real centre, sif_real r_max,
  sif_real inv_cell_len, uint32_t n_cells, int is_periodic, int32_t* out_min,
  int32_t* out_max) {

  double lo = floor(((double)centre - (double)r_max) * (double)inv_cell_len);
  double hi = floor(((double)centre + (double)r_max) * (double)inv_cell_len);

  if (is_periodic) {
    if (hi - lo + 1.0 >= (double)n_cells) {
      lo = 0.0;
      hi = (double)n_cells - 1.0;
    }
  } else {
    /* Open boundary: everything outside the box is empty, so the range is
     * pruned here instead of cell by cell inside the walk. */
    if (lo < 0.0)
      lo = 0.0;
    if (hi > (double)n_cells - 1.0)
      hi = (double)n_cells - 1.0;
  }

  *out_min = (int32_t)lo;
  *out_max = (int32_t)hi;
}

/* Fills r_edges and, for densities, the volume each bin's weight is divided
 * by: the whole sphere out to the bin's outer edge for a cumulative profile,
 * the shell between its edges for a differential one. Edges are in units of
 * the void radius, so the volumes are too and get scaled by rv^3 per void at
 * the end. */
static void fill_edges(sif_real* r_edges, sif_real* bin_vols, uint32_t n_bins,
  sif_real ext, sif_real bin_width, bool differential) {

  for (uint32_t j = 0; j < n_bins; j++) {
    r_edges[j] = (sif_real)j * bin_width;

    if (bin_vols) {
      const sif_real r_outer = (sif_real)(j + 1) * bin_width;
      const sif_real r_inner = differential ? r_edges[j] : (sif_real)0.0;

      bin_vols[j] = (sif_real)(4.0 / 3.0) * SIF_PI *
                    (r_outer * r_outer * r_outer - r_inner * r_inner * r_inner);
    }
  }

  /* Assigned rather than accumulated, so the top edge is exactly the ext the
   * caller asked for. */
  r_edges[n_bins] = ext;
}

/* --- estimator --- */

int sif_profiles(const sif_catalog_t* cat, const sif_chain_mesh_t* mesh,
  sif_real ext, uint32_t n_bins, sif_option opt,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  int status = validate_inputs(cat, mesh, n_bins,
    (const sif_density_profiles_t* const*)out_dens,
    (const sif_velocity_profiles_t* const*)out_vel);
  if (status != SIF_OK)
    return status;

  /* An extension that is not a positive number -- zero, negative, or a NaN
   * that -ffast-math would let through a comparison either way -- is read as
   * "no preference" rather than rejected: there is one sensible reach for a
   * void profile and no reason to make every caller spell it out. */
  if (!(ext > (sif_real)0.0))
    ext = SIF_PROFILES_DEFAULT_EXT;

  const int compute_dens = (out_dens != NULL);
  const int compute_vel = (out_vel != NULL);

  const bool differential =
    ((opt & SIF__PROFILES_BIN_MASK) == SIF_PROFILES_DIFFERENTIAL);

  const sif_real box_length = mesh->box_length;

  SIF_LOG_INFO("profiles",
    "computing %s profiles for %" PRIu64 " voids out to %g void radii",
    differential ? "differential" : "cumulative", cat->n_voids, (double)ext);

  /* Allocated here only if the caller passed a pointer to NULL; a set it
   * already owns is refilled, which is how a sweep over several catalogues
   * avoids reallocating the same shape every time. */
  if (compute_dens && *out_dens == NULL) {
    *out_dens =
      sif__density_profiles_alloc(cat->n_voids, n_bins, ext, differential);
    if (!*out_dens) {
      SIF_LOG_ERROR("profiles", "OOM allocating density profiles");
      return SIF_ERR_ALLOC;
    }
  }

  if (compute_vel && *out_vel == NULL) {
    *out_vel = sif__velocity_profiles_alloc(cat->n_voids, n_bins, ext);
    if (!*out_vel) {
      SIF_LOG_ERROR("profiles", "OOM allocating velocity profiles");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
  }

  const sif_real bin_width = ext / (sif_real)n_bins;
  sif_real* bin_vols = NULL;

  if (compute_dens) {
    bin_vols = sif_malloc_aligned(n_bins * sizeof(sif_real));
    if (!bin_vols) {
      SIF_LOG_ERROR("profiles", "OOM allocating the bin volumes");
      status = SIF_ERR_ALLOC;
      goto fail;
    }
    fill_edges(
      (*out_dens)->r_edges, bin_vols, n_bins, ext, bin_width, differential);
  }

  if (compute_vel)
    fill_edges((*out_vel)->r_edges, NULL, n_bins, ext, bin_width, false);

  const uint32_t n_cells = mesh->n_cells;
  const sif_real half_box = box_length * (sif_real)0.5;
  const sif_real inv_cell_len = (sif_real)1.0 / mesh->cell_length;

  const sif_real mean_dens =
    compute_dens ? (sif_real)mean_density(mesh) : (sif_real)1.0;

  const int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const int have_weights = (mesh->weights != NULL);

  const bin_ctx_t ctx = {.x = SIF_ASSUME_ALIGNED(mesh->x),
    .y = SIF_ASSUME_ALIGNED(mesh->y),
    .z = SIF_ASSUME_ALIGNED(mesh->z),
    .w = have_weights ? SIF_ASSUME_ALIGNED(mesh->weights) : NULL,
    .vx = mesh->vx ? SIF_ASSUME_ALIGNED(mesh->vx) : NULL,
    .vy = mesh->vy ? SIF_ASSUME_ALIGNED(mesh->vy) : NULL,
    .vz = mesh->vz ? SIF_ASSUME_ALIGNED(mesh->vz) : NULL,
    .box_length = box_length,
    .half_box = half_box,
    .inv_box = (sif_real)1.0 / box_length,
    .cell_length = mesh->cell_length,
    .half_cell = mesh->cell_length * (sif_real)0.5,
    .n_bins = n_bins};

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
    sif_free_aligned(bin_vols);
    status = SIF_ERR_ALLOC;
    goto fail;
  }

/* The flags are literals in each expansion, so the compiler builds one
 * specialized loop per combination rather than testing them per tracer. */
#define SIF_BIN_CELL(wrap, dens, vel, wgt)                                     \
  bin_cell_run(&ctx, p_start, p_end, wrap ? cx : ecx, wrap ? cy : ecy,         \
    wrap ? cz : ecz, r_max_sq, inv_bin, local_mass, local_vrad, local_count,   \
    wrap, dens, vel, wgt)

#define SIF_BIN_DISPATCH(wrap)                                                 \
  do {                                                                         \
    if (compute_dens && compute_vel) {                                         \
      if (have_weights)                                                        \
        SIF_BIN_CELL(wrap, 1, 1, 1);                                           \
      else                                                                     \
        SIF_BIN_CELL(wrap, 1, 1, 0);                                           \
    } else if (compute_dens) {                                                 \
      if (have_weights)                                                        \
        SIF_BIN_CELL(wrap, 1, 0, 1);                                           \
      else                                                                     \
        SIF_BIN_CELL(wrap, 1, 0, 0);                                           \
    } else {                                                                   \
      SIF_BIN_CELL(wrap, 0, 1, 0);                                             \
    }                                                                          \
  } while (0)

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

    const sif_real r_max = rv * ext;
    const sif_real r_max_sq = r_max * r_max;

    /* One reciprocal per void instead of a division per accepted tracer. */
    const sif_real inv_bin = (sif_real)1.0 / (rv * bin_width);

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

    /* Whether the void centre can be placed once per cell instead of the
     * separation being wrapped once per tracer.
     *
     * Placing it per cell is exact as long as every tracer the walk reaches
     * lands within half a box of the image chosen for its cell, which holds
     * when the sphere plus one cell fits inside that half. It is the ordinary
     * case; a mesh coarse enough to break it -- a couple of cells across the
     * whole box -- keeps the per-tracer wrap. */
    const int per_cell_image =
      !is_periodic || (r_max + mesh->cell_length <= half_box);

    /* Bounding box indices, already capped at one full row and pruned to the
     * box, so the walk below only has to place them. */
    int32_t ix_min, ix_max, iy_min, iy_max, iz_min, iz_max;
    axis_cell_range(
      cx, r_max, inv_cell_len, n_cells, is_periodic, &ix_min, &ix_max);
    axis_cell_range(
      cy, r_max, inv_cell_len, n_cells, is_periodic, &iy_min, &iy_max);
    axis_cell_range(
      cz, r_max, inv_cell_len, n_cells, is_periodic, &iz_min, &iz_max);

    for (int32_t ix = ix_min; ix <= ix_max; ix++) {
      sif_real ecx, gap_x;
      const uint32_t w_ix =
        cell_image(ix, cx, &ctx, n_cells, is_periodic, &ecx, &gap_x);

      /* The sphere does not reach this slab, so nothing in it can be inside
       * r_max however the other two axes fall. */
      const sif_real gap_x2 = gap_x * gap_x;
      if (gap_x2 > r_max_sq)
        continue;

      for (int32_t iy = iy_min; iy <= iy_max; iy++) {
        sif_real ecy, gap_y;
        const uint32_t w_iy =
          cell_image(iy, cy, &ctx, n_cells, is_periodic, &ecy, &gap_y);

        const sif_real gap_xy2 = gap_x2 + gap_y * gap_y;
        if (gap_xy2 > r_max_sq)
          continue;

        for (int32_t iz = iz_min; iz <= iz_max; iz++) {
          sif_real ecz, gap_z;
          const uint32_t w_iz =
            cell_image(iz, cz, &ctx, n_cells, is_periodic, &ecz, &gap_z);

          /* The corner test that the bounding box cannot do: a cube's corners
           * hold about a third of its volume, and none of it is in the
           * sphere. */
          if (gap_xy2 + gap_z * gap_z > r_max_sq)
            continue;

          const uint64_t flat_idx = (uint64_t)w_ix * n_cells * n_cells +
                                    (uint64_t)w_iy * n_cells + w_iz;

          const uint64_t p_start = mesh->cell_offsets[flat_idx];
          const uint64_t p_end = mesh->cell_offsets[flat_idx + 1];

          if (p_start == p_end)
            continue;

          if (per_cell_image)
            SIF_BIN_DISPATCH(0);
          else
            SIF_BIN_DISPATCH(1);
        }
      }
    }

    /* A velocity profile is a shell mean whatever the density profile is; a
     * mean infall over everything inside a radius is not a quantity anyone
     * wants. */
    const sif_real rv_cubed = rv * rv * rv;
    sif_real cumulative_mass = (sif_real)0.0;

    for (uint32_t j = 0; j < n_bins; j++) {
      const uint64_t global_idx = i * n_bins + j;

      if (compute_dens) {
        cumulative_mass += local_mass[j];

        /* Weight over the volume it is spread through: the whole sphere out to
         * this bin's outer edge, or this shell alone. bin_vols is in units of
         * the void radius, so rv^3 puts it back in physical units. */
        const sif_real weight = differential ? local_mass[j] : cumulative_mass;
        const sif_real raw_rho = weight / (bin_vols[j] * rv_cubed);

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

#undef SIF_BIN_DISPATCH
#undef SIF_BIN_CELL

  sif_free_aligned(scratch_mass);
  sif_free_aligned(scratch_vrad);
  sif_free_aligned(scratch_count);
  sif_free_aligned(bin_vols);

  /* Stamped here rather than at the allocation, so that only a set that was
   * actually filled claims to have come from this catalogue -- and so that a
   * set the caller passed back to be refilled carries the catalogue it holds
   * now, not the one it held last time. */
  if (compute_dens)
    (*out_dens)->source_id = sif_catalog_id(cat);
  if (compute_vel)
    (*out_vel)->source_id = sif_catalog_id(cat);

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
