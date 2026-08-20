/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/tessellation.h"

#include "sif/structures/grid.h"

#include "core/system_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"
#include "structures/chain_mesh_internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * How many samples cell `c` receives, and where its run starts.
 *
 * The samples are spread over the cells as evenly as an integer count allows:
 * the first `n_samples % total_cells` cells take one extra. Written this way
 * rather than as `c * n_samples / total_cells` because that product overflows
 * a 64-bit integer for a large mesh, while `c * base` cannot -- it is bounded
 * by the sample count itself.
 */
static inline uint64_t cell_sample_count(
  uint64_t c, uint64_t base, uint64_t rem) {
  return base + (c < rem ? 1u : 0u);
}

static inline uint64_t cell_sample_offset(
  uint64_t c, uint64_t base, uint64_t rem) {
  return c * base + (c < rem ? c : rem);
}

void sif_tessellation_free(sif_tessellation_t* tess) {
  if (!tess)
    return;

  sif_free_aligned(tess->volumes);
  sif_field_free(tess->samples);
  free(tess);
}

sif_tessellation_t* sif_tessellation_alloc(const sif_chain_mesh_t* mesh,
  uint32_t samples_per_tracer, uint64_t seed, sif_option opt) {

  if (!mesh || mesh->n_particles == 0 || !(mesh->box_length > (sif_real)0.0)) {
    SIF_LOG_ERROR("tessellation", "invalid or empty chain mesh");
    return NULL;
  }

  if (samples_per_tracer == 0) {
    SIF_LOG_ERROR("tessellation",
      "samples_per_tracer must be non-zero; a tessellation nothing was thrown "
      "at has no volumes to report");
    return NULL;
  }

  const uint64_t n_tracers = mesh->n_particles;
  const uint64_t n_samples = n_tracers * (uint64_t)samples_per_tracer;

  /* The sample count multiplies the field, so it is the one number here that
   * can turn a reasonable field into an impossible allocation. */
  if (n_samples / (uint64_t)samples_per_tracer != n_tracers) {
    SIF_LOG_ERROR("tessellation",
      "%" PRIu64 " tracers at %u samples each overflows a sample count",
      n_tracers, samples_per_tracer);
    return NULL;
  }

  /* One sample per cell is the finest the stratification below can go: it
   * hands cell c the samples in [c * base, ...), so once there are fewer
   * samples than cells every cell takes either one or none, and "the first
   * n_samples cells" is not a subset of the box -- the flat index runs x
   * fastest, so it is a pencil along one edge. The samples would cover a
   * corner of the box and the volumes would be meaningless, while still
   * summing to the box exactly, since that sum is an identity rather than a
   * measurement. Refuse instead. */
  if (n_samples < mesh->total_cells) {
    const uint64_t needed =
      (mesh->total_cells + n_tracers - 1) / n_tracers; /* ceil */

    SIF_LOG_ERROR("tessellation",
      "%" PRIu64 " samples cannot be stratified over a %" PRIu64
      "-cell mesh. Either raise samples_per_tracer to at least %" PRIu64
      ", or tessellate on a mesh sized for the tracers rather than this one",
      n_samples, mesh->total_cells, needed);
    return NULL;
  }

  sif_tessellation_t* tess = calloc(1, sizeof(sif_tessellation_t));
  if (!tess)
    return NULL;

  tess->n_tracers = n_tracers;
  tess->n_samples = n_samples;
  tess->n_cells = mesh->n_cells;
  tess->box_length = mesh->box_length;

  const double box = (double)mesh->box_length;
  tess->sample_volume = (sif_real)(box * box * box / (double)n_samples);

  tess->volumes = sif_calloc_aligned(n_tracers, sizeof(sif_real));
  tess->samples = sif_field_alloc(n_samples);

  if (!tess->volumes || !tess->samples) {
    SIF_LOG_ERROR("tessellation", "OOM allocating the tessellation");
    sif_tessellation_free(tess);
    return NULL;
  }

  /* Weights always, whether or not the tracers carried any: a sample stands
   * for its owner's share of the box, and that share is never 1. */
  if (sif_field_reserve_positions(tess->samples) != SIF_OK ||
      sif_field_reserve_weights(tess->samples) != SIF_OK ||
      (mesh->vx && sif_field_reserve_velocities(tess->samples) != SIF_OK)) {
    SIF_LOG_ERROR("tessellation",
      "OOM allocating %" PRIu64 " samples (%.2f GiB)", n_samples,
      (double)(n_samples * (mesh->vx ? 7 : 4) * sizeof(sif_real)) /
        (1024.0 * 1024.0 * 1024.0));
    sif_tessellation_free(tess);
    return NULL;
  }

  /* Which tracer owns each sample, kept only until the weights are worked out.
   * A slot rather than a field index, so it reads the mesh's payload directly
   * and a mesh built with SIF_MESH_DROP_INDICES still works. */
  uint64_t* owners = sif_malloc_aligned(n_samples * sizeof(uint64_t));
  uint64_t* counts = sif_calloc_aligned(n_tracers, sizeof(uint64_t));

  if (!owners || !counts) {
    SIF_LOG_ERROR("tessellation", "OOM allocating the ownership pass");
    sif_free_aligned(owners);
    sif_free_aligned(counts);
    sif_tessellation_free(tess);
    return NULL;
  }

  const int is_periodic = ((opt & SIF__PBC_MASK) == SIF_PBC_PERIODIC);

  const uint64_t total_cells = mesh->total_cells;
  const uint64_t base = n_samples / total_cells;
  const uint64_t rem = n_samples % total_cells;
  const uint32_t n_cells = mesh->n_cells;
  const sif_real cell_length = mesh->cell_length;
  const sif_real box_length = mesh->box_length;

  sif_real* sx = tess->samples->x;
  sif_real* sy = tess->samples->y;
  sif_real* sz = tess->samples->z;

  SIF_LOG_INFO("tessellation",
    "throwing %" PRIu64 " samples at %" PRIu64 " tracers (%u per tracer)",
    n_samples, n_tracers, samples_per_tracer);

  /* --- ownership pass --- */
#pragma omp parallel for schedule(static)
  for (uint64_t c = 0; c < total_cells; c++) {
    const uint64_t n_c = cell_sample_count(c, base, rem);
    if (n_c == 0)
      continue;

    const uint64_t off = cell_sample_offset(c, base, rem);

    /* Seeded from the cell rather than from the thread, so the samples are the
     * same however the loop is divided up. */
    sif_prng_state_t rng;
    sif_prng_init(&rng, seed + c * 0x9e3779b97f4a7c15ULL);

    const uint64_t ix = c / ((uint64_t)n_cells * n_cells);
    const uint64_t iy = (c / n_cells) % n_cells;
    const uint64_t iz = c % n_cells;

    const sif_real x0 = (sif_real)ix * cell_length;
    const sif_real y0 = (sif_real)iy * cell_length;
    const sif_real z0 = (sif_real)iz * cell_length;

    for (uint64_t k = 0; k < n_c; k++) {
      sif_real px = x0 + sif_prng_next_real(&rng) * cell_length;
      sif_real py = y0 + sif_prng_next_real(&rng) * cell_length;
      sif_real pz = z0 + sif_prng_next_real(&rng) * cell_length;

      /* The mesh is anchored at the origin and rejects anything at or past the
       * far face, which rounding can otherwise produce in the last cell. */
      if (!(px < box_length))
        px = box_length * (sif_real)0.999999f;
      if (!(py < box_length))
        py = box_length * (sif_real)0.999999f;
      if (!(pz < box_length))
        pz = box_length * (sif_real)0.999999f;

      const uint64_t owner =
        is_periodic ? sif__chain_mesh_find_nearest_slot_pbc(mesh, px, py, pz)
                    : sif__chain_mesh_find_nearest_slot_open(mesh, px, py, pz);

      sx[off + k] = px;
      sy[off + k] = py;
      sz[off + k] = pz;
      owners[off + k] = owner;

      if (owner != UINT64_MAX) {
#pragma omp atomic
        counts[owner]++;
      }
    }
  }

  /* --- volumes, and the weight each sample carries --- */
  uint64_t n_empty = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_empty)
  for (uint64_t i = 0; i < n_tracers; i++) {
    tess->volumes[i] =
      (sif_real)((double)counts[i] * (double)tess->sample_volume);
    if (counts[i] == 0)
      n_empty++;
  }

  tess->n_empty = n_empty;

  sif_real* sw = tess->samples->weights;
  const sif_real* mw = mesh->weights;

#pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < n_samples; j++) {
    const uint64_t owner = owners[j];

    if (owner == UINT64_MAX) {
      sw[j] = (sif_real)0.0;
      continue;
    }

    /* The owner's weight, split between the samples it caught. Summed over
     * every sample this gives the tracer weight back exactly, which is what
     * makes the mean density of the sample field the mean density of the
     * tracers. */
    const sif_real m = mw ? mw[owner] : (sif_real)1.0;
    sw[j] = m / (sif_real)counts[owner];
  }

  if (mesh->vx) {
    sif_real* svx = tess->samples->vx;
    sif_real* svy = tess->samples->vy;
    sif_real* svz = tess->samples->vz;

#pragma omp parallel for schedule(static)
    for (uint64_t j = 0; j < n_samples; j++) {
      const uint64_t owner = owners[j];

      /* Copied unchanged rather than shared out: a velocity is not extensive,
       * so every sample of a cell moves at the cell's velocity. Averaging over
       * samples then weights by volume, which is the point. */
      svx[j] = (owner == UINT64_MAX) ? (sif_real)0.0 : mesh->vx[owner];
      svy[j] = (owner == UINT64_MAX) ? (sif_real)0.0 : mesh->vy[owner];
      svz[j] = (owner == UINT64_MAX) ? (sif_real)0.0 : mesh->vz[owner];
    }
  }

  sif_free_aligned(owners);
  sif_free_aligned(counts);

  if (n_empty > 0) {
    /* What the empties actually cost is the weight they take with them, so
     * that is what decides whether this is worth interrupting for: a handful
     * of tiny cells out of millions moves nothing, and warning about it every
     * run would only teach the reader to ignore the warning that matters. */
    const double missing = (double)n_empty / (double)n_tracers;

    if (missing > 1e-3) {
      SIF_LOG_WARNING("tessellation",
        "%" PRIu64 " of %" PRIu64
        " tracers (%.2f%%) caught no sample; their volume reads zero and "
        "their weight is missing from the samples, which shifts every density "
        "normalized to it. Raise samples_per_tracer above %u",
        n_empty, n_tracers, 100.0 * missing, samples_per_tracer);
    } else {
      SIF_LOG_INFO("tessellation",
        "%" PRIu64 " of %" PRIu64
        " tracers (%.4f%%) caught no sample; too little weight to matter, but "
        "their volume reads zero",
        n_empty, n_tracers, 100.0 * missing);
    }
  }

  SIF_LOG_INFO("tessellation", "tessellation complete");
  return tess;
}

/*
 * Both mesh constructors want the same complaint about the same bad inputs.
 */
static int mesh_inputs_valid(const sif_tessellation_t* tess, uint32_t n_cells) {
  if (!tess || !tess->samples) {
    SIF_LOG_ERROR("tessellation", "invalid tessellation");
    return 0;
  }

  if (tess->samples->n_particles == 0) {
    SIF_LOG_ERROR("tessellation",
      "these samples have already been consumed by "
      "sif_chain_mesh_alloc_tessellation_consume()");
    return 0;
  }

  if (n_cells == 0) {
    SIF_LOG_ERROR("tessellation", "n_cells must be non-zero");
    return 0;
  }

  return 1;
}

sif_chain_mesh_t* sif_chain_mesh_alloc_tessellation(
  const sif_tessellation_t* tess, uint32_t n_cells, sif_option opt) {

  if (!mesh_inputs_valid(tess, n_cells))
    return NULL;

  return sif_chain_mesh_alloc(n_cells, tess->box_length, tess->samples, opt);
}

sif_chain_mesh_t* sif_chain_mesh_alloc_tessellation_consume(
  sif_tessellation_t* tess, uint32_t n_cells, sif_option opt) {

  if (!mesh_inputs_valid(tess, n_cells))
    return NULL;

  return sif_chain_mesh_alloc_consume(
    n_cells, tess->box_length, tess->samples, opt);
}

void sif_grid_assign_cic_tessellation(
  sif_grid_t* grid, const sif_tessellation_t* tess) {

  if (!grid || !tess || !tess->samples) {
    SIF_LOG_ERROR("tessellation", "invalid grid or tessellation");
    return;
  }

  if (tess->samples->n_particles == 0) {
    SIF_LOG_ERROR("tessellation",
      "these samples have already been consumed by "
      "sif_chain_mesh_alloc_tessellation_consume(); there is nothing left to "
      "deposit");
    return;
  }

  /* The grid and the tessellation each carry their own box, and a mismatch
   * would not fail -- the samples all lie inside either one -- it would just
   * put the density on the wrong scale. */
  if (grid->box_length != tess->box_length) {
    SIF_LOG_ERROR("tessellation",
      "the grid spans a box of %g but the tessellation was built in one of %g",
      (double)grid->box_length, (double)tess->box_length);
    return;
  }

  /* The samples are a field, so this is the ordinary assignment: what makes
   * the result a tessellation density is the weights they carry, not the way
   * they are deposited. */
  sif_grid_assign_cic(grid, tess->samples);
}
