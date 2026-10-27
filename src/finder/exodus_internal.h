/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file exodus_internal.h
 * @brief What the three parts of the exodus finder share: the driver
 * (exodus_finder.c), the radius rescaling (exodus_rescale.c) and the survey
 * geometry (exodus_survey.c). Private to the library.
 */

#ifndef SIF__FINDER_EXODUS_INTERNAL_H
#define SIF__FINDER_EXODUS_INTERNAL_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/grid.h"

/*
 * Floor on the rescaled radius, as a fraction of the rung that found the void.
 * Fixed, so that a catalogue depends only on the grid, the radius ladder and
 * the threshold.
 *
 * At 1 it coincides with the detection limit rather than sitting below it. A
 * void of geometric size p registers at rung R only once p >= |t|^(1/3) R, and
 * its crossing radius is p * |t|^(-1/3) -- so a cell is a candidate at R
 * exactly when its crossing radius is at least R. A rung can therefore never
 * legitimately return less than it, and any floor under R fences off a range
 * that only measurement error can reach: a top-hat evaluated on a grid whose
 * cells are not small against R reads a little too deep, and the tracers then
 * decline to support the radius the grid implied.
 *
 * Rejecting those costs nothing, because the same void meets the next rung
 * down with a crossing radius comfortably inside its own bracket, and gets a
 * centre smoothed at its own scale into the bargain. What it buys is that a
 * radius in the catalogue is never one the grid and the tracers disagreed
 * about.
 *
 * The enclosed-density condition is the only thing that decides a radius;
 * nothing here or below asks how the tracers inside are arranged.
 */
#define RMIN_FACTOR 1.00f

/*
 * Refinement window.
 *
 * The histogram keeps counts, not distances, so the exact radius has to be
 * recovered from particles once the walk reaches the bin that holds it. A bin
 * is a far thinner shell than a mesh cell -- at r = 40 it is about 1/256 of
 * one -- so recovering it means visiting every cell the shell passes through
 * and recomputing every distance in them, then keeping the handful that landed
 * in the bin. Measured, that is 68k distances computed to keep 141.
 *
 * Adjacent bins are adjacent shells, so consecutive recoveries repeat almost
 * exactly the same traversal. Resolving a window of W neighbouring bins in one
 * pass therefore costs what resolving one costs -- W bins together are still
 * thinner than a cell, so the same cells are visited either way -- and spares
 * the repeats. Measured, it takes the recoveries per scan from 4.0 to 1.0.
 *
 * W is not a constant: the bin count adapts to the expected occupancy, bins are
 * uniform in d^2 so their radial thickness varies with d, and the annulus
 * geometry changes with every smoothing radius. What W has to cover is how many
 * bins the walk spans before it finds its answer, which is measured directly --
 * every successful scan reports its span -- and W is then set to a multiple of
 * the mean.
 *
 * Widening past that buys nothing and starts costing: the collected buffer
 * grows with W and has to be sorted, and eventually the window does get thick
 * enough to pull in extra cells.
 */
#define RESOLVE_MARGIN  2u /* W = margin x mean span */
#define RESOLVE_MIN     1u
#define RESOLVE_MAX     64u
#define RESOLVE_INITIAL 8u

/* A distance collected by the refinement on a weighted mesh, with the weight
 * that has to leave the enclosed total when the walk passes it. */
typedef struct {
  sif_real d2;
  sif_real w;
} refine_pair_t;

/*
 * A weighted mesh swaps both buffers for their weighted twins: the histogram
 * holds weight per bin instead of tracers per bin, and the refinement keeps
 * each distance's weight beside it. Only the pair a run needs is allocated, so
 * an unweighted run holds exactly what it always has.
 */
typedef struct {
  uint32_t* bins;        /* unweighted: tracers per bin */
  double* wbins;         /* weighted: weight per bin */
  double* rbins;         /* survey: random weight per bin, beside wbins */
  sif_real* refine;      /* unweighted: squared distances */
  refine_pair_t* pairs;  /* weighted: squared distances and their weights */
  uint32_t refine_count; /* entries in whichever of the two is in use */
  uint32_t refine_capacity;

  /* How far down the bins the walk had to reach before the answer turned up,
   * summed over the scans that found one. This is what sizes the refinement
   * window; see RESOLVE_* below. Accumulated per thread and folded in during
   * the sequential commit, so no synchronisation is needed. */
  uint64_t span_sum;
  uint64_t span_count;
} radial_scratch_t;

/* --- Mesh traversal template --- */

typedef struct {
  int32_t* dx;
  int32_t* dy;
  int32_t* dz;
  uint8_t* type;
  /* Squared distance bounds of the cell relative to the query centre. Kept so
   * the refinement pass can drop every cell that cannot reach into the bin it
   * is resolving, which turns that pass from a sphere scan into a shell scan.
   */
  sif_real* min_d2;
  sif_real* max_d2;
  uint32_t count;
} mesh_template_t;

/* --- Radius rescaling --- */

/*
 * Everything a mesh traversal needs to turn a template entry into a particle
 * range. Both passes of the rescaling go through it, so they cannot disagree
 * about which particles a cell holds or which periodic image they are measured
 * against.
 */
typedef struct {
  const sif_chain_mesh_t* mesh;
  const mesh_template_t* tpl;
  sif_real cx, cy, cz;
  int32_t centre_ix, centre_iy, centre_iz;
  int32_t n_cells;
  sif_real box_length;
} mesh_query_t;

/*
 * Why a rescaling produced no radius.
 *
 * The failures are not variations on one theme, and a bare count of them says
 * nothing about what to change. A crossing below the rung is a void smaller
 * than this rung can express; a search sphere still underdense at its outer
 * edge holds one larger than the rung can reach. Those two ask for opposite
 * corrections to the radius ladder -- extend it down, extend it up -- so they
 * are counted apart.
 */
typedef enum {
  RESCALE_OK = 0,
  RESCALE_EMPTY,         /* not one tracer inside r_search */
  RESCALE_BEYOND_SEARCH, /* still underdense out at r_search */
  RESCALE_BELOW_RUNG,    /* the crossing lies below the rung itself */
  RESCALE_DEGENERATE,    /* r_search <= rmin, so there is no annulus */
  RESCALE_ALLOC,         /* the refinement buffer could not grow */
  RESCALE_NO_RANDOMS,    /* survey: no randoms inside r_search at all */
  RESCALE_N_REASONS
} rescale_reason_t;

/* Buckets of the r/R histogram; see RESCALE_BUCKET_EDGE. */
#define RESCALE_N_BUCKETS 6

/* What every rescaling at one rung did, successes and failures alike. */
typedef struct {
  uint64_t n_ok;
  uint64_t fail[RESCALE_N_REASONS];
  uint64_t bucket[RESCALE_N_BUCKETS];
  double ratio_sum;
  sif_real ratio_min;
  sif_real ratio_max;
} rescale_report_t;

/** @brief A thread's scratch, for a box run (`weighted` if the mesh is) or a
 * survey. SIF_OK or SIF_ERR_ALLOC; released with sif__exodus_scratch_free()
 * either way. */
SIF_NODISCARD int sif__exodus_scratch_init(
  radial_scratch_t* s, int weighted, int survey);
void sif__exodus_scratch_free(radial_scratch_t* s);

/** @brief The traversal of the [rmin, r_search] annulus about a centre
 * anywhere in its mesh cell. SIF_OK or SIF_ERR_ALLOC. */
SIF_NODISCARD int sif__exodus_template_build(mesh_template_t* tpl,
  uint32_t mesh_n_cells, sif_real cell_length, sif_real rmin,
  sif_real r_search);
void sif__exodus_template_free(mesh_template_t* tpl);

void sif__exodus_rescale_report_add(rescale_report_t* rep, sif_real ratio);
void sif__exodus_rescale_report_log(const rescale_report_t* rep);

/**
 * @brief The largest radius at which the enclosed number density is still at
 * or below (1 + threshold) times the mean, on an unweighted mesh; the weighted
 * form takes the mean weight density as `mass_factor`, and the survey form
 * measures the data against the randoms. -1 with `reason` set if there is
 * none.
 */
sif_real sif__exodus_find_radius(const sif_chain_mesh_t* mesh, sif_real cx,
  sif_real cy, sif_real cz, sif_real r_search, sif_real threshold,
  sif_real vol_factor, radial_scratch_t* scratch, sif_real rmin,
  const mesh_template_t* tpl, uint32_t window, uint8_t* reason);
sif_real sif__exodus_find_radius_weighted(const sif_chain_mesh_t* mesh,
  sif_real cx, sif_real cy, sif_real cz, sif_real r_search, sif_real threshold,
  sif_real vol_factor, sif_real mass_factor, radial_scratch_t* scratch,
  sif_real rmin, const mesh_template_t* tpl, uint32_t window, uint8_t* reason);
/**
 * @brief Whether sif__exodus_find_radius(), unweighted, would certainly return
 * a radius above sqrt(rho2), from counts alone; `closed` also asks that the
 * search sphere is not underdense out to r_search. `tpl` is the rescaling's,
 * and rho2 must be at least its core's rmin^2. 1 for proved, 0 for not
 * proved.
 */
int sif__exodus_witness(const sif_chain_mesh_t* mesh, sif_real cx, sif_real cy,
  sif_real cz, sif_real rho2, sif_real r_search, sif_real threshold,
  sif_real vol_factor, const mesh_template_t* tpl, int closed);
sif_real sif__exodus_find_radius_survey(const mesh_query_t* dq,
  const mesh_query_t* rq, sif_real r_search, sif_real K,
  radial_scratch_t* scratch, sif_real rmin, uint32_t window, uint8_t* reason);

/* The query a survey rescaling runs on one mesh. */
static inline mesh_query_t survey_query(const sif_chain_mesh_t* mesh,
  const mesh_template_t* tpl, sif_real cx, sif_real cy, sif_real cz) {

  const sif_real inv_l = 1.0f / mesh->cell_length;
  return (mesh_query_t){
    .mesh = mesh,
    .tpl = tpl,
    .cx = cx,
    .cy = cy,
    .cz = cz,
    .centre_ix = (int32_t)(cx * inv_l),
    .centre_iy = (int32_t)(cy * inv_l),
    .centre_iz = (int32_t)(cz * inv_l),
    .n_cells = (int32_t)mesh->n_cells,
    .box_length = mesh->box_length,
  };
}

/* --- Surveys --- */

/** @brief The observed grid cells, from the randoms, into `*occupied`
 * (allocated here, the caller's to free whatever is returned). SIF_OK or
 * SIF_ERR_ALLOC. */
SIF_NODISCARD int sif__exodus_survey_occupancy(sif_bitmask_t** occupied,
  const sif_chain_mesh_t* random_mesh, const sif_grid_t* grid);

/** @brief The scanned field, written over the smoothed data. */
void sif__exodus_survey_contrast(sif_real* data, const sif_real* randoms,
  const sif_bitmask_t* occupied, sif_real alpha, uint64_t n_cells_total);

/** @brief Every void's footprint, once the catalogue is final. SIF_OK or
 * SIF_ERR_ALLOC. */
SIF_NODISCARD int sif__exodus_survey_footprint(
  sif_catalogue_t* cat, const sif_bitmask_t* occupied, const sif_grid_t* grid);

/* --- Driver --- */

/** @brief A run, box or survey (a random grid and mesh given), once its
 * arguments are checked. */
sif_catalogue_t* sif__exodus_run(sif_grid_t* grid, const sif_chain_mesh_t* mesh,
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option options, sif_grid_t* random_grid,
  const sif_chain_mesh_t* random_mesh, const sif_real offset[3],
  const sif_finder_exodus_extras_t* extras);

/** @brief The argument checks a box and a survey run share, each logging
 * what it refuses: SIF_OK or SIF_ERR_INVALID. */
SIF_NODISCARD int sif__exodus_check_seeds(sif_option options, int survey);
SIF_NODISCARD int sif__exodus_check_extras(
  const sif_finder_exodus_extras_t* extras, sif_option options,
  const sif_real offset[3], sif_real box_length);
SIF_NODISCARD int sif__exodus_check_weights(const sif_chain_mesh_t* mesh);

/** @brief The largest search radius a run over this ladder can reach. */
sif_real sif__exodus_max_search_radius(const sif_real* radii, uint32_t n_radii,
  sif_real cell_length, sif_option options);

#endif /* SIF__FINDER_EXODUS_INTERNAL_H */
