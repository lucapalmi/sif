/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The survey half of the exodus finder: the footprint the randoms mark out,
 * the field scanned against them, how much of each void was observed, and the
 * survey entry points, which check what a survey run needs and hand it to the
 * same driver as a box. Why nothing else has to know the survey is not
 * periodic, see survey_check_margin().
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/chain_mesh.h"
#include "sif/utils/logger.h"

#include "core/system_internal.h"
#include "structures/field_internal.h"
#include "utils/logger_internal.h"

#include "exodus_internal.h"
#include "utils.h"

#define TAG "exodus_finder"

/*
 * Marks every grid cell that holds at least one random. Each random lands in
 * exactly one cell -- no assignment window -- since the question is whether
 * that piece of space was observed, not how much was expected there.
 *
 * "Cell" here is the one around a grid node, the node nearest the random:
 * that is how a candidate's centre is placed (on node i, at i * cell_length),
 * so a candidate is inside exactly when its own node is.
 */
static inline uint32_t survey_node(sif_real p, sif_real inv_l, uint32_t n) {
  const uint32_t i = (uint32_t)(p * inv_l + 0.5f);
  return i < n ? i : n - 1;
}

/*
 * The closing that fills the footprint's chance holes, see
 * sif__exodus_survey_occupancy(). A node inside the footprint stays unobserved
 * only when a whole window next to it drew no random, about W exp(-W lambda)
 * for a window of W nodes at lambda randoms a node. The window is the smallest
 * that expects CLOSING_MIN_RANDOMS at the footprint's mean density, which
 * leaves fewer than one node in 10^4 unobserved where the selection is ten
 * times sparser than the mean. CLOSING_MAX_RADIUS bounds it, since a closing
 * over 2k + 1 nodes also fills real gaps up to 2k nodes wide.
 */
#define CLOSING_MIN_RANDOMS 128.0
#define CLOSING_MAX_RADIUS  4

/*
 * Randoms per node over the footprint, from m, the randoms per marked node.
 * The marked nodes miss the ones that drew no random, so for Poisson counts
 * m = lambda / (1 - exp(-lambda)), which overstates a sparse catalogue; this
 * solves it back for lambda.
 */
static double survey_randoms_per_node(double m) {
  if (!(m > 1.0))
    return 0.0;
  double lo = 0.0, hi = m;
  for (int i = 0; i < 64; i++) {
    const double mid = 0.5 * (lo + hi);
    if (mid < m * (1.0 - exp(-mid)))
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

/*
 * One axis of a box dilation (`erode` clear) or erosion (set) of `src` into
 * `dst`, over a window of 2k + 1 nodes along `axis`. Nodes past the grid's
 * faces count as unobserved both ways, so the closing never reaches past the
 * randoms' outermost nodes and the padding the survey box was sized for still
 * holds.
 */
static void survey_close_pass(const sif_bitmask_t* src, sif_bitmask_t* dst,
  uint32_t n, int axis, int32_t k, int erode) {

  const uint64_t n2 = (uint64_t)n * n;
  const uint64_t stride = axis == 0 ? n2 : axis == 1 ? n : 1;
  /* The other two axes, slower first, that pick out one line. */
  const uint64_t step_a = axis == 0 ? n : n2;
  const uint64_t step_b = axis == 2 ? n : 1;
  const int32_t len = (int32_t)n;
  const int32_t full = 2 * k + 1;

  sif_bitmask_clear_all(dst);

#pragma omp parallel for schedule(static)
  for (uint64_t line = 0; line < n2; line++) {
    const uint64_t base = (line / n) * step_a + (line % n) * step_b;

    /* Set nodes in the window [i - k, i + k], here for i = -1. */
    int32_t count = 0;
    for (int32_t j = 0; j < k && j < len; j++)
      count += sif_bitmask_get(src, base + (uint64_t)j * stride);

    for (int32_t i = 0; i < len; i++) {
      if (i + k < len)
        count += sif_bitmask_get(src, base + (uint64_t)(i + k) * stride);
      if (i - k - 1 >= 0)
        count -= sif_bitmask_get(src, base + (uint64_t)(i - k - 1) * stride);
      if (erode ? count == full : count > 0)
        sif_bitmask_set_atomic(dst, base + (uint64_t)i * stride);
    }
  }
}

/*
 * Closes the occupancy over a box of 2k + 1 nodes: dilate, then erode. A box
 * is the product of its sides, so each is three passes along lines rather
 * than one over a cube, and the two masks take turns as source and
 * destination, the result landing back in `occupied`.
 */
static int survey_close(sif_bitmask_t* occupied, uint32_t n, int32_t k) {
  sif_bitmask_t* tmp = sif_bitmask_alloc(occupied->n_bits);
  if (!tmp)
    return SIF_ERR_ALLOC;

  survey_close_pass(occupied, tmp, n, 0, k, 0);
  survey_close_pass(tmp, occupied, n, 1, k, 0);
  survey_close_pass(occupied, tmp, n, 2, k, 0);
  survey_close_pass(tmp, occupied, n, 0, k, 1);
  survey_close_pass(occupied, tmp, n, 1, k, 1);
  survey_close_pass(tmp, occupied, n, 2, k, 1);

  sif_bitmask_free(tmp);
  return SIF_OK;
}

/*
 * The footprint at grid resolution: the nodes the randoms reach, with the
 * holes they leave by chance filled in.
 *
 * A grid fine enough to search on puts only a few randoms in each cell --
 * sif-exodus's default is a cell per mean data separation, so about
 * N_random / N_data of them, and fewer where the selection falls off -- and
 * a cell that draws none is a hole the survey never had. Marked as it is, a
 * void well inside the footprint reads as partly unobserved, and a candidate
 * centred in such a cell is dropped. Closing the mask over a window that
 * expects enough randoms (CLOSING_MIN_RANDOMS) fills them, at the price of
 * also filling real gaps narrower than the window, which the smoothed fields
 * do not resolve anyway; the edge stays where the randoms put it.
 */
int sif__exodus_survey_occupancy(sif_bitmask_t** occupied,
  const sif_chain_mesh_t* rm, const sif_grid_t* grid) {

  sif_bitmask_t* occ = *occupied = sif_bitmask_alloc(grid->total_cells);
  if (!occ)
    return SIF_ERR_ALLOC;

  const uint32_t n = grid->n_cells;
  const sif_real inv_l = 1.0f / grid->cell_length;

#pragma omp parallel for schedule(static)
  for (uint64_t p = 0; p < rm->n_particles; p++) {
    const uint32_t ix = survey_node(rm->x[p], inv_l, n);
    const uint32_t iy = survey_node(rm->y[p], inv_l, n);
    const uint32_t iz = survey_node(rm->z[p], inv_l, n);

    sif_bitmask_set_atomic(occ, sif__flat_index(n, ix, iy, iz));
  }

  const uint64_t marked = sif_bitmask_count_set(occ);
  if (marked == 0)
    return SIF_OK;

  const double lambda =
    survey_randoms_per_node((double)rm->n_particles / (double)marked);
  int32_t k = 1;
  while (k < CLOSING_MAX_RADIUS &&
         lambda * (2 * k + 1) * (2 * k + 1) * (2 * k + 1) < CLOSING_MIN_RANDOMS)
    k++;
  const double window = (double)(2 * k + 1) * (2 * k + 1) * (2 * k + 1);

  if (survey_close(occ, n, k) != SIF_OK)
    return SIF_ERR_ALLOC;

  const uint64_t observed = sif_bitmask_count_set(occ);
  SIF_LOG_TRACE(TAG,
    "footprint: %.3g randoms per grid cell, closed over %d^3 cells: %" PRIu64
    " cells reached by a random, %" PRIu64 " observed",
    lambda, 2 * k + 1, marked, observed);

  if (lambda * window < CLOSING_MIN_RANDOMS)
    SIF_LOG_WARNING(TAG,
      "randoms: %.3g per grid cell over the footprint, too sparse to fill its "
      "chance holes (expected at least %.3g): footprints may read below 1 "
      "inside the survey; use more randoms or a coarser grid",
      lambda, CLOSING_MIN_RANDOMS / window);

  return SIF_OK;
}

/*
 * What an unobserved cell reads as on the scanned field: far above any
 * threshold, so it is never a candidate, and the value the minimum scan knows
 * to keep out of its smoothing.
 */
#define SURVEY_OUTSIDE SIF__FINDER_UNOBSERVED

/*
 * The field the survey finder scans, written over the smoothed data grid:
 * the smoothed data against the smoothed randoms, delta = rho_d / (alpha rho_r)
 * - 1. Smoothing the two separately and dividing afterwards is what keeps the
 * edges honest -- a sphere that straddles the footprint is measured against
 * the randoms that fall in it, not against the box mean, so unobserved space
 * reads as missing rather than as average.
 */
void sif__exodus_survey_contrast(sif_real* data, const sif_real* randoms,
  const sif_bitmask_t* occupied, sif_real alpha, uint64_t n_cells_total) {

#pragma omp parallel for schedule(static)
  for (uint64_t c = 0; c < n_cells_total; c++) {
    const sif_real expected = alpha * randoms[c];
    data[c] = (sif_bitmask_get(occupied, c) && expected > 0.0f)
                ? data[c] / expected - 1.0f
                : SURVEY_OUTSIDE;
  }
}

/*
 * How much of a void was observed: the fraction of its sphere, and of the
 * shell out to twice its radius, that falls in occupied cells.
 *
 * Measured on a lattice of points through the cube around the void rather
 * than cell by cell, so a void a few cells across still gets a fraction finer
 * than whole cells. The spacing is half a cell for small voids and a
 * sixteenth of the diameter for large ones, which caps the work at about 32k
 * points a void whatever its size. A point outside the box is outside the
 * footprint.
 */
static void survey_void_footprint(const sif_bitmask_t* occupied,
  const sif_grid_t* grid, sif_real cx, sif_real cy, sif_real cz, sif_real r,
  sif_real* inside, sif_real* shell) {

  const uint32_t n = grid->n_cells;
  const sif_real l = grid->cell_length;
  const sif_real inv_l = 1.0f / l;
  const sif_real box = grid->box_length;

  sif_real h = 0.125f * r;
  if (h < 0.5f * l)
    h = 0.5f * l;

  const int32_t half = (int32_t)(2.0f * r / h) + 1;
  const sif_real r2 = r * r;
  const sif_real r2_out = 4.0f * r2;

  uint64_t n_in = 0, occ_in = 0, n_sh = 0, occ_sh = 0;

  for (int32_t a = -half; a < half; a++) {
    const sif_real dx = ((sif_real)a + 0.5f) * h;
    for (int32_t b = -half; b < half; b++) {
      const sif_real dy = ((sif_real)b + 0.5f) * h;
      for (int32_t c = -half; c < half; c++) {
        const sif_real dz = ((sif_real)c + 0.5f) * h;
        const sif_real d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2_out)
          continue;

        const sif_real px = cx + dx, py = cy + dy, pz = cz + dz;
        int occ = 0;
        if (px >= 0.0f && px < box && py >= 0.0f && py < box && pz >= 0.0f &&
            pz < box) {
          occ = sif_bitmask_get(
            occupied, sif__flat_index(n, survey_node(px, inv_l, n),
                        survey_node(py, inv_l, n), survey_node(pz, inv_l, n)));
        }

        if (d2 <= r2) {
          n_in++;
          occ_in += (uint64_t)occ;
        } else {
          n_sh++;
          occ_sh += (uint64_t)occ;
        }
      }
    }
  }

  *inside = n_in ? (sif_real)((double)occ_in / (double)n_in) : 0.0f;
  *shell = n_sh ? (sif_real)((double)occ_sh / (double)n_sh) : 0.0f;
}

/* Every void's footprint, once the catalogue is final. */
int sif__exodus_survey_footprint(
  sif_catalogue_t* cat, const sif_bitmask_t* occupied, const sif_grid_t* grid) {

  if (sif_catalogue_reserve_footprint(cat) != SIF_OK)
    return SIF_ERR_ALLOC;

#pragma omp parallel for schedule(dynamic, 16)
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    survey_void_footprint(occupied, grid, cat->cx[i], cat->cy[i], cat->cz[i],
      cat->radii[i], &cat->footprint[i], &cat->footprint_shell[i]);
  }

  return SIF_OK;
}

/* The bounding box of a mesh's tracers. */
static void mesh_extent(
  const sif_chain_mesh_t* mesh, sif_real lo[3], sif_real hi[3]) {

  sif_real x0 = mesh->box_length, y0 = x0, z0 = x0;
  sif_real x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;

#pragma omp parallel for schedule(static) reduction(min : x0, y0, z0)          \
  reduction(max : x1, y1, z1)
  for (uint64_t p = 0; p < mesh->n_particles; p++) {
    x0 = SIF_REAL_MIN(x0, mesh->x[p]);
    y0 = SIF_REAL_MIN(y0, mesh->y[p]);
    z0 = SIF_REAL_MIN(z0, mesh->z[p]);
    x1 = SIF_REAL_MAX(x1, mesh->x[p]);
    y1 = SIF_REAL_MAX(y1, mesh->y[p]);
    z1 = SIF_REAL_MAX(z1, mesh->z[p]);
  }

  lo[0] = x0, lo[1] = y0, lo[2] = z0;
  hi[0] = x1, hi[1] = y1, hi[2] = z1;
}

/*
 * The one geometric promise a survey run needs from its caller: every tracer
 * at least `margin` from every face of the box.
 *
 * Everything downstream of the grid is periodic -- the FFT, the mesh walk, the
 * overlap tests -- and the survey finder uses it all unchanged, because with
 * this margin in place the periodicity never shows. Checked rather than
 * assumed, since a survey too close to the edge produces voids measured
 * against its own far side with nothing to say so. What the margin has to
 * cover, with m the padding on each side:
 *
 *   - Two voids never meet through the wrap. Their spheres are at most
 *     r_search across and their centres at least m from the faces, so the
 *     wrapped separation is at least 2m: m >= r_search.
 *   - A sphere walk that reaches past a face finds only empty padding. The
 *     walk overshoots r_search by up to a mesh cell, so it can reach
 *     r_search + cell - m past the face, into the far side's padding of m:
 *     m >= (r_search + cell) / 2. The FFT's top-hat, never wider than the
 *     largest rung, is covered by either.
 *   - A candidate's node is observed, so it is within half a grid cell of a
 *     random, and a centre refined off its node is within another half cell
 *     of that: plus a grid cell.
 */
static int survey_check_margin(const sif_chain_mesh_t* mesh, const char* what,
  sif_real margin, const char* hint) {

  sif_real lo[3], hi[3];
  mesh_extent(mesh, lo, hi);

  const sif_real box = mesh->box_length;
  for (int k = 0; k < 3; k++) {
    if (lo[k] < margin || hi[k] > box - margin) {
      SIF_LOG_ERROR(TAG,
        "%s: span [%g, %g] on axis %d of a box of %g, %g from its face "
        "(needs %g of padding on every side%s)",
        what, (double)lo[k], (double)hi[k], k, (double)box,
        (double)SIF_REAL_MIN(lo[k], box - hi[k]), (double)margin, hint);
      return SIF_ERR_RANGE;
    }
  }

  return SIF_OK;
}

sif_catalogue_t* sif_finder_exodus_survey(sif_grid_t* data_grid,
  sif_grid_t* random_grid, const sif_chain_mesh_t* data_mesh,
  const sif_chain_mesh_t* random_mesh, const sif_real offset[3],
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_exodus_variant_t variant, const sif_finder_exodus_extras_t* extras) {

  sif_real overlap_fraction;
  sif_option options;
  if (sif_finder_exodus_preset(variant, &overlap_fraction, &options) !=
      SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  return sif_finder_exodus_survey_raw(data_grid, random_grid, data_mesh,
    random_mesh, offset, radii, n_radii, threshold, overlap_fraction, options,
    extras);
}

sif_catalogue_t* sif_finder_exodus_survey_raw(sif_grid_t* data_grid,
  sif_grid_t* random_grid, const sif_chain_mesh_t* data_mesh,
  const sif_chain_mesh_t* random_mesh, const sif_real offset[3],
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option options,
  const sif_finder_exodus_extras_t* extras) {

  const char* missing = !data_grid             ? "data grid: NULL"
                        : !data_grid->values   ? "data grid: no values"
                        : !random_grid         ? "random grid: NULL"
                        : !random_grid->values ? "random grid: no values"
                        : !data_mesh           ? "data mesh: NULL"
                        : !random_mesh         ? "random mesh: NULL"
                        : !data_mesh->x || data_mesh->n_particles == 0
                          ? "data mesh: no particles"
                        : !random_mesh->x || random_mesh->n_particles == 0
                          ? "random mesh: no particles"
                          : NULL;
  if (missing) {
    SIF_LOG_ERROR(TAG, "%s", missing);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (sif__finder_check_args(
        TAG, radii, n_radii, threshold, overlap_fraction) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (sif__exodus_check_seeds(options, 1) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  for (int k = 0; offset && k < 3; k++) {
    if (!isfinite(offset[k])) {
      SIF_LOG_ERROR(
        TAG, "offset[%d]: %g, expected finite", k, (double)offset[k]);
      sif__error_status_set(SIF_ERR_INVALID);
      return NULL;
    }
  }

  /* The contrast is formed here, against the randoms, so a grid that has
   * already been turned into one against the box mean is the wrong input. */
  if (data_grid->content == SIF_GRID_DENSITY_CONTRAST ||
      random_grid->content == SIF_GRID_DENSITY_CONTRAST) {
    SIF_LOG_ERROR(TAG,
      "%s grid: holds a density contrast; a survey takes the CIC densities "
      "as assign_cic leaves them",
      data_grid->content == SIF_GRID_DENSITY_CONTRAST ? "data" : "random");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (data_grid->n_cells != random_grid->n_cells ||
      data_grid->box_length != random_grid->box_length) {
    SIF_LOG_ERROR(TAG,
      "data grid: %u cells over %g, random grid: %u over %g (must match)",
      data_grid->n_cells, (double)data_grid->box_length, random_grid->n_cells,
      (double)random_grid->box_length);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (data_mesh->box_length != data_grid->box_length ||
      random_mesh->box_length != data_grid->box_length) {
    SIF_LOG_ERROR(TAG,
      "meshes: boxes of %g (data) and %g (randoms), grids: %g (must match)",
      (double)data_mesh->box_length, (double)random_mesh->box_length,
      (double)data_grid->box_length);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if ((data_mesh->weights && sif__exodus_check_weights(data_mesh) != SIF_OK) ||
      (random_mesh->weights &&
        sif__exodus_check_weights(random_mesh) != SIF_OK)) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (!(data_mesh->total_weight > 0.0) || !(random_mesh->total_weight > 0.0)) {
    SIF_LOG_ERROR(TAG, "%s mesh: total weight is zero",
      data_mesh->total_weight > 0.0 ? "random" : "data");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  const sif_real r_search = sif__exodus_max_search_radius(
    radii, n_radii, data_grid->cell_length, options);
  const sif_real mesh_cell =
    SIF_REAL_MAX(data_mesh->cell_length, random_mesh->cell_length);
  const sif_real margin =
    SIF_REAL_MAX(r_search, 0.5f * (r_search + mesh_cell)) +
    data_grid->cell_length;

  /* survey_box() sizes the padding for grid cells up to the smallest radius;
   * a coarser grid is the likeliest reason it falls short, so say so. */
  sif_real r_min = radii[0];
  for (uint32_t i = 1; i < n_radii; i++)
    r_min = SIF_REAL_MIN(r_min, radii[i]);
  char hint[192] = ", as survey_box sizes it";
  if (data_grid->cell_length > r_min)
    snprintf(hint, sizeof(hint),
      "; survey_box allows for grid cells up to the smallest radius, %g, and "
      "these are %g",
      (double)r_min, (double)data_grid->cell_length);

  if (survey_check_margin(random_mesh, "randoms", margin, hint) != SIF_OK ||
      survey_check_margin(data_mesh, "data", margin, hint) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  if (sif__exodus_check_extras(
        extras, options, offset, data_grid->box_length) != SIF_OK) {
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  sif_catalogue_t* cat =
    sif__exodus_run(data_grid, data_mesh, radii, n_radii, threshold,
      overlap_fraction, options, random_grid, random_mesh, offset, extras);

  /* Back to the caller's frame, so that a forgotten step cannot leave the
   * voids where the box put them. */
  if (cat && offset) {
    const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
    const int status = sif_catalogue_translate(cat, back);
    if (status != SIF_OK) {
      sif_catalogue_free(cat);
      sif__error_status_set(status);
      return NULL;
    }

    /* The prior voids went into the box and back out, and in floating point
     * x + o - o need not be x. They are the caller's own, so they come back
     * as given. */
    const sif_catalogue_t* prior = extras ? extras->prior : NULL;
    if (prior && prior->n_voids > 0) {
      const size_t bytes = (size_t)prior->n_voids * sizeof(sif_real);
      memcpy(cat->cx, prior->cx, bytes);
      memcpy(cat->cy, prior->cy, bytes);
      memcpy(cat->cz, prior->cz, bytes);
    }
  }
  return cat;
}

/*
 * Slack on the padding sif_finder_exodus_survey_box() hands out, so that the
 * rounding of every coordinate on its way into the box cannot put a tracer a
 * hair inside the margin the finder then insists on.
 */
#define SURVEY_BOX_SLACK 1.01f

/*
 * The bounding box of a field's positions, grown into lo and hi. fminf and
 * fmaxf pass over a NaN rather than propagate it, so a coordinate that is not
 * finite would otherwise leave the box looking fine; it is counted instead.
 */
static int survey_field_extent(
  const sif_field_t* field, const char* what, sif_real lo[3], sif_real hi[3]) {

  sif_real x0 = lo[0], y0 = lo[1], z0 = lo[2];
  sif_real x1 = hi[0], y1 = hi[1], z1 = hi[2];
  uint64_t bad = 0;

#pragma omp parallel for schedule(static) reduction(min : x0, y0, z0)          \
  reduction(max : x1, y1, z1) reduction(+ : bad)
  for (uint64_t p = 0; p < field->n_particles; p++) {
    const sif_real x = field->x[p], y = field->y[p], z = field->z[p];
    bad += !(isfinite(x) && isfinite(y) && isfinite(z));
    x0 = SIF_REAL_MIN(x0, x);
    y0 = SIF_REAL_MIN(y0, y);
    z0 = SIF_REAL_MIN(z0, z);
    x1 = SIF_REAL_MAX(x1, x);
    y1 = SIF_REAL_MAX(y1, y);
    z1 = SIF_REAL_MAX(z1, z);
  }

  if (bad) {
    SIF_LOG_ERROR(TAG,
      "%s: %" PRIu64 " of %" PRIu64 " with a coordinate that "
      "is not finite",
      what, bad, field->n_particles);
    return SIF_ERR_INVALID;
  }

  lo[0] = x0, lo[1] = y0, lo[2] = z0;
  hi[0] = x1, hi[1] = y1, hi[2] = z1;
  return SIF_OK;
}

int sif_finder_exodus_survey_box(const sif_field_t* data,
  const sif_field_t* randoms, const sif_real* radii, uint32_t n_radii,
  sif_option opt, sif_real offset[3], sif_real* box_length) {

  const char* fields[2] = {"data", "randoms"};
  const sif_field_t* given[2] = {data, randoms};
  for (int i = 0; i < 2; i++) {
    const sif_field_t* f = given[i];
    if (!f || !f->x || !f->y || !f->z || f->n_particles == 0) {
      SIF_LOG_ERROR(TAG, "%s: %s", fields[i],
        !f                    ? "NULL"
        : f->n_particles == 0 ? "no particles"
                              : "no positions");
      return SIF_ERR_INVALID;
    }
    if (sif__field_require_cartesian(f, TAG, fields[i]) != SIF_OK)
      return SIF_ERR_INVALID;
  }
  if (!radii || n_radii == 0 || !offset || !box_length) {
    SIF_LOG_ERROR(TAG, "%s",
      !radii || n_radii == 0 ? "radii: none given"
                             : "offset, box_length: NULL");
    return SIF_ERR_INVALID;
  }
  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f) || !isfinite(radii[i])) {
      SIF_LOG_ERROR(
        TAG, "radii[%u] = %g: not a positive radius", i, (double)radii[i]);
      return SIF_ERR_INVALID;
    }
  }

  /*
   * The finder checks the padding around the data as well as the randoms, and
   * the data are not bound to the randoms' bounding box: a sparse random
   * catalogue samples the edge of the footprint less finely than a dense
   * galaxy one, so the outermost galaxy can sit a few Mpc past the outermost
   * random. The box is sized around both.
   */
  sif_real lo[3] = {randoms->x[0], randoms->y[0], randoms->z[0]};
  sif_real hi[3] = {lo[0], lo[1], lo[2]};
  if (survey_field_extent(randoms, "randoms", lo, hi) != SIF_OK)
    return SIF_ERR_INVALID;

  const sif_real r_lo[3] = {lo[0], lo[1], lo[2]};
  const sif_real r_hi[3] = {hi[0], hi[1], hi[2]};
  if (survey_field_extent(data, "data", lo, hi) != SIF_OK)
    return SIF_ERR_INVALID;

  sif_real beyond = 0.0f;
  for (int k = 0; k < 3; k++)
    beyond =
      SIF_REAL_MAX(beyond, SIF_REAL_MAX(r_lo[k] - lo[k], hi[k] - r_hi[k]));
  if (beyond > 0.0f)
    SIF_LOG_WARNING(TAG,
      "the data reach %g past the randoms' bounding box; the box is sized "
      "around both, but tracers outside the randoms are outside the "
      "footprint",
      (double)beyond);

  sif_real extent = 0.0f;
  for (int k = 0; k < 3; k++)
    extent = SIF_REAL_MAX(extent, hi[k] - lo[k]);

  sif_real r_min = radii[0], r_max = radii[0];
  for (uint32_t i = 1; i < n_radii; i++) {
    r_min = SIF_REAL_MIN(r_min, radii[i]);
    r_max = SIF_REAL_MAX(r_max, radii[i]);
  }

  /*
   * The box has to leave, on every side, the margin survey_check_margin()
   * asks for:
   *
   *   margin = r_search + cell,  r_search = r_max + max((f - 1) r_max, 3 cell),
   *
   * with cell the grid's. The grid is chosen after the box, so the margin is
   * taken at the largest cell it allows for, the smallest radius: a grid any
   * coarser cannot resolve that radius at all. A finer grid gets more padding
   * than it needs, r_min + max(0, 3 r_min - (f - 1) r_max) at most -- r_min
   * for a ladder spanning a factor of 6 at f = 1.5, against a margin of at
   * least 1.5 r_max -- and nothing has to be solved for: the box no longer
   * depends on the grid.
   *
   * This assumes mesh cells no larger than the search sphere, which a mesh
   * sized for either finder always is; the finder checks again with the grid
   * and meshes it is actually given.
   */
  const sif_real f = (sif_real)sif__finder_search_factor(opt);
  const sif_real cell = r_min;
  const sif_real margin =
    r_max + SIF_REAL_MAX((f - 1.0f) * r_max, 3.0f * cell) + cell;
  const sif_real L = extent + 2.0f * SURVEY_BOX_SLACK * margin;

  /* Centred: the axis that set the box gets exactly the margin, the others
   * more. */
  for (int k = 0; k < 3; k++)
    offset[k] = 0.5f * L - 0.5f * (lo[k] + hi[k]);

  *box_length = L;

  SIF_LOG_INFO(TAG,
    "survey box: %g across (extent %g, margin %g per side), offset (%g, %g, "
    "%g)",
    (double)L, (double)extent, 0.5 * (double)(L - extent), (double)offset[0],
    (double)offset[1], (double)offset[2]);

  return SIF_OK;
}
