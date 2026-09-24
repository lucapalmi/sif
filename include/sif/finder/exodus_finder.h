/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file exodus_finder.h
 * @brief Spherical void finder that grows each void to the radius its tracers
 * support.
 *
 * The plain spherical finder can only return radii from the ladder it was
 * given. This one uses the ladder to *locate* candidates and then walks the
 * enclosed density outward from each accepted centre, stopping where it
 * crosses the threshold, so the radius comes from the tracer distribution
 * rather than from the input grid.
 *
 * That means it needs the particles as well as the grid, which is what the
 * chain mesh is for.
 */

#ifndef SIF_FINDER_EXODUS_FINDER_H
#define SIF_FINDER_EXODUS_FINDER_H

#include <math.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

/**
 * @brief Run the exodus void finder on a density grid.
 *
 * The mesh is borrowed, never freed: build it once and it can be reused across
 * several runs at different thresholds or radius sets. Building it also means
 * the particle field is no longer needed here, so it can be released before
 * this call -- which matters, because the finder's own peak sits right on top
 * of whatever the caller is still holding.
 *
 * The finder reads the mesh positions and, when the mesh carries them, its
 * weights: a void's radius is then where the enclosed *weight* first reaches
 * (1 + threshold) times the mean weight density, the same criterion the grid
 * applies when it is built by sif_grid_assign_cic() from the same weighted
 * field. A mesh without weights is the unweighted finder, unchanged. A mesh
 * of a tessellation's samples has all of its density in the weights, so it
 * has to be run weighted to mean anything.
 *
 * Weights must be finite and non-negative; the call fails otherwise. The
 * enclosed weight has to grow with the radius for the search to be able to
 * rule whole shells out, and a negative weight would break that silently.
 *
 * Velocities are never read, and at these particle counts they are a
 * substantial allocation for nothing -- so hand this a field stripped of them
 * (and of weights, if the run is meant to be unweighted) when the mesh is
 * being built for the finder alone.
 *
 * @param grid Density contrast field, as produced by
 * sif_grid_to_density_contrast(). Smoothed in place and restored at the end,
 * as in sif_finder_spherical(); see the warning there about `grid->values`
 * being reassigned.
 * @param mesh The particle chain mesh. Must span the same box as the grid.
 * @param radii Array of smoothing radii. Each must be small enough that the
 * search sphere it implies (roughly twice the radius) fits inside the box.
 * @param n_radii Number of radii
 * @param threshold Density contrast a cell must be at or below to seed a
 * void. Negative, since voids are underdensities.
 * @param overlap_fraction How much two voids may overlap, as a fraction of
 * the smaller one's radius; see sif_finder_spherical().
 * @param opt Finder options. Honours SIF_FINDER_CONSUME_GRID,
 * SIF_FINDER_KEEP_CIC_WINDOW and SIF_FINDER_SEARCH_*. The box is always
 * periodic; a survey, with edges, is sif_finder_exodus_survey().
 *
 * @note The grid is assumed to have been built by sif_grid_assign_cic(), whose
 * window is divided back out before the first smoothing so that the top-hat is
 * the only window applied. Pass SIF_FINDER_KEEP_CIC_WINDOW for a grid that was
 * filled some other way.
 *
 * @return Newly allocated catalogue, released with sif_catalog_free(), or
 * NULL on invalid input -- a negative or non-finite weight included -- or
 * failure.
 */
SIF_NODISCARD sif_catalog_t* sif_finder_exodus(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option opt);

/**
 * @brief Run the exodus finder on a survey: any geometry, described by a
 * random catalogue.
 *
 * The same finder as sif_finder_exodus() -- same ladder, same speculation,
 * same overlap rules -- with the box mean replaced by the randoms. A sphere's
 * expected content is the random weight inside it times
 * alpha = W_data / W_random, so its density contrast is
 * D(<r) / (alpha R(<r)) - 1, and the footprint, its holes and the radial
 * selection all come in through the randoms at once. A void's radius is then
 * the largest data distance at which that contrast is still at or below the
 * threshold, exactly as in the box.
 *
 * **Inputs.** Data and randoms each come as a CIC density grid and a chain
 * mesh, in the same Cartesian frame and the same box. The randoms may follow
 * the survey's n(z) -- they normally do -- and both sets may carry weights;
 * the meshes' total weights set alpha. Where the randoms are is taken as the
 * footprint: a grid cell holding at least one random is observed, any other
 * is not, so the randoms have to be dense enough to put one in every observed
 * cell, including where the selection is sparsest.
 *
 * **Geometry.** Nothing wraps around. The survey has to sit inside the box
 * with about the largest search sphere of empty space on every side -- the
 * largest radius times the SIF_FINDER_SEARCH_* factor, plus a grid cell --
 * and the call fails otherwise, saying how much it needs; that padding is what
 * keeps the FFT smoothing and the sphere searches from seeing the far side of
 * the box. Coordinates are the caller's: convert sky positions and redshifts to
 * comoving Cartesian ones beforehand, and shift both sets into the box by the
 * same offset.
 *
 * **Candidates** are only ever centred on an observed cell, and are scanned
 * on the ratio of the smoothed data and random fields rather than on a
 * contrast against the box mean, so a sphere that straddles an edge is
 * measured against the part of it that was observed.
 *
 * **Every void found is kept**, with its sif_catalog_t::footprint and
 * sif_catalog_t::footprint_shell: the fraction of its sphere, and of the
 * shell out to twice its radius, that lies in observed cells. A void cut by
 * the edge is still a void of the observed volume; the fractions say how far
 * to trust it.
 *
 * @param data_grid CIC density of the data, from sif_grid_assign_cic(). NOT
 * a density contrast. Smoothed in place and restored, as in
 * sif_finder_exodus().
 * @param random_grid CIC density of the randoms, on the same cells and box.
 * Restored the same way.
 * @param data_mesh The data, in a chain mesh over the same box.
 * @param random_mesh The randoms, in a chain mesh over the same box. Its
 * cells need not match the data mesh's; size it for the randoms.
 * @param radii Array of smoothing radii.
 * @param n_radii Number of radii.
 * @param threshold Density contrast a void is grown to, against the randoms.
 * @param overlap_fraction As in sif_finder_exodus().
 * @param opt Finder options: SIF_FINDER_CONSUME_GRID (both grids),
 * SIF_FINDER_KEEP_CIC_WINDOW and SIF_FINDER_SEARCH_*.
 *
 * @return Newly allocated catalogue, with its footprint columns, released
 * with sif_catalog_free(); NULL on invalid input, a survey too close to the
 * box faces, or failure.
 */
SIF_NODISCARD sif_catalog_t* sif_finder_exodus_survey(sif_grid_t* data_grid,
  sif_grid_t* random_grid, const sif_chain_mesh_t* data_mesh,
  const sif_chain_mesh_t* random_mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option opt);

/**
 * @brief The box a survey has to be searched in, and the offset that moves it
 * there.
 *
 * sif_finder_exodus_survey() needs empty padding around the survey, about the
 * largest search sphere on every side, and how much depends on the radii, the
 * SIF_FINDER_SEARCH_* factor and the grid cell -- which depends on the box.
 * This works all of that out from the randoms, which are the footprint: a
 * cubic box, just large enough, with the survey centred in it.
 *
 * The workflow it belongs to:
 *
 * @code
 * sif_real offset[3], box;
 * sif_finder_exodus_survey_box(randoms, radii, n_radii, n_cells, opt,
 *   offset, &box);
 * sif_field_translate(data, offset);
 * sif_field_translate(randoms, offset);
 * // grids of n_cells over box, meshes over box, then:
 * sif_catalog_t* cat = sif_finder_exodus_survey(...);
 * const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
 * sif_catalog_translate(cat, back);
 * @endcode
 *
 * @param randoms The random catalogue, in the caller's Cartesian frame. Only
 * read.
 * @param radii The radii the finder will be run with.
 * @param n_radii Number of radii.
 * @param n_cells Cells per side of the grids that will be built over the
 * box; at least 16.
 * @param opt The options the finder will be run with, for the
 * SIF_FINDER_SEARCH_* factor.
 * @param[out] offset What to add to every position, data and randoms alike.
 * @param[out] box_length Side of the box to build the grids and meshes over.
 * @return SIF_OK, or SIF_ERR_INVALID on bad arguments.
 *
 * @note Run the finder with the same radii, options and n_cells, or the
 * padding may no longer be enough; the finder checks and says so. The box is
 * sized for meshes whose cells are no larger than the search sphere, which is
 * what sif_finder_suggest_mesh_cells() gives.
 */
SIF_NODISCARD int sif_finder_exodus_survey_box(const sif_field_t* randoms,
  const sif_real* radii, uint32_t n_radii, uint32_t n_cells, sif_option opt,
  sif_real offset[3], sif_real* box_length);

/** @brief Particles per mesh cell that exodus runs fastest at. The cost
 * splits between per-cell overhead, which grows as the mesh is refined, and
 * per-particle work in the cells straddling the annulus boundaries, which grows
 * as it is coarsened; measured across radii and tracer densities, the balance
 * sits here.
 */
#define SIF_FINDER_MESH_PARTICLES_PER_CELL 30.0

/** @brief Cap on the suggested resolution.
 *
 * The mesh's cell_offsets array alone is 8 * n_cells^3 bytes: 8.6 GiB at this
 * cap, against 1.1 GiB at the 512 it replaces. That sounds worse than it is,
 * because the cap only binds above 30 * 1024^3 = 3.2e10 tracers -- a mesh
 * whose payload is already several hundred GiB. What it fixes is the range
 * just below: at 2048^3 tracers the rule asks for 659 cells and the old cap
 * forced 512, which is 64 tracers per cell against the 30 this finder is
 * fastest at, and it doubled the per-cell cost of the canonical sort.
 */
#define SIF_FINDER_MESH_MAX_CELLS 1024u

/**
 * @brief Suggested chain-mesh resolution for this finder.
 *
 * Mesh resolution changes only speed and memory: the catalog is identical at
 * any resolution, so it is safe to tune. It is worth tuning -- at box/4, the
 * rule the finder used back when it built the mesh itself, an 8 million
 * particle run measured about 1.3x slower than at this resolution, and a
 * coarser mesh costs considerably more than that.
 *
 * @param n_particles Number of tracers the mesh will hold.
 * @param box_length Physical side length, which must match the grid's
 * @param max_radius Largest smoothing radius the run will use. The search
 * sphere it implies has to fit inside the mesh, which puts a floor under the
 * resolution; pass 0 to skip that constraint.
 *
 * @return n_cells to hand to sif_chain_mesh_alloc, or 0 if the geometry is
 * unusable, since no mesh can hold a search sphere wider than the box.
 */
static inline uint32_t sif_finder_suggest_mesh_cells(
  uint64_t n_particles, sif_real box_length, sif_real max_radius) {

  if (n_particles == 0 || !(box_length > 0.0f))
    return 0;

  double n = cbrt((double)n_particles / SIF_FINDER_MESH_PARTICLES_PER_CELL);

  /* The traversal wraps a cell index with a single step, so the stencil for the
   * widest search sphere has to fit inside the mesh. That is a floor on the
   * resolution, not a ceiling: it is the coarse meshes that fail. */
  if (max_radius > 0.0f) {
    const double r_search = 2.0 * (double)max_radius;
    const double box = (double)box_length;

    if (r_search >= box)
      return 0;

    const double floor_cells = box / (box - r_search) + 1.0;
    if (n < floor_cells)
      n = floor_cells;
  }

  if (n < 8.0)
    n = 8.0;
  if (n > (double)SIF_FINDER_MESH_MAX_CELLS)
    n = (double)SIF_FINDER_MESH_MAX_CELLS;

  return (uint32_t)n;
}

#endif /* SIF_FINDER_EXODUS_FINDER_H */