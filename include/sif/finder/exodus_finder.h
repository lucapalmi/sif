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
#include "sif/structures/catalogue.h"
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
 * @param radii Array of smoothing radii, each positive. Each must be small
 * enough that the search sphere it implies (roughly twice the radius) fits
 * inside the box.
 * @param n_radii Number of radii
 * @param threshold Density contrast a cell must be at or below to seed a
 * void, in (-1, 0).
 * @param overlap_fraction How much two voids may overlap, as a fraction of
 * the smaller one's radius, in [0, 1]; see sif_finder_spherical().
 * @param opt Finder options. Honours SIF_FINDER_CONSUME_GRID,
 * SIF_FINDER_KEEP_CIC_WINDOW and SIF_FINDER_SEARCH_*. The box is always
 * periodic; a survey, with edges, is sif_finder_exodus_survey().
 *
 * @note The grid is assumed to have been built by sif_grid_assign_cic(), whose
 * window is divided back out before the first smoothing so that the top-hat is
 * the only window applied. Pass SIF_FINDER_KEEP_CIC_WINDOW for a grid that was
 * filled some other way.
 *
 * @return Newly allocated catalogue, released with sif_catalogue_free(), or
 * NULL on invalid input -- a negative or non-finite weight included -- or
 * failure. A run that fails partway returns NULL rather than the voids found
 * so far, with the grid restored as on success; sif_error_message() says
 * why.
 *
 * @note Requires sif_system_init(). Called without it, returns NULL with
 * sif_error_status() at SIF_ERR_UNINITIALISED.
 */
SIF_NODISCARD sif_catalogue_t* sif_finder_exodus(sif_grid_t* grid,
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
 * footprint: the grid cells holding at least one random, with the holes a
 * sparse catalogue leaves by chance closed. The closing runs over a box of
 * 3^3 cells, or wider where the randoms are too few to expect about a hundred
 * in it (up to 9^3), so it also fills real gaps up to two cells wide -- four
 * at the widest -- and a warning says when even the widest leaves chance
 * holes behind.
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
 * **Every void found is kept**, with its sif_catalogue_t::footprint and
 * sif_catalogue_t::footprint_shell: the fraction of its sphere, and of the
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
 * @param offset What the survey was moved into the box by, as
 * sif_finder_exodus_survey_box() gave it, or NULL for a survey that was not
 * moved. The voids are moved back by it, so they come out in the caller's
 * own frame -- where sif_catalogue_to_sky() needs them, with the observer at
 * the origin.
 * @param radii Array of smoothing radii.
 * @param n_radii Number of radii.
 * @param threshold Density contrast a void is grown to, against the randoms,
 * in (-1, 0).
 * @param overlap_fraction As in sif_finder_exodus().
 * @param opt Finder options: SIF_FINDER_CONSUME_GRID (both grids),
 * SIF_FINDER_KEEP_CIC_WINDOW and SIF_FINDER_SEARCH_*.
 *
 * @return Newly allocated catalogue, with its footprint columns, in the
 * caller's frame, released with sif_catalogue_free(); NULL on invalid input
 * (an offset that is not finite included), a survey too close to the box
 * faces, or failure, as for sif_finder_exodus().
 *
 * @note Requires sif_system_init(). Called without it, returns NULL with
 * sif_error_status() at SIF_ERR_UNINITIALISED.
 */
SIF_NODISCARD sif_catalogue_t* sif_finder_exodus_survey(sif_grid_t* data_grid,
  sif_grid_t* random_grid, const sif_chain_mesh_t* data_mesh,
  const sif_chain_mesh_t* random_mesh, const sif_real offset[3],
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option opt);

/**
 * @brief The box a survey has to be searched in, and the offset that moves it
 * there.
 *
 * sif_finder_exodus_survey() needs empty padding around the survey, about the
 * largest search sphere on every side, and how much depends on the radii and
 * the SIF_FINDER_SEARCH_* factor. This works it out from the tracers: a cubic
 * box, just large enough, with the survey centred in it. The data are
 * required as well as the randoms: the finder checks the padding around both,
 * and the outermost galaxies can sit a little past the outermost randoms,
 * which sample the edge of the footprint only so finely.
 *
 * The padding also depends on the grid cell, and the box is chosen before
 * the grid is. So it is sized for any grid whose cells are no larger than the
 * smallest radius -- every grid that resolves the smallest radius at all.
 * Against a grid of negligible cells that costs
 * `r_min + max(0, 3 r_min - (f - 1) r_max)` of extra padding a side, f the
 * search factor: r_min for any ladder spanning a factor of 6 or more at the
 * default factor. A coarser grid is refused by the finder, which says how
 * much padding it would need.
 *
 * The workflow it belongs to:
 *
 * @code
 * sif_real offset[3], box;
 * sif_finder_exodus_survey_box(data, randoms, radii, n_radii, opt, offset,
 *   &box);
 * sif_field_translate(data, offset);
 * sif_field_translate(randoms, offset);
 * // grids and meshes over box, then, with the voids back in the
 * // caller's frame:
 * sif_catalogue_t* cat = sif_finder_exodus_survey(data_grid, random_grid,
 *   data_mesh, random_mesh, offset, radii, n_radii, threshold, 0.0f, opt);
 * @endcode
 *
 * @param data The data catalogue, in the caller's Cartesian frame. Only read.
 * @param randoms The random catalogue, in the same frame. Only read.
 * @param radii The radii the finder will be run with, each positive.
 * @param n_radii Number of radii.
 * @param opt The options the finder will be run with, for the
 * SIF_FINDER_SEARCH_* factor.
 * @param[out] offset What to add to every position, data and randoms alike.
 * @param[out] box_length Side of the box to build the grids and meshes over.
 * @return SIF_OK, or SIF_ERR_INVALID on bad arguments or a coordinate that
 * is not finite.
 *
 * @note Run the finder with the same radii and options, or the padding may
 * no longer be enough; the finder checks and says so. The box is sized for
 * meshes whose cells are no larger than the search sphere, which is what
 * sif_finder_suggest_mesh_cells() gives.
 */
SIF_NODISCARD int sif_finder_exodus_survey_box(const sif_field_t* data,
  const sif_field_t* randoms, const sif_real* radii, uint32_t n_radii,
  sif_option opt, sif_real offset[3], sif_real* box_length);

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
 * Mesh resolution changes only speed and memory: the catalogue is identical at
 * any resolution, so it is safe to tune. It is worth tuning -- at box/4, the
 * rule the finder used back when it built the mesh itself, an 8 million
 * particle run measured about 1.3x slower than at this resolution, and a
 * coarser mesh costs considerably more than that.
 *
 * The rule sizes the cells for the density the tracers are at. In a periodic
 * box they fill the box; in a survey box they fill only the footprint, and
 * the padding around it is empty by construction. Sized from the box average
 * there, a mesh would put several times the intended number of tracers in
 * every cell the survey covers -- the only cells that cost anything -- so
 * pass the footprint's volume, from sif_survey_volume(), and call once for
 * each mesh, the data's and the randoms', with that set's own count.
 *
 * @param n_particles Number of tracers the mesh will hold.
 * @param box_length Physical side length, which must match the grid's
 * @param max_radius Largest smoothing radius the run will use. The search
 * sphere it implies has to fit inside the mesh, which puts a floor under the
 * resolution; pass 0 to skip that constraint.
 * @param volume The volume the tracers occupy, or 0 for the whole box.
 *
 * @return n_cells to hand to sif_chain_mesh_alloc, or 0 for invalid input (no
 * tracers, a box or volume that is not positive and finite) or a geometry no
 * mesh can serve, since none can hold a search sphere wider than the box.
 */
static inline uint32_t sif_finder_suggest_mesh_cells(uint64_t n_particles,
  sif_real box_length, sif_real max_radius, sif_real volume) {

  if (n_particles == 0 || !(box_length > 0.0f) || !isfinite(box_length) ||
      !(volume >= 0.0f) || !isfinite(volume))
    return 0;

  /* The count the whole box would hold at the tracers' own density. */
  const double box = (double)box_length;
  const double filled = volume > 0.0f
                          ? (double)n_particles * box * box * box / volume
                          : (double)n_particles;
  double n = cbrt(filled / SIF_FINDER_MESH_PARTICLES_PER_CELL);

  /* The traversal wraps a cell index with a single step, so the stencil for the
   * widest search sphere has to fit inside the mesh. That is a floor on the
   * resolution, not a ceiling: it is the coarse meshes that fail. */
  if (max_radius > 0.0f) {
    const double r_search = 2.0 * (double)max_radius;

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

/** @brief The unit a length is given in. */
typedef enum {
  /** The box's own length unit. */
  SIF_FINDER_UNITS_PHYSICAL = 0,
  /** Mean tracer separation, (volume / n_particles)^(1/3). */
  SIF_FINDER_UNITS_MPS
} sif_finder_units_t;

/**
 * @brief Grid cells per side that make each cell a given size.
 *
 * The density grid is the finder's resolution: its smallest radius should
 * span two cells at least, so a grid of one cell per mean separation suits a
 * ladder that starts at two separations. This turns a cell size into the
 * n_cells sif_grid_alloc() takes, rounding up so that the cells are never
 * larger than asked for. A size that divides the box to within rounding
 * error -- 1 mps on a 256^3 box, say -- gives that exact count.
 *
 * The count is not snapped to an FFT-friendly size; do that separately if
 * the transform's speed matters.
 *
 * @param n_particles Number of tracers; read only for SIF_FINDER_UNITS_MPS.
 * @param box_length Side of the box the grid spans.
 * @param cell_size Wanted cell side, in `units`; positive and finite.
 * @param units SIF_FINDER_UNITS_PHYSICAL or SIF_FINDER_UNITS_MPS.
 * @param volume The volume the tracers occupy, which sets the mean
 * separation, or 0 for the whole box; read only for SIF_FINDER_UNITS_MPS. In
 * a survey box the tracers fill only the footprint: pass its volume, from
 * sif_survey_volume(), or the separation comes out inflated by the padding.
 * @return n_cells to hand to sif_grid_alloc(), or 0 for invalid input or a
 * count that does not fit in 32 bits; sif_error_message() says which.
 */
SIF_NODISCARD uint32_t sif_finder_grid_cells_from_size(uint64_t n_particles,
  sif_real box_length, sif_real cell_size, sif_finder_units_t units,
  sif_real volume);

#endif /* SIF_FINDER_EXODUS_FINDER_H */