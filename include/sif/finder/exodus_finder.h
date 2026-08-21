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
 * The finder reads only the mesh positions. Velocities and weights ride along
 * if the field carries them, which at these particle counts is a substantial
 * allocation for nothing -- so hand this a field stripped of both when the
 * mesh is being built for the finder alone.
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
 * @param opt Finder options. Honours SIF_PBC_PERIODIC / SIF_PBC_OPEN,
 * SIF_FINDER_CONSUME_GRID and SIF_FINDER_KEEP_CIC_WINDOW.
 *
 * @note The grid is assumed to have been built by sif_grid_assign_cic(), whose
 * window is divided back out before the first smoothing so that the top-hat is
 * the only window applied. Pass SIF_FINDER_KEEP_CIC_WINDOW for a grid that was
 * filled some other way.
 *
 * @return Newly allocated catalogue, released with sif_catalog_free(), or
 * NULL on invalid input or failure.
 */
SIF_NODISCARD sif_catalog_t* sif_finder_exodus(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const sif_real* radii, uint32_t n_radii,
  sif_real threshold, sif_real overlap_fraction, sif_option opt);

/** @brief Particles per mesh cell that exodus runs fastest at. The cost
 * splits between per-cell overhead, which grows as the mesh is refined, and
 * per-particle work in the cells straddling the annulus boundaries, which grows
 * as it is coarsened; measured across radii and tracer densities, the balance
 * sits here. */
#define SIF_FINDER_MESH_PARTICLES_PER_CELL 30.0

/** @brief Cap on the suggested resolution. The mesh's cell_offsets array alone
 * is 8 * n_cells^3 bytes, which is already ~1 GiB here. */
#define SIF_FINDER_MESH_MAX_CELLS 512u

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