/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file spherical_finder.h
 * @brief Spherical void finder: fixed-radius spheres on a density grid.
 *
 * Walks the smoothing radii from largest to smallest. At each radius it
 * collects every grid cell at or below the threshold, visits them deepest
 * first, and accepts a sphere of exactly that radius wherever one does not
 * overlap a void already accepted. Cells swallowed by an accepted void are
 * masked, so later and smaller radii cannot re-find the same underdensity.
 *
 * Voids therefore come out with radii drawn from the ladder that was passed in,
 * not from the data. sif_finder_exodus() relaxes that by growing
 * each sphere to the radius the tracers actually support.
 */

#ifndef SIF_FINDER_SPHERICAL_FINDER_H
#define SIF_FINDER_SPHERICAL_FINDER_H

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/grid.h"

/**
 * @brief Run the spherical void finder on a density grid.
 *
 * @param grid Density contrast field, as produced by
 * sif_grid_to_density_contrast(). The finder smooths it in place at each
 * radius and transforms the original back at the end, so on return it holds
 * what it held on entry. Pass SIF_FINDER_CONSUME_GRID to skip that final
 * transform.
 * @param radii Smoothing radii to try. Order does not matter: the finder sorts
 * them descending, since accepting the large voids first is what makes the
 * mask meaningful.
 * @param n_radii Number of radii.
 * @param threshold Density contrast a cell must be at or below to seed a void.
 * Negative, since voids are underdensities.
 * @param overlap_fraction How much two voids may overlap, as a fraction of the
 * smaller one's radius: they collide when their separation falls below
 * r1 + r2 - overlap_fraction * min(r1, r2). 0 forbids overlap entirely; 1 lets
 * the smaller void's radius be swallowed.
 * @param opt Finder options. Honours SIF_FINDER_CONSUME_GRID and
 * SIF_FINDER_KEEP_CIC_WINDOW. The box is always periodic.
 *
 * @note The grid is assumed to have been built by sif_grid_assign_cic(), whose
 * window is divided back out before the first smoothing so that the top-hat is
 * the only window applied. Pass SIF_FINDER_KEEP_CIC_WINDOW for a grid that was
 * filled some other way.
 * @return Newly allocated catalogue, released with sif_catalog_free(), or NULL
 * on invalid input or failure.
 *
 * @warning `grid->values` is reassigned during the run: the original buffer is
 * released once its contents are in the FFT workspace, and the workspace's is
 * handed back at the end. The contents are restored, the pointer is not, so a
 * pointer cached from before the call dangles afterwards.
 */
SIF_NODISCARD sif_catalog_t* sif_finder_spherical(sif_grid_t* grid,
  const sif_real* radii, uint32_t n_radii, sif_real threshold,
  sif_real overlap_fraction, sif_option opt);

#endif /* SIF_FINDER_SPHERICAL_FINDER_H */
