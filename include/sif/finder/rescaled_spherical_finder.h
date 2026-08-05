#ifndef __SIF_RESCALED_SPHERICAL_FINDER_H__
#define __SIF_RESCALED_SPHERICAL_FINDER_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

/*
 * @brief Run the rescaled spherical void finder on a cubic grid
 *
 * @param grid The overdensity field
 * @param field The particle field
 * @param radii Array of smoothing radii
 * @param n_radii Number of radii
 * @param threshold Void identification threhsold
 * @param options Finder options
 *
 * @return Pointer to a newly allocated sif_catalog_t, or NULL on failure.
 */
NODISCARD sif_catalog_t* sif_finder_rescaled_spherical(sif_grid_t* grid, sif_field_t* field,
  const real_t* radii, uint32_t n_radii, real_t threshold, real_t overlap_fraction, sif_option_t opt);

#endif /* __SIF_RESCALED_SPHERICAL_FINDER_H__*/