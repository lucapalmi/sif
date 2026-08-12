#ifndef __SIF_RESCALED_SPHERICAL_FINDER_H__
#define __SIF_RESCALED_SPHERICAL_FINDER_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/grid.h"

/*
 * @brief Run the rescaled spherical void finder on a cubic grid
 *
 * The mesh is borrowed, never freed: build it once and it can be reused across
 * several runs at different thresholds or radius sets. Building it also means
 * the particle field is no longer needed here, so it can be released before
 * this call -- which matters, because the finder's own peak sits right on top
 * of whatever the caller is still holding.
 *
 * The mesh only has to carry positions; masses, velocities and original
 * indices are never read, and at these particle counts each of them is a
 * substantial allocation to have made for nothing.
 *
 * @param grid The overdensity field
 * @param mesh The particle chain mesh. Must span the same box as the grid.
 * @param radii Array of smoothing radii. Each must be small enough that the
 * search sphere it implies (roughly twice the radius) fits inside the box.
 * @param n_radii Number of radii
 * @param threshold Void identification threhsold
 * @param options Finder options
 *
 * @return Pointer to a newly allocated sif_catalog_t, or NULL on failure.
 */
NODISCARD sif_catalog_t* sif_finder_rescaled_spherical(sif_grid_t* grid,
  const sif_chain_mesh_t* mesh, const real_t* radii, uint32_t n_radii,
  real_t threshold, real_t overlap_fraction, sif_option_t opt);

/* Particles per mesh cell that the rescale runs fastest at. The cost splits
 * between per-cell overhead, which grows as the mesh is refined, and
 * per-particle work in the cells straddling the annulus boundaries, which grows
 * as it is coarsened; measured across radii and tracer densities, the balance
 * sits here. */
#define SIF_FINDER_MESH_PARTICLES_PER_CELL 30.0

/* Cap on the suggested resolution. The mesh's cell_offsets array alone is
 * 8 * n_cells^3 bytes, which is already ~1 GiB here. */
#define SIF_FINDER_MESH_MAX_CELLS 512u

/*
 * @brief Suggested chain-mesh resolution for this finder
 *
 * Mesh resolution changes only speed and memory: the catalog is identical at
 * any resolution, so it is safe to tune. It is worth tuning -- at box/4, the
 * rule the finder used back when it built the mesh itself, an 8 million
 * particle run measured about 1.3x slower than at this resolution, and a
 * coarser mesh costs considerably more than that.
 *
 * @param n_particles Number of tracers the mesh will hold
 * @param box_length Physical side length, which must match the grid's
 * @param max_radius Largest smoothing radius the run will use. The search
 * sphere it implies has to fit inside the mesh, which puts a floor under the
 * resolution; pass 0 to skip that constraint.
 *
 * @return n_cells to hand to sif_chain_mesh_alloc, or 0 if the geometry is
 * unusable (no mesh can hold a search sphere wider than the box)
 */
static inline uint32_t sif_finder_suggest_mesh_cells(
  uint64_t n_particles, real_t box_length, real_t max_radius) {

  if (n_particles == 0 || !(box_length > 0.0f))
    return 0;

  double n =
    cbrt((double)n_particles / SIF_FINDER_MESH_PARTICLES_PER_CELL);

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

#endif /* __SIF_RESCALED_SPHERICAL_FINDER_H__*/