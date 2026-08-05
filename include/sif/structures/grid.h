#ifndef __SIF_GRID_H__
#define __SIF_GRID_H__

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

/*
 * @ brief A cubic grid
 */
typedef struct {
  real_t* delta;        // overdensity field, flat row-major
  uint32_t n_cells;     // number of cells per side
  uint64_t total_cells; // total number of cells
  real_t box_length;    // physical side length
  real_t cell_length;   // physical cell size
  
  uint32_t p2_mask;     // (N-1) if N is a power of two, else 0
  uint32_t p2_shift;    // log2(N) if N is a power of two, else 0
} sif_grid_t;

/*
 * @brief Allocate memory for a new, zero initialized cubic grid
 *
 * @param grid_size The size of the grid
 * @param box_size The physical size of the box
 *
 * @return Pointer to the initialized grid
 */
NODISCARD sif_grid_t* sif_grid_alloc(uint32_t n_cells, real_t box_length);

/*
 * @brief Frees a cubic grid
 *
 * @param grid The grid to free
 */
void sif_grid_free(sif_grid_t* grid);

/*
 * @brief Assign particles to the grid using Cloud-In-Cell interpolation.
 *
 * @param grid The target grid
 * @param field The particle field
 */
void sif_grid_assign_cic(sif_grid_t* grid, sif_field_t* field);

/*
 * @brief Compute the overdensity field in a cubic grid
 *
 * @param grid The cubic grid
 * @param total_mass The total mass contained in the grid
 */
void sif_grid_compute_overdensity(sif_grid_t* grid);

#endif /* __SIF_GRID_H__ */
