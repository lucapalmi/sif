#ifndef __SIF_CHAIN_MESH_H__
#define __SIF_CHAIN_MESH_H__

#include "sif/core/macros.h"
#include "sif/structures/field.h"
#include <stdbool.h>
#include <stdint.h>

/*
 * @brief Represents a 3D cubic chain mesh
 *
 * @note The mesh is anchored at the origin: all particle coordinates, and all
 * query coordinates, must lie in [0, box_length) on every axis. This is
 * enforced at construction time. Callers working in a shifted frame have to
 * translate into box-local coordinates first.
 */
typedef struct {
  uint32_t n_cells;
  uint64_t total_cells;
  real_t box_length;
  real_t cell_length;

  real_t* _position_block;
  real_t* _velocity_block;

  /* Sorted contiguous payload arrays */
  real_t* x;
  real_t* y;
  real_t* z;

  real_t* vx;
  real_t* vy;
  real_t* vz;

  real_t* masses;
  uint64_t* original_idx;

  uint64_t n_particles;
  uint64_t* cell_offsets;
} sif_chain_mesh_t;

/*
 * @brief Allocates and populates a new 3D cubic chain mesh
 *
 * @param n_cells Number of cells in the grid
 * @param box_length The physical size of the box
 * @param field The particle field to bin. Every coordinate must be in
 * [0, box_length); the call fails if any particle lies outside.
 * @param allocate_masses Boolean flag to allocate mass tracking array
 * @param allocate_velocities Boolean flag to allocate velocity tracking arrays
 * @param allocate_original_idx Allocate the map back to field indices.
 * Required by the find_nearest queries.
 *
 * @return Pointer to the populated mesh, NULL on invalid input or failure
 */
NODISCARD sif_chain_mesh_t* sif_chain_mesh_alloc(uint32_t n_cells,
  real_t box_length, const sif_field_t* field, bool allocate_masses,
  bool allocate_velocities, bool allocate_original_idx);

/*
 * @brief Frees a 3D cubic chain mesh (NULL is a no-op)
 */
void sif_chain_mesh_free(sif_chain_mesh_t* mesh);

/*
 * @brief Finds the index of the nearest particle using open boundaries.
 *
 * @return The index of the nearest particle in the original field, or
 * UINT64_MAX if the mesh is empty or was built without original indices.
 */
uint64_t sif_chain_mesh_find_nearest_open(
  const sif_chain_mesh_t* mesh, real_t px, real_t py, real_t pz);

/*
 * @brief Finds the index of the nearest particle using periodic boundaries.
 *
 * @return The index of the nearest particle in the original field, or
 * UINT64_MAX if the mesh is empty or was built without original indices.
 */
uint64_t sif_chain_mesh_find_nearest_pbc(
  const sif_chain_mesh_t* mesh, real_t px, real_t py, real_t pz);

#endif /* __SIF_CHAIN_MESH_H__ */
