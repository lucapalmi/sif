#ifndef __SIF_OCTREE_H__
#define __SIF_OCTREE_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/field.h"

typedef struct {
  uint32_t first_child;
  uint32_t p_start;
  uint32_t p_counting;
  uint32_t padding;
} sif_octree_node_t;

typedef struct {
  sif_octree_node_t* nodes;
  uint32_t capacity;
  uint32_t count;

  real_t root_center[3];
  real_t root_half_span;
} sif_octree_t;

/*
 * @brief Allocates a new octree
 *
 * @param initial_capacity The initial capacity of the octree
 *
 * @return Initialized octree (NULL on failure)
 */
sif_octree_t* sif_octree_alloc(uint32_t initial_capacity);

/*
 * @brief Free the memory associated to an octree
 *
 * @param tree The octree to free
 */
void sif_octree_free(sif_octree_t* tree);

/*
 * @brief Build an octree from a field
 *
 * @param tree The octree to fill
 * @param field The particle field
 * @param max_per_leaf Maximum number of particles in each leaf
 *
 * @return 0 on success, -1 on failure
 */
int sif_octree_build(sif_octree_t* tree, sif_field_t* field, uint32_t max_per_leaf);

/*
 * @brief Find the nearest particle to the specified coordinates
 *
 * @param tree The octree
 * @param field The particle field
 * @param px X-axis position
 * @param py Y-axis position
 * @param pz Z-axis position
 *
 * @return Index of nearest particle
 */
uint64_t sif_octree_find_nearest(
  const sif_octree_t* tree, const sif_field_t* field, real_t px, real_t py, real_t pz);

/*
 * @brief Finds all particles within a sphere of given radius.
 *
 * @param tree The octree
 * @param field The particle field
 * @param px X-axis position
 * @param py Y-axis position
 * @param pz Z-axis position
 * @param radius The search radius
 * @param out_indices Output array for the particle indices
 * @param max_capacity Size of out_indices
 *
 * @return Total number of found particles
 */
uint64_t sif_octree_search_radius(const sif_octree_t* tree, const sif_field_t* field,
  real_t px, real_t py, real_t pz, real_t radius, uint64_t* out_indices,
  uint64_t max_capacity);

/*
 * @brief Finds all particles inside an axis-aligned bounding box
 *
 * @param tree The octree
 * @param field The particle field
 * @param min_x Lower x bound
 * @param min_y Lower y bound
 * @param min_z Lower z bound
 * @param max_x Upper x bound
 * @param max_y Upper y bound
 * @param max_z Upper z bound
 * @param out_indices Output array for the particle indices
 * @param max_capacity Size of out_indices
 *
 * @return Total number of found particles
 */
uint64_t sif_octree_search_box(const sif_octree_t* tree, const sif_field_t* field,
  real_t min_x, real_t min_y, real_t min_z, real_t max_x, real_t max_y,
  real_t max_z, uint64_t* out_indices, uint64_t max_capacity);

#endif /* __SIF_OCTREE_H__ */
