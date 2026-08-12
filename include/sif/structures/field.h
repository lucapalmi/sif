#ifndef __SIF_FIELD_H__
#define __SIF_FIELD_H__

#include "sif/core/macros.h"
#include <stdint.h>

/*
 * A field always owns every buffer it points at.
 *
 * There used to be a borrowing mode (FIELD_POINTS) alongside per-array
 * ownership bits. It could not be honoured: assigning masses or velocities to a
 * sorted field silently took ownership anyway so the incoming data could be
 * permuted, and sif_field_sort_morton reallocates unconditionally, which
 * detached the field from the caller's arrays with no diagnostic. What is left
 * describes the state of the data, not who frees it.
 *
 * To fill a field without paying for a copy, reserve the arrays and write into
 * field->x and friends directly; that is what the readers do.
 */
#define __FIELD_STATE_MORTON_SORTED (1u << 0)
#define __FIELD_STATE_BOUNDS_VALID  (1u << 1)

/* Bits per axis in the 3D Morton code. Three of these must fit in a uint64. */
#define SIF_MORTON_BITS 21

/*
 * @brief Quantizes one coordinate onto the field's bounding-cube grid.
 *
 * Bit (SIF_MORTON_BITS - 1 - d) of the result answers "is this point in the
 * upper half at subdivision level d". Everything that subdivides space the way
 * sif_field_sort_morton does -- notably the octree -- must go through this
 * function, so the two agree exactly instead of approximately. Recomputing an
 * equivalent float comparison independently is what used to put particles in
 * the wrong octree cell.
 *
 * @param origin The low corner of the bounding cube (center - half_span)
 * @param inv_side 1 / (2 * half_span)
 */
static inline uint32_t sif_field_quantize(
  real_t p, real_t origin, real_t inv_side) {

  const real_t t = (p - origin) * inv_side;
  const int32_t q_max = (int32_t)((1u << SIF_MORTON_BITS) - 1u);

  int32_t q = (int32_t)(t * (real_t)(1u << SIF_MORTON_BITS));
  return (uint32_t)((q < 0) ? 0 : ((q > q_max) ? q_max : q));
}

/*
 * @brief Represents a physical particle field
 */
typedef struct {
  real_t* _position_block;
  real_t* _velocity_block;

  real_t* x;
  real_t* y;
  real_t* z;

  real_t* vx;
  real_t* vy;
  real_t* vz;

  real_t* masses;
  uint64_t* original_indices;

  real_t min_p[3];
  real_t max_p[3];
  real_t center[3];
  real_t half_span;

  uint32_t state_flags;
  uint64_t n_particles;
} sif_field_t;

/*
 * @brief Initializes a field
 *
 * @param n_particles The number of particles in the field
 *
 * @return Initialized field
 */
NODISCARD sif_field_t* sif_field_alloc(uint64_t n_particles);

/*
 * @brief Per-array stride inside a packed x/y/z block
 *
 * Each sub-array of a unified block starts on a cache line, so a block holds
 * 3 * sif_field_padded_n(n) elements rather than 3 * n. Anything that lays out
 * its own block the same way (the chain mesh, the readers) has to agree on
 * this, so it lives here rather than being open-coded per call site.
 */
PURE_FUNCTION uint64_t sif_field_padded_n(uint64_t n_particles);

/*
 * @brief Frees a field
 *
 * @param field The field to free
 */
void sif_field_free(sif_field_t* field);

/*
 * @brief Allocate the position arrays without filling them
 *
 * Sized from field->n_particles, which must be set. Writing straight into
 * field->x, field->y and field->z afterwards is the zero-copy way to populate a
 * field. A no-op if the arrays already exist.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on an empty field, SIF_ERR_ALLOC
 * on failure
 */
int sif_field_reserve_positions(sif_field_t* field);

/*
 * @brief Allocate the velocity arrays without filling them. See
 * sif_field_reserve_positions.
 */
int sif_field_reserve_velocities(sif_field_t* field);

/*
 * @brief Allocate the mass array without filling it. See
 * sif_field_reserve_positions.
 */
int sif_field_reserve_masses(sif_field_t* field);

/*
 * @brief Copy positions into the field
 *
 * @param field The field
 * @param x X-axis positions
 * @param y Y-axis positions
 * @param z Z-axis positions
 *
 * @note Invalidates the bounds and the Morton order, and drops the permutation
 * a previous sort had recorded: it no longer describes this data.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC
 * on failure
 */
int sif_field_assign_positions(
  sif_field_t* field, const real_t* x, const real_t* y, const real_t* z);

/*
 * @brief Copy velocities into the field
 *
 * @param field The field
 * @param vx X-axis velocities, indexed as the positions were BEFORE any Morton
 * sort (i.e. in the caller's original particle order)
 * @param vy Y-axis velocities
 * @param vz Z-axis velocities
 *
 * @note If the field has already been Morton-sorted the incoming arrays are
 * permuted into the field's current order on the way in, so that every particle
 * keeps its own velocity.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC
 * on failure
 */
int sif_field_assign_velocities(
  sif_field_t* field, const real_t* vx, const real_t* vy, const real_t* vz);

/*
 * @brief Copy per-particle masses into the field
 *
 * @param field The field
 * @param masses Masses in the caller's original particle order
 *
 * @note Same permutation rule as sif_field_assign_velocities.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC
 * on failure
 */
int sif_field_assign_masses(sif_field_t* field, const real_t* masses);

/*
 * @brief Computes and caches the bounding box and center of the field.
 *
 * Recomputes unconditionally. Prefer sif_field_require_bounds unless you
 * specifically need to force a refresh.
 *
 * @param field The field
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on an empty or positionless field
 */
int sif_field_compute_bounds(sif_field_t* field);

/*
 * @brief Ensures the field has a valid bounding box, computing it if needed.
 *
 * Idempotent and quiet: safe to call from anything that reads min_p, max_p,
 * center or half_span.
 *
 * @return SIF_OK if the bounds are valid on return, an error code otherwise
 */
int sif_field_require_bounds(sif_field_t* field);

/*
 * @brief Physically sorts the field arrays in memory using a 3D Morton curve.
 *
 * @note On success the field owns its positions and indices regardless of how
 * they were assigned: the permuted copy cannot alias the caller's arrays.
 *
 * @return SIF_OK on success, SIF_ERR_INVALID on an empty field, SIF_ERR_ALLOC
 * on failure (in which case the field is left unmodified)
 */
int sif_field_sort_morton(sif_field_t* field);

/*
 * @brief Ensures the field is Morton-sorted, sorting it if needed.
 *
 * Consumers that structurally depend on Morton order (the octree, which stores
 * contiguous particle ranges per node) should call this instead of testing the
 * flag and failing.
 *
 * @return SIF_OK if the field is Morton-sorted on return, an error otherwise
 */
int sif_field_require_morton(sif_field_t* field);

#endif /* __SIF_FIELD_H__ */
