/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file field.h
 * @brief The particle field: positions, velocities and weights, in
 * structure-of-arrays layout.
 *
 * Coordinates are held as three separate arrays rather than as interleaved
 * triples, because every loop over the field reads one axis at a time and an
 * interleaved layout would fetch three times the cache lines it uses.
 *
 * **A field always owns every buffer it points at.** There used to be a
 * borrowing mode (`FIELD_POINTS`) alongside per-array ownership bits. It could
 * not be honoured: assigning weights or velocities to a sorted field silently
 * took ownership anyway so the incoming data could be permuted, and
 * sif_field_sort_morton() reallocates unconditionally, which detached the field
 * from the caller's arrays with no diagnostic. What is left describes the state
 * of the data, not who frees it.
 *
 * To fill a field without paying for a copy, reserve the arrays and write into
 * `field->x` and friends directly; that is what the readers do.
 *
 * **Positions are Cartesian, or sky coordinates waiting to become so.** A
 * survey arrives as right ascension, declination and redshift; loaded into
 * x, y and z, and declared as such with sif_field_t::units =
 * #SIF_FIELD_SKY, it is turned into comoving Cartesian positions by
 * sif_field_convert_sky_coordinates(). Until then, everything that bins,
 * sorts, bounds or moves positions refuses the field rather than read angles
 * as lengths.
 */

#ifndef SIF_STRUCTURES_FIELD_H
#define SIF_STRUCTURES_FIELD_H

#include "sif/core/macros.h"
#include "sif/model/cosmology.h"
#include <stdint.h>

/** @brief What a field's positions are. */
typedef enum {
  /**
   * Cartesian coordinates in x, y and z, in the units of the box. What every
   * function that bins, sorts, bounds or moves positions needs, and what
   * every reader in sif produces.
   */
  SIF_FIELD_CARTESIAN = 0,
  /**
   * Sky coordinates: x holds the right ascension and y the declination, both
   * in degrees, and z the redshift. Only sif_field_convert_sky_coordinates()
   * takes a field like this.
   */
  SIF_FIELD_SKY
} sif_field_units_t;

/**
 * @defgroup field_state Field state flags
 * @brief Bits of sif_field_t::state_flags, describing what is currently true
 * of the data.
 * @{
 */
/** Arrays are physically ordered along the Morton curve. */
#define SIF_FIELD_STATE_MORTON_SORTED (1u << 0)
/** The cached bounding box and centre match the current positions. */
#define SIF_FIELD_STATE_BOUNDS_VALID (1u << 1)
/** @} */

/** @brief Bits per axis in the 3D Morton code. Three of these must fit in a
 * uint64.
 */
#define SIF_MORTON_BITS 21

/**
 * @brief Quantize one coordinate onto the field's bounding-cube grid.
 *
 * Bit (SIF_MORTON_BITS - 1 - d) of the result answers "is this point in the
 * upper half at subdivision level d". Everything that subdivides space the way
 * sif_field_sort_morton() does -- notably the octree -- must go through this
 * function, so the two agree exactly instead of approximately. Recomputing an
 * equivalent float comparison independently is what used to put particles in
 * the wrong octree cell.
 *
 * @param p Coordinate to quantize.
 * @param origin The low corner of the bounding cube (center - half_span).
 * @param inv_side 1 / (2 * half_span).
 * @return The quantized coordinate, clamped to the cube.
 */
static inline uint32_t sif_field_quantize(
  sif_real p, sif_real origin, sif_real inv_side) {

  const sif_real t = (p - origin) * inv_side;
  const int32_t q_max = (int32_t)((1u << SIF_MORTON_BITS) - 1u);

  int32_t q = (int32_t)(t * (sif_real)(1u << SIF_MORTON_BITS));
  return (uint32_t)((q < 0) ? 0 : ((q > q_max) ? q_max : q));
}

/**
 * @brief A physical particle field.
 *
 * Any of the data arrays may be NULL: a field carries only what was reserved or
 * assigned into it. Check before reading, or reserve up front.
 */
typedef struct {
  /** Backing store for x/y/z. Owned; not for callers. */
  sif_real* _position_block;
  /** Backing store for vx/vy/vz. Owned; not for callers. */
  sif_real* _velocity_block;

  /** Position, x axis. Views into #_position_block. */
  sif_real* x;
  sif_real* y;
  sif_real* z;

  /** Velocity, x axis. Views into #_velocity_block. */
  sif_real* vx;
  sif_real* vy;
  sif_real* vz;

  /** Per-particle weight -- a mass, a luminosity, a selection weight -- or
   * NULL, which every consumer reads as a weight of 1.
   */
  sif_real* weights;
  /** Where each particle sat before the Morton sort, or NULL if unsorted. */
  uint64_t* original_indices;

  /** Lower corner of the bounding box. */
  sif_real min_p[3];
  /** Upper corner of the bounding box. */
  sif_real max_p[3];
  /** Centre of the bounding cube. */
  sif_real center[3];
  /** Half-side of the bounding cube. */
  sif_real half_span;

  /** Bitwise OR of the SIF_FIELD_STATE_* flags. */
  uint32_t state_flags;
  uint64_t n_particles;

  /**
   * What the positions are. #SIF_FIELD_CARTESIAN from sif_field_alloc() and
   * from every reader; set it to #SIF_FIELD_SKY after putting right
   * ascension, declination and redshift into x, y and z. Assigning positions
   * does not change it: the caller knows what its arrays hold.
   */
  sif_field_units_t units;
} sif_field_t;

/**
 * @brief Allocate an empty field header for @p n_particles particles.
 *
 * Allocates only the struct: every data array is left NULL and the field is not
 * usable until the arrays are reserved (sif_field_reserve_positions() and
 * friends) or assigned (sif_field_assign_positions() and friends).
 *
 * @param n_particles Particle count the arrays will be sized for.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on allocation failure.
 */
SIF_NODISCARD sif_field_t* sif_field_alloc(uint64_t n_particles);

/**
 * @brief Per-array stride inside a packed x/y/z block.
 *
 * Each sub-array of a unified block starts on a cache line, so a block holds
 * 3 * sif_field_padded_n(n) elements rather than 3 * n. Anything that lays out
 * its own block the same way (the chain mesh, the readers) has to agree on
 * this, so it lives here rather than being open-coded per call site.
 *
 * @param n_particles Logical element count.
 * @return The padded count.
 */
SIF_PURE_FUNCTION uint64_t sif_field_padded_n(uint64_t n_particles);

/**
 * @brief Release a field and every buffer it owns.
 * @param field Field to free. NULL is accepted and ignored.
 */
void sif_field_free(sif_field_t* field);

/**
 * @brief Allocate the position arrays without filling them.
 *
 * Sized from sif_field_t::n_particles, which must be set. Writing straight into
 * `field->x`, `field->y` and `field->z` afterwards is the zero-copy way to
 * populate a field. A no-op if the arrays already exist.
 *
 * @return SIF_OK, SIF_ERR_INVALID on an empty field, SIF_ERR_ALLOC on failure.
 */
int sif_field_reserve_positions(sif_field_t* field);

/**
 * @brief Allocate the velocity arrays without filling them.
 * @see sif_field_reserve_positions
 */
int sif_field_reserve_velocities(sif_field_t* field);

/**
 * @brief Allocate the weight array without filling it.
 * @see sif_field_reserve_positions
 */
int sif_field_reserve_weights(sif_field_t* field);

/**
 * @brief Copy positions into the field.
 *
 * @param field The field.
 * @param x X-axis positions.
 * @param y Y-axis positions.
 * @param z Z-axis positions.
 * @return SIF_OK, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC on failure.
 *
 * @note Invalidates the bounds and the Morton order, and drops the permutation
 * a previous sort had recorded: it no longer describes this data.
 */
int sif_field_assign_positions(
  sif_field_t* field, const sif_real* x, const sif_real* y, const sif_real* z);

/**
 * @brief Copy velocities into the field.
 *
 * @param field The field.
 * @param vx X-axis velocities, indexed as the positions were BEFORE any Morton
 * sort -- that is, in the caller's original particle order.
 * @param vy Y-axis velocities.
 * @param vz Z-axis velocities.
 * @return SIF_OK, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC on failure.
 *
 * @note If the field has already been Morton-sorted the incoming arrays are
 * permuted into the field's current order on the way in, so that every particle
 * keeps its own velocity.
 */
int sif_field_assign_velocities(sif_field_t* field, const sif_real* vx,
  const sif_real* vy, const sif_real* vz);

/**
 * @brief Copy per-particle weights into the field.
 *
 * @param field The field.
 * @param weights Weights in the caller's original particle order.
 * @return SIF_OK, SIF_ERR_INVALID on bad arguments, SIF_ERR_ALLOC on failure.
 *
 * @note Same permutation rule as sif_field_assign_velocities().
 */
int sif_field_assign_weights(sif_field_t* field, const sif_real* weights);

/**
 * @brief Fold every coordinate into [0, box_length) periodically.
 *
 * Everything that bins the field -- the CIC assignment, the chain mesh --
 * requires coordinates strictly inside the box, and rejects the field
 * otherwise. The common reason a periodic snapshot fails that check is not bad
 * data but rounding: in single precision the representable values near the box
 * edge are spaced ~1e-4 apart at a box of 2000, so any coordinate within half
 * that of the edge lands on exactly box_length when it is stored. Those
 * particles are one ULP from the origin, not out of bounds.
 *
 * Wrapping is only ever correct for a field that really is periodic in this
 * box, so it is never applied implicitly; call this when you know it is.
 *
 * The two populations are counted separately because they mean different
 * things. A handful of boundary folds is the expected rounding artifact. A
 * large @p n_wrapped means the coordinates were not in this box to begin with
 * -- the box length is wrong, or the data was never wrapped -- and folding them
 * produces a silently meaningless density field rather than an error.
 *
 * @param field The field, modified in place.
 * @param box_length The periodic box the field lives in.
 * @param n_boundary Optional; coordinates that sat exactly on box_length.
 * @param n_wrapped Optional; coordinates that were genuinely outside the box.
 * @return SIF_OK, or SIF_ERR_INVALID on an empty or positionless field or a
 * non-positive box.
 *
 * @note NaN cannot be repaired here and is left alone; the binning validators
 * reject it, and it is counted in neither total.
 */
int sif_field_wrap_periodic(sif_field_t* field, sif_real box_length,
  uint64_t* n_boundary, uint64_t* n_wrapped);

/**
 * @brief Shift every position by @p offset.
 *
 * What moves a survey into the box sif_finder_exodus_survey_box() chose for
 * it: data and randoms both, by the same offset. The catalogue found there is
 * moved back with sif_catalog_translate() and the negated offset.
 *
 * @param field The field, modified in place.
 * @param offset Added to x, y and z respectively.
 * @return SIF_OK, or SIF_ERR_INVALID on an empty or positionless field.
 *
 * @note Invalidates the bounds and the Morton order, whose quantization the
 * rounding of every coordinate can disturb. Keeps the permutation of a
 * previous sort, which still describes the arrays, so velocities and weights
 * assigned afterwards still land on their own particles.
 */
int sif_field_translate(sif_field_t* field, const sif_real offset[3]);

/**
 * @brief Recompute the bounding box and centre, unconditionally.
 *
 * Prefer sif_field_require_bounds() unless you specifically need to force a
 * refresh -- after writing into `field->x` directly, for instance, which the
 * field cannot notice.
 *
 * @return SIF_OK, or SIF_ERR_INVALID on an empty or positionless field.
 */
int sif_field_refresh_bounds(sif_field_t* field);

/**
 * @brief Ensure the field has a valid bounding box, computing it if needed.
 *
 * Idempotent and quiet: safe to call from anything that reads
 * sif_field_t::min_p, ::max_p, ::center or ::half_span.
 *
 * @return SIF_OK if the bounds are valid on return, an error code otherwise.
 */
int sif_field_require_bounds(sif_field_t* field);

/**
 * @brief Sort the field arrays in memory along a 3D Morton curve.
 *
 * Reorders the data physically rather than producing an index, so that
 * neighbouring particles in space end up neighbouring in memory and the
 * spatial queries walk contiguous ranges. The permutation is recorded in
 * sif_field_t::original_indices.
 *
 * @return SIF_OK, SIF_ERR_INVALID on an empty field, SIF_ERR_ALLOC on failure
 * -- in which case the field is left unmodified.
 *
 * @note On success the field owns its positions and indices regardless of how
 * they were assigned: the permuted copy cannot alias the caller's arrays.
 */
int sif_field_sort_morton(sif_field_t* field);

/**
 * @brief Ensure the field is Morton-sorted, sorting it if needed.
 *
 * Consumers that structurally depend on Morton order -- the octree, which
 * stores contiguous particle ranges per node -- should call this instead of
 * testing the flag and failing.
 *
 * @return SIF_OK if the field is Morton-sorted on return, an error otherwise.
 */
int sif_field_require_morton(sif_field_t* field);

/**
 * @brief Turn sky coordinates into comoving Cartesian positions, in place.
 *
 * Each tracer is placed at the line-of-sight comoving distance of its
 * redshift, in @p cosmo, along the direction its right ascension and
 * declination give:
 *
 *   x = D_C(z) cos(dec) cos(ra),  y = D_C(z) cos(dec) sin(ra),
 *   z = D_C(z) sin(dec),
 *
 * with the observer at the origin -- the convention of pyrecon's
 * sky_to_cartesian(). Positions come out in Mpc/h (see cosmology.h), and the
 * field is then #SIF_FIELD_CARTESIAN. Weights and velocities are untouched.
 *
 * The distances come from a table in z with exact derivatives at its nodes,
 * so a field of any size costs one integral per node, not per tracer, and
 * stays well within single precision of the exact integral.
 *
 * Every coordinate is checked before any is changed: a right ascension that
 * is not finite, a declination outside [-90, 90] or a redshift that is not
 * finite or is negative fails the call with the counts in the log, and leaves
 * the field as it was.
 *
 * @param field Field holding sky coordinates (sif_field_t::units ==
 * #SIF_FIELD_SKY), modified in place.
 * @param cosmo The cosmology the redshifts are converted in.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, a field without
 * positions or already Cartesian, a coordinate out of range, or cosmological
 * parameters sif_cosmology_comoving_distance() refuses; SIF_ERR_RANGE for a
 * cosmology with no expansion history out to the largest redshift;
 * SIF_ERR_ALLOC.
 *
 * @note Invalidates the bounds and the Morton order, as any change of
 * positions does.
 */
SIF_NODISCARD int sif_field_convert_sky_coordinates(
  sif_field_t* field, const sif_cosmology_t* cosmo);

#endif /* SIF_STRUCTURES_FIELD_H */
