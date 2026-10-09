/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* What the functions that take a field share: the checks on what it holds,
 * and the messages that say what it lacks. */

#ifndef SIF__STRUCTURES_FIELD_INTERNAL_H
#define SIF__STRUCTURES_FIELD_INTERNAL_H

#include <float.h>

#include "sif/structures/field.h"

/*
 * The box faces, and the rounding around them.
 *
 * A coordinate of a periodic snapshot is in [0, box_length) by construction,
 * but not always once it has been stored: converted to single precision, a
 * value within half an ULP of the edge becomes exactly box_length, and unit
 * conversions or a shift done in another precision can land a few ULPs either
 * side. Those particles are at the edge, not outside the box, and refusing
 * them asked the caller to wrap a field that was never wrong.
 *
 * So everything that requires positions in the box accepts this much on
 * either side of it -- a few ULPs of box_length at sif_real precision, about
 * 5e-4 at a box of 1000 in single precision -- and folds such a coordinate
 * periodically. Anything further out still means the box is not the one the
 * data lives in, and is still refused.
 */
#define SIF__BOX_EDGE_ULPS 4

#ifdef SIF_USE_DOUBLE
#  define SIF__REAL_EPSILON DBL_EPSILON
#else
#  define SIF__REAL_EPSILON FLT_EPSILON
#endif

static inline double sif__box_edge_tolerance(double box_length) {
  return box_length * SIF__BOX_EDGE_ULPS * SIF__REAL_EPSILON;
}

/* Whether `v` is inside the box or within rounding of it. False for NaN. In
 * double, so the file checkers can ask it of values they have not yet stored.
 */
static inline int sif__box_accepts(double v, double box_length) {
  const double tol = sif__box_edge_tolerance(box_length);
  return v >= -tol && v <= box_length + tol;
}

/*
 * A coordinate sif__box_accepts() let through, folded into [0, box_length).
 * The final test is not redundant: v + box_length for a tiny negative v rounds
 * to box_length itself, which periodically is 0.
 */
static inline sif_real sif__box_fold(sif_real v, sif_real box_length) {
  if (v >= box_length)
    v -= box_length;
  else if (v < 0)
    v += box_length;
  return v >= box_length ? (sif_real)0 : v;
}

/* How much of a field a function needs, each level including the ones
 * before it. */
typedef enum {
  /* Non-NULL, with particles. */
  SIF__FIELD_NEED_PARTICLES,
  /* ...and positions in them. */
  SIF__FIELD_NEED_POSITIONS,
  /* ...and positions that are lengths, not sky coordinates. */
  SIF__FIELD_NEED_CARTESIAN
} sif__field_need_t;

/*
 * SIF_OK if `field` has what `need` asks for; otherwise SIF_ERR_INVALID, with
 * one error logged under `tag` that names the field as `name` ("field",
 * "randoms", ...) and says which part is missing:
 *
 *   field: NULL
 *   field: 0 particles (none were given, or a consuming chain mesh build...)
 *   field: no positions (1000 particles declared, none assigned or read)
 *   field: sky coordinates (ra, dec, z), not positions; ...
 */
int sif__field_check(const sif_field_t* field, sif__field_need_t need,
  const char* tag, const char* name);

/* sif__field_check() at SIF__FIELD_NEED_CARTESIAN for a field the caller has
 * already checked for particles: SIF_OK for no field, or one of Cartesian
 * positions; SIF_ERR_INVALID, logged, for one still holding sky coordinates,
 * which binning, sorting, bounding or moving would read as lengths. */
int sif__field_require_cartesian(
  const sif_field_t* field, const char* tag, const char* name);

/*
 * SIF_OK if every position is one sif__box_accepts(); otherwise SIF_ERR_RANGE,
 * logged under `tag` with the count and the first particle outside,
 * coordinates included -- NaN counts as outside. For the structures anchored
 * at the origin (a grid, a chain mesh), which would otherwise fold such a
 * particle into a boundary cell.
 *
 * `n_edge`, if not NULL, receives the number of particles accepted only
 * through the tolerance, which the caller has to fold with sif__box_fold();
 * a count above zero is also logged, so that the fold is never silent.
 */
int sif__field_check_in_box(const sif_real* x, const sif_real* y,
  const sif_real* z, uint64_t n, sif_real box_length, const char* tag,
  const char* name, uint64_t* n_edge);

#endif /* SIF__STRUCTURES_FIELD_INTERNAL_H */
