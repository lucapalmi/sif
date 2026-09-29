/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* A comoving-distance table, for converting many redshifts at once: the
 * distance and its derivative, c / H(z), on an even grid in z, interpolated
 * with cubic Hermite polynomials. Both values at a node are exact to about
 * machine precision, so the interpolation error is the Hermite one, of order
 * dz^4 -- far below single precision at the spacing used. One integral per
 * node rather than one per tracer. */

#ifndef SIF__MODEL_COSMOLOGY_INTERNAL_H
#define SIF__MODEL_COSMOLOGY_INTERNAL_H

#include "sif/model/cosmology.h"

#include <stdint.h>

typedef struct {
  /** Node spacing in z; node i is at i * dz. */
  double dz;
  /** Nodes, at least 2. */
  uint64_t n;
  /** D_C at each node, in Mpc/h. */
  double* distance;
  /** dD_C/dz = c / H(z) at each node, in Mpc/h. */
  double* slope;
} sif__distance_table_t;

/* Build the table from z = 0 to at least z_max. Returns SIF_OK, or as
 * sif_cosmology_comoving_distance() for the parameters and the range, or
 * SIF_ERR_ALLOC; on failure the table holds nothing to free. */
int sif__distance_table_build(
  const sif_cosmology_t* cosmo, double z_max, sif__distance_table_t* table);

/* D_C(z) for z in [0, z_max] of the table. */
double sif__distance_table_eval(const sif__distance_table_t* table, double z);

void sif__distance_table_free(sif__distance_table_t* table);

#endif /* SIF__MODEL_COSMOLOGY_INTERNAL_H */
