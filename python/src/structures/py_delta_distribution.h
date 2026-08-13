/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_DELTA_DISTRIBUTION_H
#define SIF_PY_STRUCTURES_PY_DELTA_DISTRIBUTION_H

#include "py_common.h"
#include "sif/structures/delta_distribution.h"

typedef struct {
  PyObject_HEAD sif_delta_distribution_t* dist;
} sifDeltaDistributionObject;

extern PyTypeObject sifDeltaDistributionType;

PyObject* py_sif_wrap_distribution(sif_delta_distribution_t* dist);

#endif /* SIF_PY_STRUCTURES_PY_DELTA_DISTRIBUTION_H */
