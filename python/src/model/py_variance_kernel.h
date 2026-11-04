/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MODEL_PY_VARIANCE_KERNEL_H
#define SIF_PY_MODEL_PY_VARIANCE_KERNEL_H

#include "py_common.h"
#include "sif/model/variance_kernel.h"

typedef struct {
  PyObject_HEAD sif_variance_kernel_t* kernel;
} sifVarianceKernelObject;

typedef struct {
  PyObject_HEAD sif_walk_t* walk;
} sifWalkObject;

extern PyTypeObject sifVarianceKernelType;
extern PyTypeObject sifWalkType;

#endif /* SIF_PY_MODEL_PY_VARIANCE_KERNEL_H */
