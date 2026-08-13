/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_DELTA_MOMENTS_H
#define SIF_PY_STRUCTURES_PY_DELTA_MOMENTS_H

#include "py_common.h"
#include "sif/structures/delta_moments.h"

typedef struct {
  PyObject_HEAD sif_delta_moments_t* moments;
} sifDeltaMomentsObject;

extern PyTypeObject sifDeltaMomentsType;

/*
 * @brief Wraps a freshly computed moment set, taking ownership.
 *
 * Shared by the measure-side and model-side producers.
 */
PyObject* py_sif_wrap_moments(sif_delta_moments_t* moments);

#endif /* SIF_PY_STRUCTURES_PY_DELTA_MOMENTS_H */
