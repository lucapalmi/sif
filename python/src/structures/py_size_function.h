/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_SIZE_FUNCTION_H
#define SIF_PY_STRUCTURES_PY_SIZE_FUNCTION_H

#include "py_common.h"
#include "sif/structures/size_function.h"

typedef struct {
  PyObject_HEAD sif_size_function_t* vsf;
} sifSizeFunctionObject;

extern PyTypeObject sifSizeFunctionType;

/*
 * @brief Wraps a freshly computed size function, taking ownership.
 *
 * Shared by the measure-side and (eventually) model-side producers.
 */
PyObject* py_sif_wrap_size_function(sif_size_function_t* vsf);

#endif /* SIF_PY_STRUCTURES_PY_SIZE_FUNCTION_H */
