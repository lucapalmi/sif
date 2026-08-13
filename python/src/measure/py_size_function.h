/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MEASURE_PY_SIZE_FUNCTION_H
#define SIF_PY_MEASURE_PY_SIZE_FUNCTION_H

#include "py_common.h"

PyObject* py_sif_size_function_catalog(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_combine(
  PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_MEASURE_PY_SIZE_FUNCTION_H */
