/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MEASURE_PY_DELTA_H
#define SIF_PY_MEASURE_PY_DELTA_H

#include "py_common.h"

PyObject* py_sif_delta_distribution_grid(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_moments_grid(
  PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_MEASURE_PY_DELTA_H */
