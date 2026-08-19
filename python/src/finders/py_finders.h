/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_FINDERS_PY_FINDERS_H
#define SIF_PY_FINDERS_PY_FINDERS_H

#include "py_common.h"

/* Module-level functional entry points */
PyObject* py_sif_finder_exodus(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_sif_finder_spherical(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_sif_finder_suggest_mesh_cells(
  PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_FINDERS_PY_FINDERS_H */