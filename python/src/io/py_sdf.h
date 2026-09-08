/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_IO_PY_SDF_H
#define SIF_PY_IO_PY_SDF_H

#include "py_common.h"
#include "sif/io/sdf.h"

/*
 * @brief Python object wrapping an open .sdf file.
 *
 * The handle is closed by close(), by leaving a `with` block, or by
 * collection, whichever comes first; every method checks that it is still
 * open, so a closed file raises rather than crashing.
 */
typedef struct {
  PyObject_HEAD sif_sdf_t* file;
} sifSDFObject;

extern PyTypeObject sifSDFType;

/* Module-level entry points: checking and recovering a file take a path
 * rather than a handle, since a damaged file cannot be opened. */
PyObject* pysif_sdf_verify(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_sdf_repair(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_IO_PY_SDF_H */
