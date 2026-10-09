/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_FIELD_H
#define SIF_PY_STRUCTURES_PY_FIELD_H

#include "py_common.h"
#include "sif/structures/field.h"

typedef struct {
  PyObject_HEAD sif_field_t* field;
  /* NumPy views into the field's arrays that are still alive. While there are
   * any, nothing may reallocate or take those arrays. */
  Py_ssize_t n_exports;
} sifFieldObject;

extern PyTypeObject sifFieldType;

/*
 * Refuse, with a BufferError, an operation that would reallocate or take the
 * field's arrays while NumPy views into them are alive -- they would be left
 * pointing at freed memory. The same rule bytearray applies to resizing.
 *
 * @param action What was about to happen, for the message: "sort_morton()".
 * @return 0 if there are no views, -1 with the exception set if there are.
 */
int py_sif_field_check_exports(sifFieldObject* self, const char* action);

/*
 * Refuse, with a ValueError, an operation that reads positions as lengths --
 * binning, sorting, bounding, moving -- on a field that still holds sky
 * coordinates. The C side refuses too, but its remedy is spelled for C.
 *
 * @param action What was about to happen, for the message: "build a mesh".
 * @return 0 for Cartesian positions, -1 with the exception set otherwise.
 */
int py_sif_field_check_cartesian(sifFieldObject* self, const char* action);

/*
 * py_sif_raise() for a failed deposit or mesh build, with the Python remedy
 * added to the library's message when particles lay outside the box -- the
 * refusal a caller most often meets, and one whose C remedy names no Python
 * method. Always returns NULL.
 */
PyObject* py_sif_field_raise_binning(int status, const char* what);

/*
 * After a wrap: a RuntimeWarning if `n_nonfinite` coordinates were NaN or
 * infinite, which the C side only logs -- and a Python caller usually has the
 * log off. Returns -1 if the warning was turned into an exception.
 */
int py_sif_field_warn_nonfinite(uint64_t n_nonfinite);

/* pysif.field_from_numpy(): a new Field from NumPy arrays. */
PyObject* pysif_field_from_numpy(
  PyObject* module, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_STRUCTURES_PY_FIELD_H */