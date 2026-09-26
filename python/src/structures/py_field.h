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

#endif /* SIF_PY_STRUCTURES_PY_FIELD_H */