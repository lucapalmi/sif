/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_PY_COMMON_H
#define SIF_PY_PY_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "sif/core/macros.h"

#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pysif_ARRAY_API
#ifndef PYSIF_MAIN_MODULE
#  define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>

#ifdef SIF_USE_DOUBLE
#  define NPY_REAL_T NPY_FLOAT64
#else
#  define NPY_REAL_T NPY_FLOAT32
#endif

/*
 * @brief Wraps a buffer owned by a C struct as a NumPy array, without copying.
 *
 * The array keeps `owner` alive through its base, and is read-only: the buffer
 * belongs to the owner, and a stray in-place write from Python would corrupt
 * it. Returns NULL with an exception set on failure.
 */
static inline PyObject* py_sif_wrap_borrowed(
  PyObject* owner, int ndim, npy_intp* dims, void* data) {

  PyObject* array = PyArray_SimpleNewFromData(ndim, dims, NPY_REAL_T, data);
  if (!array)
    return NULL;

  Py_INCREF(owner);
  if (PyArray_SetBaseObject((PyArrayObject*)array, owner) < 0) {
    /* SetBaseObject steals the reference even when it fails. */
    Py_DECREF(array);
    return NULL;
  }

  PyArray_CLEARFLAGS((PyArrayObject*)array, NPY_ARRAY_WRITEABLE);
  return array;
}

#endif /* SIF_PY_PY_COMMON_H */