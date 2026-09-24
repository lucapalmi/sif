/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_IO_PY_IO_H
#define SIF_PY_IO_PY_IO_H

#include "py_common.h"

/* Module-level functional entry points for discrete file formats */
PyObject* pysif_write_field(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_field(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_field_header(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_grid(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_grid(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_field_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_catalog_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_catalog_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);

/* The HDF5 catalogue file (py_hdf5.c) */
PyObject* pysif_write_catalog_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_catalog_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_profiles_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_profiles_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_size_function_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_size_function_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_set_hdf5_attr(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_get_hdf5_attr(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_get_hdf5_attrs(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_IO_PY_IO_H */