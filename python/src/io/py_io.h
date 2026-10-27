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
PyObject* pysif_read_field_binary(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_catalogue_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_catalogue_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds);

/* The HDF5 catalogue file (py_hdf5.c) */
PyObject* pysif_write_catalogue_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_catalogue_hdf5(
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

/* GADGET snapshots (py_gadget.c) */
PyObject* pysif_read_gadget(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_gadget_header(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_inspect_gadget(PyObject* self, PyObject* args, PyObject* kwds);

/* FITS catalogues (py_fits.c) */
PyObject* pysif_read_fits(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_inspect_fits(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_catalogue_fits(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_catalogue_fits(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_fits_key(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_set_fits_key(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_write_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds);
PyObject* pysif_read_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds);

/* Particles from any HDF5 file (py_hdf5.c) */
PyObject* pysif_read_hdf5(PyObject* self, PyObject* args, PyObject* kwds);

/* A path (str or os.PathLike) or a sequence of them, as a new list of
 * file-system bytes objects; NULL with an exception set. */
PyObject* py_sif_paths_list(PyObject* obj);

/* 0 if the path exists; -1 with the OSError the system gives (a
 * FileNotFoundError, as open() raises) if not. */
int py_sif_require_file(const char* path);

/* How a binding takes one path as any of str, bytes or os.PathLike without
 * holding a reference it must release on every return. The binding's public
 * function is
 *
 *     return py_sif_call_with_path(impl, self, args, kwds, "filepath");
 *
 * which converts the path argument -- the first positional one, or the
 * keyword `name` -- to file-system bytes (os.fsencode) and calls impl with
 * them in its place. impl parses it with "O&", py_sif_path_arg, &filepath,
 * which borrows the bytes' const char*: alive until impl returns, since the
 * arguments impl was called with hold them. */
PyObject* py_sif_call_with_path(PyCFunctionWithKeywords impl, PyObject* self,
  PyObject* args, PyObject* kwds, const char* name);
int py_sif_path_arg(PyObject* obj, void* out);

#endif /* SIF_PY_IO_PY_IO_H */