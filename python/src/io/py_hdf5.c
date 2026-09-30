/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The HDF5 catalogue file from Python. Thin: the C side decides everything,
 * including the plain-text fallback of a build without HDF5. What is added
 * here is what a Python caller cannot see otherwise -- the C log -- turned
 * into a RuntimeWarning when a write fell back, and into an exception when a
 * read cannot happen.
 *
 * The GIL is held throughout. HDF5 as commonly built is not thread-safe, and
 * releasing the GIL would let two Python threads into it at once; the files
 * are small next to what the finders cost, so nothing is lost by it.
 */

#include "io/py_io.h"

#include "measure/py_profiles.h"
#include "structures/py_catalogue.h"
#include "structures/py_field.h"
#include "structures/py_size_function.h"

#include "sif/io/hdf5_io.h"

#include <stdio.h>
#include <string.h>

/* After a successful write in a build without HDF5: the data is safe, in
 * text, and the caller should hear where. -1 if the warning was turned into
 * an exception by the caller's warning filters. */
static int warn_if_fallback(const char* filepath) {
#ifdef SIF_HAVE_HDF5
  (void)filepath;
  return 0;
#else
  return PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
    "pysif was built without HDF5, so %s was not written; the data was "
    "saved as plain text in %s.<product>.txt instead",
    filepath, filepath);
#endif
}

/* After a failed read, with the C library's reason; @p status is SIF_ERR_IO
 * for a reader that only returns NULL. A build without HDF5 says how to get
 * it, in pip's terms rather than CMake's. */
static PyObject* read_error(int status, const char* filepath) {
#ifdef SIF_HAVE_HDF5
  return py_sif_raise(status, filepath);
#else
  (void)status;
  return PyErr_Format(PyExc_RuntimeError,
    "%s: HDF5 support not built (rebuild pysif with "
    "-C cmake.define.SIF_HDF5_SUPPORT=ON)",
    filepath);
#endif
}

/* After a write; the caller clears the error record before making it. */
static PyObject* write_status(int status, const char* filepath) {
  if (status != SIF_OK)
    return py_sif_raise(status, filepath);
  if (warn_if_fallback(filepath) < 0)
    return NULL;
  Py_RETURN_NONE;
}

/* --- catalogue --- */

PyObject* pysif_write_catalogue_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* cat_obj;
  static char* kwlist[] = {"filepath", "catalogue", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifCatalogueType, &cat_obj))
    return NULL;

  sif_error_clear();
  return write_status(
    sif_catalogue_write_hdf5(filepath, ((sifCatalogueObject*)cat_obj)->catalogue),
    filepath);
}

PyObject* pysif_read_catalogue_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  sif_error_clear();
  sif_catalogue_t* cat = sif_catalogue_read_hdf5(filepath);
  if (!cat)
    return read_error(SIF_ERR_IO, filepath);

  sifCatalogueObject* obj =
    (sifCatalogueObject*)sifCatalogueType.tp_alloc(&sifCatalogueType, 0);
  if (!obj) {
    sif_catalogue_free(cat);
    return PyErr_NoMemory();
  }
  obj->catalogue = cat;
  return (PyObject*)obj;
}

/* --- profiles --- */

PyObject* pysif_write_profiles_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* prof_obj;
  static char* kwlist[] = {"filepath", "profiles", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifProfilesType, &prof_obj))
    return NULL;

  const sifProfilesObject* prof = (const sifProfilesObject*)prof_obj;
  sif_error_clear();
  return write_status(
    sif_profiles_write_hdf5(filepath, prof->dens, prof->vel), filepath);
}

PyObject* pysif_read_profiles_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  /* Whatever sets the file holds, which the C reader will not guess. */
  int has_dens = 0, has_vel = 0;
  sif_error_clear();
  int status = sif_profiles_read_header_hdf5(filepath, &has_dens, &has_vel);
  if (status != SIF_OK)
    return read_error(status, filepath);

  if (!has_dens && !has_vel)
    return PyErr_Format(PyExc_ValueError,
      "%s: no /density_profiles or /velocity_profiles", filepath);

  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  status = sif_profiles_read_hdf5(
    filepath, has_dens ? &dens : NULL, has_vel ? &vel : NULL);
  if (status != SIF_OK)
    return read_error(status, filepath);

  sifProfilesObject* prof =
    (sifProfilesObject*)sifProfilesType.tp_alloc(&sifProfilesType, 0);
  if (!prof) {
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    return PyErr_NoMemory();
  }
  prof->dens = dens;
  prof->vel = vel;
  return (PyObject*)prof;
}

/* --- size function --- */

PyObject* pysif_write_size_function_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* vsf_obj;
  static char* kwlist[] = {"filepath", "size_function", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifSizeFunctionType, &vsf_obj))
    return NULL;

  sif_error_clear();
  return write_status(sif_size_function_write_hdf5(
                        filepath, ((sifSizeFunctionObject*)vsf_obj)->vsf),
    filepath);
}

PyObject* pysif_read_size_function_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  sif_error_clear();
  sif_size_function_t* vsf = sif_size_function_read_hdf5(filepath);
  if (!vsf)
    return read_error(SIF_ERR_IO, filepath);

  sifSizeFunctionObject* obj =
    (sifSizeFunctionObject*)sifSizeFunctionType.tp_alloc(
      &sifSizeFunctionType, 0);
  if (!obj) {
    sif_size_function_free(vsf);
    return PyErr_NoMemory();
  }
  obj->vsf = vsf;
  return (PyObject*)obj;
}

/* --- free-form metadata --- */

/* One entry, as the Python value it holds; NULL with an exception set. An
 * entry of a kind these functions do not read comes back as None when
 * listing a header (`skip_other`), and as an error when asked for by name. */
static PyObject* meta_value(
  const char* filepath, const char* group, const char* key, int skip_other) {

  sif_hdf5_attr_kind_t kind;
  sif_error_clear();
  int status = sif_hdf5_attr_kind(filepath, group, key, &kind);
  if (status != SIF_OK)
    return read_error(status, filepath);

  switch (kind) {
  case SIF_HDF5_ATTR_INT: {
    int64_t v;
    status = sif_hdf5_get_attr_int(filepath, group, key, &v);
    if (status != SIF_OK)
      return read_error(status, filepath);
    return PyLong_FromLongLong((long long)v);
  }
  case SIF_HDF5_ATTR_REAL: {
    double v;
    status = sif_hdf5_get_attr_real(filepath, group, key, &v);
    if (status != SIF_OK)
      return read_error(status, filepath);
    return PyFloat_FromDouble(v);
  }
  case SIF_HDF5_ATTR_STRING: {
    /* Grown until it fits: a header value has no length limit. */
    size_t len = 256;
    for (;;) {
      char* buf = PyMem_Malloc(len);
      if (!buf)
        return PyErr_NoMemory();
      const int st = sif_hdf5_get_attr_string(filepath, group, key, buf, len);
      if (st == SIF_OK) {
        PyObject* out = PyUnicode_FromString(buf);
        PyMem_Free(buf);
        return out;
      }
      PyMem_Free(buf);
      if (st != SIF_ERR_RANGE)
        return read_error(st, filepath);
      len *= 4;
    }
  }
  default:
    if (skip_other)
      Py_RETURN_NONE;
    return PyErr_Format(PyExc_TypeError,
      "%s: '%s': not a scalar integer, number or string (read it with h5py)",
      filepath, key);
  }
}

PyObject* pysif_set_hdf5_attr(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* key;
  PyObject* value;
  const char* group = NULL;
  static char* kwlist[] = {"filepath", "key", "value", "group", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "ssO|z", kwlist, &filepath, &key, &value, &group))
    return NULL;

  int status;
  sif_error_clear();

  /* bool before int, since bool is one; anything with __index__ (a numpy
   * integer) as an integer; anything with __float__ as a real. */
  if (PyUnicode_Check(value)) {
    const char* str = PyUnicode_AsUTF8(value);
    if (!str)
      return NULL;
    status = sif_hdf5_set_attr_string(filepath, group, key, str);
  } else if (PyBool_Check(value) || PyIndex_Check(value)) {
    PyObject* idx = PyNumber_Index(value);
    if (!idx)
      return NULL;
    const long long v = PyLong_AsLongLong(idx);
    Py_DECREF(idx);
    if (v == -1 && PyErr_Occurred())
      return NULL;
    status = sif_hdf5_set_attr_int(filepath, group, key, (int64_t)v);
  } else if (PyFloat_Check(value) || PyNumber_Check(value)) {
    const double v = PyFloat_AsDouble(value);
    if (v == -1.0 && PyErr_Occurred())
      return NULL;
    status = sif_hdf5_set_attr_real(filepath, group, key, v);
  } else {
    return PyErr_Format(PyExc_TypeError,
      "metadata values are int, float or str, not %s", Py_TYPE(value)->tp_name);
  }

  return write_status(status, filepath);
}

PyObject* pysif_get_hdf5_attr(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* key;
  const char* group = NULL;
  static char* kwlist[] = {"filepath", "key", "group", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "ss|z", kwlist, &filepath, &key, &group))
    return NULL;

  return meta_value(filepath, group, key, 0);
}

PyObject* pysif_get_hdf5_attrs(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* group = NULL;
  static char* kwlist[] = {"filepath", "group", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|z", kwlist, &filepath, &group))
    return NULL;

  uint32_t n = 0;
  sif_error_clear();
  const int status = sif_hdf5_attr_count(filepath, group, &n);
  if (status != SIF_OK)
    return read_error(status, filepath);

  PyObject* out = PyDict_New();
  if (!out)
    return NULL;

  for (uint32_t i = 0; i < n; i++) {
    char name[1024];
    const int st = sif_hdf5_attr_name(filepath, group, i, name, sizeof(name));
    if (st != SIF_OK) {
      Py_DECREF(out);
      return read_error(st, filepath);
    }

    PyObject* v = meta_value(filepath, group, name, 1);
    if (!v || PyDict_SetItemString(out, name, v) < 0) {
      Py_XDECREF(v);
      Py_DECREF(out);
      return NULL;
    }
    Py_DECREF(v);
  }

  return out;
}

/* --- particles from any HDF5 file --- */

PyObject* pysif_read_hdf5(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* paths_obj;
  const char *x = NULL, *y = NULL, *z = NULL, *ra = NULL, *dec = NULL;
  const char *vx = NULL, *vy = NULL, *vz = NULL, *w = NULL;
  double length_scale = 1.0, fraction = 1.0;
  unsigned long long seed = 0;
  static char* kwlist[] = {"paths", "x", "y", "z", "ra", "dec", "vx", "vy",
    "vz", "w", "length_scale", "fraction", "seed", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$zzzzzzzzzddK", kwlist,
        &paths_obj, &x, &y, &z, &ra, &dec, &vx, &vy, &vz, &w, &length_scale,
        &fraction, &seed))
    return NULL;

#ifndef SIF_HAVE_HDF5
  (void)paths_obj;
  return PyErr_Format(PyExc_RuntimeError,
    "pysif was built without HDF5 support (rebuild with "
    "-C cmake.define.SIF_HDF5_SUPPORT=ON)");
#else
  /* The C side refuses all of these too, but only says why in the log. */
  if ((x || y) && (ra || dec))
    return PyErr_Format(PyExc_ValueError,
      "give x, y and z for positions, or ra, dec and z for sky coordinates, "
      "not some of each");
  if ((ra || dec) ? !(ra && dec && z) : !(x && y && z))
    return PyErr_Format(PyExc_TypeError,
      "read_hdf5() needs the datasets x, y and z -- or ra, dec and z, for sky "
      "coordinates");
  if ((!!vx + !!vy + !!vz) % 3 != 0)
    return PyErr_Format(
      PyExc_ValueError, "vx, vy and vz are given all three or not at all");
  if (!(fraction > 0 && fraction <= 1))
    return PyErr_Format(PyExc_ValueError, "fraction must be in (0, 1]");
  if (!(length_scale > 0) || ((ra || dec) && length_scale != 1.0))
    return PyErr_Format(PyExc_ValueError,
      "length_scale must be positive, and 1 for sky coordinates");

  PyObject* list = py_sif_paths_list(paths_obj);
  if (!list)
    return NULL;
  const Py_ssize_t n = PyList_GET_SIZE(list);
  const char** paths = PyMem_Malloc((size_t)n * sizeof(char*));
  PyObject* result = NULL;
  if (!paths) {
    PyErr_NoMemory();
    goto done;
  }
  for (Py_ssize_t i = 0; i < n; i++)
    paths[i] = PyBytes_AS_STRING(PyList_GET_ITEM(list, i));

  const sif_field_columns_t columns = {.x = x,
    .y = y,
    .ra = ra,
    .dec = dec,
    .z = z,
    .vx = vx,
    .vy = vy,
    .vz = vz,
    .w = w};
  sif_error_clear();
  sif_field_t* field = sif_field_read_hdf5(
    paths, (uint32_t)n, &columns, length_scale, fraction, (uint64_t)seed);
  if (!field) {
    py_sif_raise(sif_error_status(), paths[0]);
    goto done;
  }
  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    PyErr_NoMemory();
    goto done;
  }
  obj->field = field;
  result = (PyObject*)obj;

done:
  PyMem_Free(paths);
  Py_DECREF(list);
  return result;
#endif
}
