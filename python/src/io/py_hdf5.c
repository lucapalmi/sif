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
#include "structures/py_catalog.h"
#include "structures/py_size_function.h"

#include "sif/io/hdf5_io.h"

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

/* After a failed read: which of the two kinds of failure it was. */
static PyObject* read_error(const char* filepath) {
#ifdef SIF_HAVE_HDF5
  return PyErr_Format(
    PyExc_OSError, "failed to read %s; see the log for the reason", filepath);
#else
  return PyErr_Format(PyExc_RuntimeError,
    "cannot read %s: pysif was built without HDF5 support (rebuild with "
    "-C cmake.define.SIF_HDF5_SUPPORT=ON)",
    filepath);
#endif
}

static PyObject* write_status(int status, const char* filepath) {
  if (status == SIF_ERR_INVALID)
    return PyErr_Format(PyExc_ValueError,
      "nothing to write to %s; see the log for the reason", filepath);
  if (status != SIF_OK)
    return PyErr_Format(PyExc_OSError,
      "failed to write %s; see the log for the reason (an existing file that "
      "sif did not write is never overwritten)",
      filepath);
  if (warn_if_fallback(filepath) < 0)
    return NULL;
  Py_RETURN_NONE;
}

/* --- catalogue --- */

PyObject* pysif_write_catalog_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* cat_obj;
  static char* kwlist[] = {"filepath", "catalog", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifCatalogType, &cat_obj))
    return NULL;

  return write_status(
    sif_catalog_write_hdf5(filepath, ((sifCatalogObject*)cat_obj)->catalog),
    filepath);
}

PyObject* pysif_read_catalog_hdf5(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  sif_catalog_t* cat = sif_catalog_read_hdf5(filepath);
  if (!cat)
    return read_error(filepath);

  sifCatalogObject* obj =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!obj) {
    sif_catalog_free(cat);
    return PyErr_NoMemory();
  }
  obj->catalog = cat;
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
  if (sif_profiles_read_header_hdf5(filepath, &has_dens, &has_vel) != SIF_OK)
    return read_error(filepath);

  if (!has_dens && !has_vel)
    return PyErr_Format(PyExc_ValueError, "%s holds no profiles", filepath);

  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  if (sif_profiles_read_hdf5(
        filepath, has_dens ? &dens : NULL, has_vel ? &vel : NULL) != SIF_OK)
    return read_error(filepath);

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

  sif_size_function_t* vsf = sif_size_function_read_hdf5(filepath);
  if (!vsf)
    return read_error(filepath);

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
  if (sif_hdf5_attr_kind(filepath, group, key, &kind) != SIF_OK)
    return read_error(filepath);

  switch (kind) {
  case SIF_HDF5_ATTR_INT: {
    int64_t v;
    if (sif_hdf5_get_attr_int(filepath, group, key, &v) != SIF_OK)
      return read_error(filepath);
    return PyLong_FromLongLong((long long)v);
  }
  case SIF_HDF5_ATTR_REAL: {
    double v;
    if (sif_hdf5_get_attr_real(filepath, group, key, &v) != SIF_OK)
      return read_error(filepath);
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
        return read_error(filepath);
      len *= 4;
    }
  }
  default:
    if (skip_other)
      Py_RETURN_NONE;
    return PyErr_Format(PyExc_TypeError,
      "'%s' is not a single integer, number or string; read it with h5py", key);
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

  if (status == SIF_ERR_INVALID)
    return PyErr_Format(PyExc_ValueError,
      "cannot set '%s' in %s: the group is missing, or the name is one sif "
      "keeps for itself (see the log)",
      key, filepath);
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
  if (sif_hdf5_attr_count(filepath, group, &n) != SIF_OK)
    return read_error(filepath);

  PyObject* out = PyDict_New();
  if (!out)
    return NULL;

  for (uint32_t i = 0; i < n; i++) {
    char name[1024];
    if (sif_hdf5_attr_name(filepath, group, i, name, sizeof(name)) != SIF_OK) {
      Py_DECREF(out);
      return read_error(filepath);
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
