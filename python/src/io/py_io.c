/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_common.h"

#include "io/py_io.h"

#include "measure/py_profiles.h"
#include "sif/io/profiles_io.h"
#include "structures/py_catalogue.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"

#include "sif/io/catalogue_io.h"
#include "sif/io/field_io.h"
#include "sif/io/grid_io.h"

#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* --- Field I/O --- */

PyObject* pysif_write_field(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  double box_length;
  PyObject *x_obj, *y_obj, *z_obj;
  PyObject *vx_obj = NULL, *vy_obj = NULL, *vz_obj = NULL;
  PyObject* weights_obj = NULL;

  static char* kwlist[] = {
    "filepath", "box_length", "x", "y", "z", "vx", "vy", "vz", "weights", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sdOOO|OOOO", kwlist, &filepath,
        &box_length, &x_obj, &y_obj, &z_obj, &vx_obj, &vy_obj, &vz_obj,
        &weights_obj)) {
    return NULL;
  }

  /* 1. Safely cast Python objects to contiguous NumPy arrays without copying */
  PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(
    x_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(
    y_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(
    z_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

  if (!x_arr || !y_arr || !z_arr) {
    Py_XDECREF(x_arr);
    Py_XDECREF(y_arr);
    Py_XDECREF(z_arr);
    return PyErr_Format(
      PyExc_TypeError, "Positions must be 1D numpy arrays of type float32/64");
  }

  uint64_t n_particles = PyArray_SIZE(x_arr);

  /* 2. Temporarily wrap the NumPy pointers into our C struct (Zero-Copy) */
  sif_field_t temp_field;
  memset(&temp_field, 0, sizeof(sif_field_t));
  temp_field.n_particles = n_particles;
  temp_field.x = (sif_real*)PyArray_DATA(x_arr);
  temp_field.y = (sif_real*)PyArray_DATA(y_arr);
  temp_field.z = (sif_real*)PyArray_DATA(z_arr);

  PyArrayObject *vx_arr = NULL, *vy_arr = NULL, *vz_arr = NULL, *m_arr = NULL;

  if (vx_obj && vx_obj != Py_None && vy_obj && vy_obj != Py_None && vz_obj &&
      vz_obj != Py_None) {
    vx_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vx_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vy_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vy_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vz_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vz_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (vx_arr && vy_arr && vz_arr) {
      temp_field.vx = (sif_real*)PyArray_DATA(vx_arr);
      temp_field.vy = (sif_real*)PyArray_DATA(vy_arr);
      temp_field.vz = (sif_real*)PyArray_DATA(vz_arr);
    }
  }

  if (weights_obj && weights_obj != Py_None) {
    m_arr = (PyArrayObject*)PyArray_FROM_OTF(
      weights_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (m_arr)
      temp_field.weights = (sif_real*)PyArray_DATA(m_arr);
  }

  /* 3. Release the GIL so Python can do other things while NVMe writes */
  int status = 0;
  Py_BEGIN_ALLOW_THREADS status =
    sif_field_write(filepath, &temp_field, box_length);
  Py_END_ALLOW_THREADS

    /* 4. Cleanup NumPy references */
    Py_DECREF(x_arr);
  Py_DECREF(y_arr);
  Py_DECREF(z_arr);
  Py_XDECREF(vx_arr);
  Py_XDECREF(vy_arr);
  Py_XDECREF(vz_arr);
  Py_XDECREF(m_arr);

  if (status != 0) {
    return PyErr_Format(
      PyExc_IOError, "Failed to write .xfield to %s", filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_field(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  int wrap = 0;
  static char* kwlist[] = {"filepath", "wrap", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|p", kwlist, &filepath, &wrap)) {
    return NULL;
  }

  double box_length = 0.0;
  sif_field_t* field = NULL;

  /* Drop GIL while 144 threads hit the disk via parallel pread() */
  Py_BEGIN_ALLOW_THREADS field = sif_field_read(filepath, &box_length);
  Py_END_ALLOW_THREADS

    if (!field) {
    return PyErr_Format(
      PyExc_IOError, "Failed to read .xfield from %s", filepath);
  }

  /* Off by default: folding coordinates is only ever right for a field that is
   * periodic in this box, and doing it unasked would turn a wrong box length --
   * which the binning validators currently catch loudly -- into a silently
   * meaningless density field. The box used is the file's own, so this cannot
   * paper over a box the caller got wrong; that one is still checked where the
   * caller supplies it. */
  if (wrap) {
    int status = SIF_OK;
    Py_BEGIN_ALLOW_THREADS status =
      sif_field_wrap_periodic(field, (sif_real)box_length, NULL, NULL);
    Py_END_ALLOW_THREADS

      if (status != SIF_OK) {
      sif_field_free(field);
      /* PyErr_Format has no float conversion, so the box has to be rendered
       * before it gets there. */
      char detail[512];
      snprintf(detail, sizeof(detail),
        "wrap=True, but %s declares a box length of %g, which cannot be "
        "wrapped into",
        filepath, box_length);
      PyErr_SetString(PyExc_ValueError, detail);
      return NULL;
    }
  }

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }

  obj->field = field;
  return (PyObject*)obj;
}

PyObject* pysif_read_field_header(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    return PyErr_Format(PyExc_IOError, "Failed to open %s", filepath);
  }

  sif_xfield_header_t header;
  const size_t got = fread(&header, sizeof(header), 1, file);
  fclose(file);

  if (got != 1) {
    return PyErr_Format(
      PyExc_IOError, "%s is too short to hold an .xfield header", filepath);
  }

  if (strncmp(header.magic, SIF_XFIELD_MAGIC, 4) != 0) {
    return PyErr_Format(
      PyExc_ValueError, "%s is not an .xfield file", filepath);
  }

  /* box_length is the reason this exists: it lives in the file, not in
   * sif_field_t, so reading the field is not a way to recover it. */
  return Py_BuildValue("{s:K,s:d,s:O,s:O,s:I}", "n_particles",
    (unsigned long long)header.n_particles, "box_length", header.box_length,
    "has_weights", header.has_weights ? Py_True : Py_False, "has_velocities",
    header.has_velocities ? Py_True : Py_False, "version",
    (unsigned int)header.version);
}

/* --- Grid I/O --- */

PyObject* pysif_write_grid(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* grid_obj;

  static char* kwlist[] = {"filepath", "grid", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifGridType, &grid_obj)) {
    return NULL;
  }

  sifGridObject* grid_wrap = (sifGridObject*)grid_obj;

  int status = 0;
  Py_BEGIN_ALLOW_THREADS status = sif_grid_write(filepath, grid_wrap->grid);
  Py_END_ALLOW_THREADS

    if (status != 0) {
    return PyErr_Format(
      PyExc_IOError, "Failed to write .xgrid to %s", filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_grid(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  sif_grid_t* grid = NULL;

  Py_BEGIN_ALLOW_THREADS grid = sif_grid_read(filepath);
  Py_END_ALLOW_THREADS

    if (!grid) {
    return PyErr_Format(
      PyExc_IOError, "Failed to read .xgrid from %s", filepath);
  }

  sifGridObject* obj = (sifGridObject*)sifGridType.tp_alloc(&sifGridType, 0);
  if (!obj) {
    sif_grid_free(grid);
    return PyErr_NoMemory();
  }

  obj->grid = grid;
  return (PyObject*)obj;
}

/* --- ASCII I/O --- */

PyObject* pysif_read_field_ascii(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* format = "x y z";
  char delimiter = ' ';
  int skip_lines = 0;
  const char* delim_str = " ";

  static char* kwlist[] = {
    "filepath", "format", "delimiter", "skip_lines", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|ssi", kwlist, &filepath,
        &format, &delim_str, &skip_lines)) {
    return NULL;
  }

  delimiter = delim_str[0];

  /* Allocate an empty field struct. The C ASCII reader will auto-detect
   * the particle count and allocate the NUMA blocks internally. */
  sif_field_t* field = sif_field_alloc(0);
  if (!field)
    return PyErr_NoMemory();

  int status = 0;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status =
    sif_field_read_ascii_into(field, filepath, format, delimiter, skip_lines);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    sif_field_free(field);
    return py_sif_raise(status, filepath);
  }

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }

  obj->field = field;
  return (PyObject*)obj;
}

/* --- raw binary input --- */

/* One of a fixed set of strings, to its enumerator; a ValueError naming the
 * argument otherwise. */
static int binary_choice(const char* arg, const char* s,
  const char* const* names, const int* values, int n, int* out) {
  for (int i = 0; i < n; i++) {
    if (strcmp(s, names[i]) == 0) {
      *out = values[i];
      return 0;
    }
  }
  char accepted[128] = "";
  for (int i = 0; i < n; i++) {
    strncat(accepted, i ? ", '" : "'", sizeof(accepted) - strlen(accepted) - 1);
    strncat(accepted, names[i], sizeof(accepted) - strlen(accepted) - 1);
    strncat(accepted, "'", sizeof(accepted) - strlen(accepted) - 1);
  }
  PyErr_Format(
    PyExc_ValueError, "%s must be one of %s, not '%s'", arg, accepted, s);
  return -1;
}

PyObject* pysif_read_field_binary(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* format = "x y z";
  const char* layout_s = "rows";
  const char* precision_s = "float32";
  const char* byteorder_s = "native";
  unsigned long long header_bytes = 0;
  unsigned long long n_particles = 0;
  PyObject* into = Py_None;

  static char* kwlist[] = {"filepath", "format", "layout", "precision",
    "byteorder", "header_bytes", "n_particles", "field", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s$sssKKO", kwlist, &filepath,
        &format, &layout_s, &precision_s, &byteorder_s, &header_bytes,
        &n_particles, &into))
    return NULL;

  static const char* const layouts[] = {"rows", "blocks"};
  static const int layout_values[] = {SIF_BINARY_ROWS, SIF_BINARY_BLOCKS};
  static const char* const precisions[] = {"float32", "float64"};
  static const int precision_values[] = {
    SIF_BINARY_FLOAT32, SIF_BINARY_FLOAT64};
  static const char* const orders[] = {"native", "little", "big"};
  static const int order_values[] = {
    SIF_BINARY_NATIVE, SIF_BINARY_LITTLE, SIF_BINARY_BIG};

  int layout, precision, byteorder;
  if (binary_choice("layout", layout_s, layouts, layout_values, 2, &layout) <
        0 ||
      binary_choice("precision", precision_s, precisions, precision_values, 2,
        &precision) < 0 ||
      binary_choice(
        "byteorder", byteorder_s, orders, order_values, 3, &byteorder) < 0)
    return NULL;

  /* Into a field the caller already has -- a second file adding columns to a
   * first -- or into a new one. */
  sifFieldObject* target = NULL;
  sif_field_t* field = NULL;
  if (into != Py_None) {
    if (!PyObject_TypeCheck(into, &sifFieldType)) {
      PyErr_SetString(PyExc_TypeError, "field must be a pysif.Field or None");
      return NULL;
    }
    if (n_particles) {
      PyErr_SetString(PyExc_ValueError,
        "n_particles cannot be given with field: the field's own count is "
        "used");
      return NULL;
    }
    target = (sifFieldObject*)into;
    field = target->field;
  } else {
    field = sif_field_alloc(n_particles);
    if (!field)
      return PyErr_NoMemory();
  }

  sif_error_clear();
  const int status = sif_field_read_binary_into(field, filepath, format,
    (sif_binary_layout_t)layout, (sif_binary_precision_t)precision,
    (sif_binary_endian_t)byteorder, header_bytes);

  if (status != SIF_OK) {
    if (!target)
      sif_field_free(field);
    return py_sif_raise(status, filepath);
  }

  if (target) {
    Py_INCREF(into);
    return into;
  }

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }
  obj->field = field;
  return (PyObject*)obj;
}

PyObject* pysif_write_catalogue_ascii(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* cat_obj;

  static char* kwlist[] = {"filepath", "catalogue", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "sO!", kwlist, &filepath, &sifCatalogueType, &cat_obj)) {
    return NULL;
  }

  sifCatalogueObject* cat = (sifCatalogueObject*)cat_obj;

  int status = 0;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status =
    sif_catalogue_write_ascii(filepath, cat->catalogue);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    return py_sif_raise(status, filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_catalogue_ascii(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* format = NULL;
  static char* kwlist[] = {"filepath", "format", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|z", kwlist, &filepath, &format)) {
    return NULL;
  }

  sif_catalogue_t* cat = NULL;

  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS cat = sif_catalogue_read_ascii(filepath, format);
  Py_END_ALLOW_THREADS

    if (!cat) {
    return py_sif_raise(sif_error_status(), filepath);
  }

  sifCatalogueObject* obj =
    (sifCatalogueObject*)sifCatalogueType.tp_alloc(&sifCatalogueType, 0);
  if (!obj) {
    sif_catalogue_free(cat);
    return PyErr_NoMemory();
  }

  obj->catalogue = cat;
  return (PyObject*)obj;
}
PyObject* pysif_write_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* prof_obj;
  PyObject* cat_obj;

  static char* kwlist[] = {"filepath", "profiles", "catalogue", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!", kwlist, &filepath,
        &sifProfilesType, &prof_obj, &sifCatalogueType, &cat_obj)) {
    return NULL;
  }

  sifProfilesObject* prof = (sifProfilesObject*)prof_obj;
  sifCatalogueObject* cat = (sifCatalogueObject*)cat_obj;

  if (!prof->dens && !prof->vel) {
    PyErr_SetString(
      PyExc_ValueError, "these Profiles hold neither densities nor velocities");
    return NULL;
  }

  int status = 0;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status =
    sif_profiles_write_ascii(filepath, prof->dens, prof->vel, cat->catalogue);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    return py_sif_raise(status, filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_profiles_ascii(
  PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  /* Asking for a block the file does not carry is an error, so the header
     says which to ask for. */
  int has_dens = 0, has_vel = 0;
  sif_error_clear();
  const int header_status = sif_profiles_read_header_ascii(
    filepath, NULL, NULL, NULL, &has_dens, &has_vel, NULL);
  if (header_status != SIF_OK)
    return py_sif_raise(header_status, filepath);

  sif_catalogue_t* cat = NULL;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  int status = SIF_OK;

  Py_BEGIN_ALLOW_THREADS status = sif_profiles_read_ascii(
    filepath, &cat, has_dens ? &dens : NULL, has_vel ? &vel : NULL);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    return py_sif_raise(status, filepath);
  }

  sifProfilesObject* prof =
    (sifProfilesObject*)sifProfilesType.tp_alloc(&sifProfilesType, 0);
  sifCatalogueObject* cat_obj =
    (sifCatalogueObject*)sifCatalogueType.tp_alloc(&sifCatalogueType, 0);

  if (!prof || !cat_obj) {
    Py_XDECREF(prof);
    Py_XDECREF(cat_obj);
    sif_catalogue_free(cat);
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    return PyErr_NoMemory();
  }

  prof->dens = dens;
  prof->vel = vel;
  cat_obj->catalogue = cat;

  return Py_BuildValue("NN", (PyObject*)prof, (PyObject*)cat_obj);
}

/* --- paths, shared by the readers of several files --- */

/* A str or os.PathLike, or a sequence of them, as file-system bytes objects
 * in a new list. */
PyObject* py_sif_paths_list(PyObject* obj) {
  PyObject* list = PyList_New(0);
  if (!list)
    return NULL;

  const int single = PyUnicode_Check(obj) || PyBytes_Check(obj) ||
                     PyObject_HasAttrString(obj, "__fspath__");
  PyObject* seq = single ? NULL : PySequence_Fast(obj, "");
  if (!single && !seq) {
    PyErr_Clear();
    PyErr_SetString(PyExc_TypeError,
      "paths must be a path (str or os.PathLike) or a sequence of paths");
    Py_DECREF(list);
    return NULL;
  }

  const Py_ssize_t n = single ? 1 : PySequence_Fast_GET_SIZE(seq);
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* item = single ? obj : PySequence_Fast_GET_ITEM(seq, i);
    PyObject* bytes = NULL;
    if (!PyUnicode_FSConverter(item, &bytes) ||
        PyList_Append(list, bytes) < 0) {
      Py_XDECREF(bytes);
      Py_XDECREF(seq);
      Py_DECREF(list);
      return NULL;
    }
    Py_DECREF(bytes);
  }
  Py_XDECREF(seq);

  if (PyList_GET_SIZE(list) == 0) {
    PyErr_SetString(PyExc_ValueError, "paths is empty");
    Py_DECREF(list);
    return NULL;
  }
  return list;
}

/* A missing file as the operating system's error, with its errno: a
 * FileNotFoundError, as open() would raise. */
int py_sif_require_file(const char* path) {
  struct stat sb;
  if (stat(path, &sb) == 0)
    return 0;
  PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
  return -1;
}
