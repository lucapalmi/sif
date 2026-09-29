/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_catalog.h"

#include "model/py_model.h"

#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <string.h>

/* --- Lifecycle Methods --- */

static void sifCatalog_dealloc(PyObject* self_obj) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  if (self->catalog != NULL) {
    sif_catalog_free(self->catalog);
    self->catalog = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifCatalog_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint64_t initial_capacity = 1024; // Sensible default

  static char* kwlist[] = {"capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|K", kwlist, &initial_capacity)) {
    return -1;
  }

  sif_catalog_t* tmp = sif_catalog_alloc(initial_capacity);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sif.catalog");
    return -1;
  }

  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  self->catalog = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifCatalog_get_centers(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;

  npy_intp dims[2] = {self->catalog->n_voids, 3};
  PyObject* array = PyArray_SimpleNew(2, dims, NPY_REAL_T);
  if (!array)
    return NULL;

  sif_real* data = (sif_real*)PyArray_DATA((PyArrayObject*)array);

  /* Interleave the C Struct-of-Arrays into a Python Nx3 array */
  for (uint64_t i = 0; i < self->catalog->n_voids; i++) {
    data[i * 3 + 0] = self->catalog->cx[i];
    data[i * 3 + 1] = self->catalog->cy[i];
    data[i * 3 + 2] = self->catalog->cz[i];
  }

  return array;
}

static PyObject* sifCatalog_get_radii(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;

  npy_intp dims[1] = {self->catalog->n_voids};

  /* Create an array referencing the underlying C memory (Zero-Copy) */
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->catalog->radii);
  if (!array)
    return NULL;

  /* Tie the lifecycle of the array to the catalog object */
  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);

  return array;
}

/*
 * A footprint column, zero-copy like radii, or None for a catalogue that has
 * none. Safe as a view: nothing reachable from Python grows a catalogue, so
 * the buffer cannot move under it.
 */
static PyObject* catalog_optional_column(PyObject* self_obj, sif_real* col) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  if (!col)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->catalog->n_voids};
  PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, col);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifCatalog_get_footprint(PyObject* self_obj, void* closure) {
  return catalog_optional_column(
    self_obj, ((sifCatalogObject*)self_obj)->catalog->footprint);
}

static PyObject* sifCatalog_get_footprint_shell(
  PyObject* self_obj, void* closure) {
  return catalog_optional_column(
    self_obj, ((sifCatalogObject*)self_obj)->catalog->footprint_shell);
}

static PyObject* sifCatalog_get_n_voids(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->catalog->n_voids);
}

static PyObject* sifCatalog_get_units(PyObject* self_obj, void* closure) {
  (void)closure;
  return PyUnicode_FromString(
    ((sifCatalogObject*)self_obj)->catalog->units == SIF_COORDINATES_SKY
      ? "sky"
      : "cartesian");
}

/* The catalogue's metadata, as a new dict in the order keys were set. */
static PyObject* sifCatalog_get_metadata(PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_catalog_t* cat = ((sifCatalogObject*)self_obj)->catalog;
  PyObject* dict = PyDict_New();
  for (uint32_t m = 0; dict && m < sif_catalog_meta_count(cat); m++) {
    const char* key = sif_catalog_meta_name(cat, m);
    PyObject* value = NULL;
    switch (sif_catalog_meta_kind(cat, key)) {
    case SIF_CATALOG_META_INT:
      value =
        PyLong_FromLongLong((long long)sif_catalog_meta_int_get(cat, key));
      break;
    case SIF_CATALOG_META_REAL:
      value = PyFloat_FromDouble(sif_catalog_meta_real_get(cat, key));
      break;
    case SIF_CATALOG_META_STRING:
      value = PyUnicode_FromString(sif_catalog_meta_string_get(cat, key));
      break;
    default:
      value = Py_NewRef(Py_None);
    }
    if (!value || PyDict_SetItemString(dict, key, value) < 0) {
      Py_XDECREF(value);
      Py_CLEAR(dict);
      break;
    }
    Py_DECREF(value);
  }
  return dict;
}

static PyGetSetDef sifCatalog_getset[] = {
  {"metadata", sifCatalog_get_metadata, NULL,
    "dict: the named values describing the catalogue -- the finder and its\n"
    "settings, the cosmology, anything set with set_metadata() -- a copy.\n"
    "Every writer records them and every reader gives them back: '#key=value'\n"
    "lines in text, attributes of /catalog in HDF5, keywords of VOIDS in\n"
    "FITS. Keys are in lower case.",
    NULL},
  {"units", sifCatalog_get_units, NULL,
    "What the centres are: 'cartesian', as a finder gives them, or 'sky' --\n"
    "right ascension, declination (degrees) and redshift -- after to_sky().",
    NULL},
  {"centers", sifCatalog_get_centers, NULL,
    "Nx3 NumPy array of void centers (x, y, z)", NULL},
  {"radii", sifCatalog_get_radii, NULL, "1D NumPy array of void radii", NULL},
  {"n_voids", sifCatalog_get_n_voids, NULL,
    "The number of voids in the catalog", NULL},
  {"footprint", sifCatalog_get_footprint, NULL,
    "1D NumPy array: the fraction of each void's sphere inside the survey\n"
    "footprint, or None for a catalogue with no footprint (a periodic box).\n"
    "-1 marks a void nobody measured it for.",
    NULL},
  {"footprint_shell", sifCatalog_get_footprint_shell, NULL,
    "1D NumPy array: the same fraction over the shell between one and two\n"
    "radii, or None with footprint.",
    NULL},
  {NULL}};

/* --- From and to NumPy --- */

/* One column: a contiguous 1D sif_real array of length n (n < 0 takes it from
 * this one), or NULL for None. 0, or -1 with an exception. */
static int column(
  PyObject* obj, const char* name, npy_intp n, PyArrayObject** out) {
  *out = NULL;
  if (obj == Py_None)
    return 0;
  PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(
    obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!arr) {
    PyErr_Format(PyExc_TypeError, "%s must be convertible to a %s array", name,
      sizeof(sif_real) == 8 ? "float64" : "float32");
    return -1;
  }
  if (PyArray_NDIM(arr) != 1 || (n >= 0 && PyArray_SHAPE(arr)[0] != n)) {
    PyErr_Format(
      PyExc_ValueError, "%s must be a 1D array with one entry per void", name);
    Py_DECREF(arr);
    return -1;
  }
  *out = arr;
  return 0;
}

PyObject* pysif_catalog_from_numpy(
  PyObject* module, PyObject* args, PyObject* kwds) {
  (void)module;
  PyObject *cx = Py_None, *cy = Py_None, *cz = Py_None, *r = Py_None;
  PyObject *ra = Py_None, *dec = Py_None, *z = Py_None;
  PyObject *fp = Py_None, *fps = Py_None;
  static char* kwlist[] = {"cx", "cy", "cz", "r", "ra", "dec", "z", "footprint",
    "footprint_shell", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO$OOOOO", kwlist, &cx, &cy,
        &cz, &r, &ra, &dec, &z, &fp, &fps))
    return NULL;

  /* The names say what the centres are, as everywhere else. */
  const bool cart = cx != Py_None || cy != Py_None || cz != Py_None;
  const bool sky = ra != Py_None || dec != Py_None || z != Py_None;
  if (cart && sky) {
    PyErr_SetString(PyExc_ValueError,
      "give cx, cy and cz for Cartesian centres, or ra, dec and z for sky "
      "ones, not some of each");
    return NULL;
  }
  PyObject* c[3] = {sky ? ra : cx, sky ? dec : cy, sky ? z : cz};
  if (c[0] == Py_None || c[1] == Py_None || c[2] == Py_None || r == Py_None) {
    PyErr_SetString(PyExc_ValueError,
      sky ? "ra, dec, z and r are all required"
          : "cx, cy, cz and r are required (or ra, dec, z and r)");
    return NULL;
  }
  if ((fp == Py_None) != (fps == Py_None)) {
    PyErr_SetString(PyExc_ValueError,
      "footprint and footprint_shell are given together or not at all");
    return NULL;
  }

  enum { X, Y, Z, R, FP, FPS, N_COLS };
  static const char* const cart_names[] = {"cx", "cy", "cz"};
  static const char* const sky_names[] = {"ra", "dec", "z"};
  PyArrayObject* cols[N_COLS] = {NULL};
  sif_catalog_t* cat = NULL;
  PyObject* result = NULL;

  if (column(c[0], sky ? sky_names[0] : cart_names[0], -1, &cols[X]) < 0)
    goto done;
  const npy_intp n = PyArray_SHAPE(cols[X])[0];
  if (column(c[1], sky ? sky_names[1] : cart_names[1], n, &cols[Y]) < 0 ||
      column(c[2], sky ? sky_names[2] : cart_names[2], n, &cols[Z]) < 0 ||
      column(r, "r", n, &cols[R]) < 0 ||
      column(fp, "footprint", n, &cols[FP]) < 0 ||
      column(fps, "footprint_shell", n, &cols[FPS]) < 0)
    goto done;

  cat = sif_catalog_alloc((uint64_t)n);
  if (!cat || (cols[FP] && sif_catalog_reserve_footprint(cat) != SIF_OK)) {
    PyErr_NoMemory();
    goto done;
  }
  const size_t bytes = (size_t)n * sizeof(sif_real);
  if (n > 0) {
    memcpy(cat->cx, PyArray_DATA(cols[X]), bytes);
    memcpy(cat->cy, PyArray_DATA(cols[Y]), bytes);
    memcpy(cat->cz, PyArray_DATA(cols[Z]), bytes);
    memcpy(cat->radii, PyArray_DATA(cols[R]), bytes);
    if (cols[FP]) {
      memcpy(cat->footprint, PyArray_DATA(cols[FP]), bytes);
      memcpy(cat->footprint_shell, PyArray_DATA(cols[FPS]), bytes);
    }
  }
  cat->n_voids = (uint64_t)n;
  if (sky)
    cat->units = SIF_COORDINATES_SKY;

  sifCatalogObject* obj =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!obj) {
    PyErr_NoMemory();
    goto done;
  }
  obj->catalog = cat;
  cat = NULL;
  result = (PyObject*)obj;

done:
  for (int k = 0; k < N_COLS; k++)
    Py_XDECREF(cols[k]);
  sif_catalog_free(cat);
  return result;
}

static PyObject* sifCatalog_to_numpy(PyObject* self_obj, PyObject* unused) {
  (void)unused;
  const sif_catalog_t* cat = ((sifCatalogObject*)self_obj)->catalog;
  const bool sky = cat->units == SIF_COORDINATES_SKY;
  const char* names[6] = {sky ? "ra" : "cx", sky ? "dec" : "cy",
    sky ? "z" : "cz", "r", "footprint", "footprint_shell"};
  const sif_real* cols[6] = {cat->cx, cat->cy, cat->cz, cat->radii,
    cat->footprint, cat->footprint_shell};
  const int n_fields = cat->footprint ? 6 : 4;

  /* The same names the file headers use, one field each, packed. */
  PyObject* spec = PyList_New(n_fields);
  if (!spec)
    return NULL;
  for (int k = 0; k < n_fields; k++) {
    PyObject* item =
      Py_BuildValue("(ss)", names[k], sizeof(sif_real) == 8 ? "f8" : "f4");
    if (!item) {
      Py_DECREF(spec);
      return NULL;
    }
    PyList_SET_ITEM(spec, k, item);
  }
  PyArray_Descr* descr = NULL;
  const int ok = PyArray_DescrConverter(spec, &descr);
  Py_DECREF(spec);
  if (!ok)
    return NULL;

  npy_intp dims[1] = {(npy_intp)cat->n_voids};
  PyObject* arr = PyArray_Zeros(1, dims, descr, 0); /* steals descr */
  if (!arr)
    return NULL;
  char* data = PyArray_DATA((PyArrayObject*)arr);
  const size_t stride = (size_t)n_fields * sizeof(sif_real);
  for (uint64_t i = 0; i < cat->n_voids; i++)
    for (int k = 0; k < n_fields; k++)
      memcpy(data + i * stride + (size_t)k * sizeof(sif_real), &cols[k][i],
        sizeof(sif_real));
  return arr;
}

/* --- Methods --- */

static PyObject* sifCatalog_translate(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  double ox, oy, oz;
  static char* kwlist[] = {"offset", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ddd)", kwlist, &ox, &oy, &oz))
    return NULL;

  const sif_real offset[3] = {(sif_real)ox, (sif_real)oy, (sif_real)oz};
  if (sif_catalog_translate(((sifCatalogObject*)self_obj)->catalog, offset) !=
      SIF_OK) {
    PyErr_SetString(PyExc_ValueError,
      "cannot translate: the catalogue holds sky coordinates");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* sifCatalog_to_sky(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  double omega_m;
  PyObject* omega_de = Py_None;
  double omega_r = 0.0, w0 = -1.0, wa = 0.0;
  static char* kwlist[] = {"omega_m", "omega_de", "omega_r", "w0", "wa", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "d|Oddd", kwlist, &omega_m, &omega_de, &omega_r, &w0, &wa))
    return NULL;

  sif_cosmology_t cosmo;
  if (py_sif_cosmology_from(omega_m, omega_de, omega_r, w0, wa, &cosmo) < 0)
    return NULL;

  sif_catalog_t* cat = ((sifCatalogObject*)self_obj)->catalog;
  if (cat->units != SIF_COORDINATES_CARTESIAN) {
    PyErr_SetString(PyExc_ValueError, "the catalogue is already on the sky");
    return NULL;
  }

  int status;
  Py_BEGIN_ALLOW_THREADS status = sif_catalog_to_sky(cat, &cosmo);
  Py_END_ALLOW_THREADS

    switch (status) {
  case SIF_OK:
    Py_RETURN_NONE;
  case SIF_ERR_ALLOC:
    return PyErr_NoMemory();
  case SIF_ERR_RANGE:
    PyErr_SetString(PyExc_ValueError,
      "a void centre is farther from the origin than this cosmology reaches "
      "by z = 10^4: is the observer at the origin, and are the centres in "
      "Mpc/h?");
    return NULL;
  default:
    PyErr_SetString(PyExc_ValueError,
      "some centres cannot be converted (not finite); the log has the count, "
      "and the catalogue is unchanged");
    return NULL;
  }
}

static PyObject* sifCatalog_set_metadata(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  const char* key;
  PyObject* value;
  static char* kwlist[] = {"key", "value", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", kwlist, &key, &value))
    return NULL;
  sif_catalog_t* cat = ((sifCatalogObject*)self_obj)->catalog;

  int status;
  if (value == Py_None) {
    status = sif_catalog_meta_remove(cat, key);
  } else if (PyLong_Check(value)) { /* bool included, as 1 or 0 */
    const long long v = PyLong_AsLongLong(value);
    if (v == -1 && PyErr_Occurred())
      return NULL;
    status = sif_catalog_meta_int_set(cat, key, (int64_t)v);
  } else if (PyFloat_Check(value)) {
    status = sif_catalog_meta_real_set(cat, key, PyFloat_AS_DOUBLE(value));
  } else if (PyUnicode_Check(value)) {
    const char* v = PyUnicode_AsUTF8(value);
    if (!v)
      return NULL;
    status = sif_catalog_meta_string_set(cat, key, v);
  } else {
    PyErr_SetString(
      PyExc_TypeError, "value must be an int, a float, a str or None");
    return NULL;
  }
  if (status == SIF_ERR_ALLOC)
    return PyErr_NoMemory();
  if (status != SIF_OK)
    return PyErr_Format(PyExc_ValueError,
      "cannot set %s: a key is an identifier of at most 64 characters and not "
      "one the file formats use (n, n_voids, coordinates, FITS structure); a "
      "string holds no newline or double quote; a float is finite",
      key);
  Py_RETURN_NONE;
}

static PyMethodDef sifCatalog_methods[] = {
  {"set_metadata", (PyCFunction)sifCatalog_set_metadata,
    METH_VARARGS | METH_KEYWORDS,
    "set_metadata(key, value)\n"
    "--\n\n"
    "Set one of the catalogue's named values, which every writer records:\n"
    "an int, a float or a str; None removes the key.\n\n"
    "    cat.set_metadata(\"omega_m\", 0.31)\n"
    "    cat.set_metadata(\"survey\", \"Euclid DR1\")\n\n"
    "Keys are matched without regard to case, as FITS keywords are.\n\n"
    "Raises:\n"
    "    ValueError: For a key or a value no format can keep."},
  {"to_numpy", (PyCFunction)sifCatalog_to_numpy, METH_NOARGS,
    "to_numpy()\n"
    "--\n\n"
    "The catalogue as one NumPy structured array, a copy: a record per\n"
    "void, with fields named as the file headers name them -- cx, cy, cz, r\n"
    "(or ra, dec, z, r on the sky), and footprint, footprint_shell when the\n"
    "catalogue has them. pandas.DataFrame(cat.to_numpy()) and\n"
    "astropy.table.Table(cat.to_numpy()) take it as it is."},
  {"translate", (PyCFunction)sifCatalog_translate, METH_VARARGS | METH_KEYWORDS,
    "translate(offset)\n"
    "--\n\n"
    "Shift every void centre by offset, in place.\n\n"
    "The way back out of the box a survey was searched in: pass the negated\n"
    "offset survey_box() returned, and the centres return to your own frame.\n"
    "Radii and footprints are unchanged.\n\n"
    "Args:\n"
    "    offset: Three numbers, added to x, y and z.\n\n"
    "Raises:\n"
    "    ValueError: For a catalogue on the sky."},
  {"to_sky", (PyCFunction)sifCatalog_to_sky, METH_VARARGS | METH_KEYWORDS,
    "to_sky(omega_m, omega_de=None, omega_r=0.0, w0=-1.0, wa=0.0)\n"
    "--\n\n"
    "Turn the void centres into sky coordinates, in place.\n\n"
    "The inverse of Field.convert_sky_coordinates(), for the voids a survey\n"
    "gave: each centre becomes the right ascension and declination of its\n"
    "direction from the origin, in degrees (right ascension in [0, 360)),\n"
    "and the redshift at its comoving distance from the origin. units is\n"
    "then 'sky', and every writer writes the catalogue that way.\n\n"
    "The observer has to be at the origin: move a catalogue found in a\n"
    "survey box back first, with translate(-offset). Converted with the\n"
    "cosmology the tracers were, a centre comes back to the sky within\n"
    "single precision. Radii stay comoving lengths, in Mpc/h.\n\n"
    "Args:\n"
    "    omega_m, omega_de, omega_r, w0, wa: The cosmology, as for\n"
    "        Field.convert_sky_coordinates().\n\n"
    "Raises:\n"
    "    ValueError: For a catalogue already on the sky, a centre that is\n"
    "        not finite, or one past what the cosmology reaches."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifCatalogType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.Catalog", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifCatalogObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifCatalog_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Catalog(capacity=0)\n"
            "--\n\n"
            "A list of voids: centre and radius, one entry each.\n\n"
            "What a finder produces and what pysif.measure consumes. Read one\n"
            "back with pysif.io.read_catalog_ascii(). A catalogue from\n"
            "finders.exodus_survey() also carries footprint and\n"
            "footprint_shell.\n\n"
            "Args:\n"
            "    capacity: Voids to make room for up front; it grows as\n"
            "        needed.",
  .tp_methods = sifCatalog_methods,
  .tp_getset = sifCatalog_getset,
  .tp_init = sifCatalog_init,
  .tp_new = PyType_GenericNew,
};