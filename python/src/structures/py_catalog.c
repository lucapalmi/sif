/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_catalog.h"
#include <numpy/arrayobject.h>

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

static PyGetSetDef sifCatalog_getset[] = {
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

/* --- Methods --- */

static PyObject* sifCatalog_translate(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  double ox, oy, oz;
  static char* kwlist[] = {"offset", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ddd)", kwlist, &ox, &oy, &oz))
    return NULL;

  const sif_real offset[3] = {(sif_real)ox, (sif_real)oy, (sif_real)oz};
  sif_catalog_translate(((sifCatalogObject*)self_obj)->catalog, offset);
  Py_RETURN_NONE;
}

static PyMethodDef sifCatalog_methods[] = {
  {"translate", (PyCFunction)sifCatalog_translate, METH_VARARGS | METH_KEYWORDS,
    "translate(offset)\n"
    "--\n\n"
    "Shift every void centre by offset, in place.\n\n"
    "The way back out of the box a survey was searched in: pass the negated\n"
    "offset survey_box() returned, and the centres return to your own frame.\n"
    "Radii and footprints are unchanged.\n\n"
    "Args:\n"
    "    offset: Three numbers, added to x, y and z."},
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