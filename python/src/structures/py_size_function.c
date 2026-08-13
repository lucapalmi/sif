/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_size_function.h"
#include "structures/py_catalog.h"
#include <numpy/arrayobject.h>

static void sifSizeFunction_dealloc(PyObject* self_obj) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (self->vsf != NULL) {
    sif_size_function_free(self->vsf);
    self->vsf = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifSizeFunction_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  self->vsf = NULL;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifSizeFunction_get_n_bins(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->vsf->n_bins);
}

static PyObject* sifSizeFunction_get_options(
  PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong((unsigned long)self->vsf->options);
}

static PyObject* sifSizeFunction_get_r_min(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf)
    Py_RETURN_NONE;
  return PyFloat_FromDouble((double)self->vsf->r_min);
}

static PyObject* sifSizeFunction_get_r_max(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf)
    Py_RETURN_NONE;
  return PyFloat_FromDouble((double)self->vsf->r_max);
}

static PyObject* sifSizeFunction_get_r_edges(
  PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf || !self->vsf->r_edges)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->vsf->n_bins + 1};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->vsf->r_edges);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifSizeFunction_get_r_centers(
  PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf || !self->vsf->r_centers)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->vsf->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->vsf->r_centers);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifSizeFunction_get_counts(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf || !self->vsf->counts)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->vsf->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_UINT64, self->vsf->counts);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifSizeFunction_get_vsf(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf || !self->vsf->vsf)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->vsf->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->vsf->vsf);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

/* ADDED: Getter function mapping the raw error values to a continuous Python
 * memory view */
static PyObject* sifSizeFunction_get_err(PyObject* self_obj, void* closure) {
  sifSizeFunctionObject* self = (sifSizeFunctionObject*)self_obj;
  if (!self->vsf || !self->vsf->err)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->vsf->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->vsf->err);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyGetSetDef sifSizeFunction_getset[] = {
  {"n_bins", sifSizeFunction_get_n_bins, NULL, "Number of radial bins", NULL},
  {"options", sifSizeFunction_get_options, NULL, "Bitmask options used for VSF",
    NULL},
  {"r_min", sifSizeFunction_get_r_min, NULL, "Min radius", NULL},
  {"r_max", sifSizeFunction_get_r_max, NULL, "Max radius", NULL},
  {"r_edges", sifSizeFunction_get_r_edges, NULL, "1D array of bin edges", NULL},
  {"r_centers", sifSizeFunction_get_r_centers, NULL, "1D array of bin centers",
    NULL},
  {"counts", sifSizeFunction_get_counts, NULL, "1D array of counts", NULL},
  {"vsf", sifSizeFunction_get_vsf, NULL, "1D array of VSF values", NULL},
  {"err", sifSizeFunction_get_err, NULL,
    "1D array of Poisson statistical errors", NULL}, /* ADDED */
  {NULL}};

PyTypeObject sifSizeFunctionType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.SizeFunction",
  .tp_basicsize = sizeof(sifSizeFunctionObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifSizeFunction_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SizeFunction()\n"
            "--\n\n"
            "A void size function: number density of voids per radius bin.\n\n"
            "Returned by pysif.measure.size_function_catalog() and by the\n"
            "models in pysif.model. One container serves both, so a\n"
            "measurement and a prediction can be compared directly -- but\n"
            "counts and err are meaningful only for a measurement, and are\n"
            "zero for a model.\n\n"
            "Not constructed directly.",
  .tp_getset = sifSizeFunction_getset,
  .tp_init = sifSizeFunction_init,
  .tp_new = PyType_GenericNew,
};

PyObject* py_sif_wrap_size_function(sif_size_function_t* vsf) {
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
