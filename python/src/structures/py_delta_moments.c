/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_delta_moments.h"

#include "py_delta_common.h"

/* gamma and R_star are BBKS quantities derived from a moment set; they live
 * in the model module, not with the container. */
#include "sif/model/bbks.h"

#include <numpy/arrayobject.h>

static void sifDeltaMoments_dealloc(PyObject* self_obj) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (self->moments != NULL) {
    sif_delta_moments_free(self->moments);
    self->moments = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifDeltaMoments_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;

  static char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;

  self->moments = NULL;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifDeltaMoments_get_n_radii(
  PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->moments->n_radii);
}

static PyObject* sifDeltaMoments_get_order(PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->moments->order);
}

static PyObject* sifDeltaMoments_get_n_moments(
  PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->moments->n_moments);
}

static PyObject* sifDeltaMoments_get_radii(PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments || !self->moments->radii)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->moments->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, self->moments->radii);
}

/* Exposed 2D as (n_moments, n_radii), which is exactly the order-major layout
 * the C side stores, so sigma[j] is order j across radii. */
static PyObject* sifDeltaMoments_get_sigma(PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments || !self->moments->sigma)
    Py_RETURN_NONE;

  npy_intp dims[2] = {self->moments->n_moments, self->moments->n_radii};
  return py_sif_wrap_borrowed(self_obj, 2, dims, self->moments->sigma);
}

static PyObject* sifDeltaMoments_get_high_k_fraction(
  PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments || !self->moments->high_k_fraction)
    Py_RETURN_NONE;

  npy_intp dims[2] = {self->moments->n_moments, self->moments->n_radii};
  return py_sif_wrap_borrowed(
    self_obj, 2, dims, self->moments->high_k_fraction);
}

static PyObject* sifDeltaMoments_get_offsets(
  PyObject* self_obj, void* closure) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments || !self->moments->offsets)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->moments->n_moments + 1};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_UINT32, self->moments->offsets);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  if (PyArray_SetBaseObject((PyArrayObject*)array, self_obj) < 0) {
    Py_DECREF(array);
    return NULL;
  }
  PyArray_CLEARFLAGS((PyArrayObject*)array, NPY_ARRAY_WRITEABLE);
  return array;
}

/*
 * gamma and R_star are external functions on the C side, since they are
 * combinations of three moments rather than moments. They surface here as
 * derived properties, each returning a fresh array.
 */
static PyObject* derived_array(PyObject* self_obj,
  sif_real* (*compute)(const sif_delta_moments_t*), const char* what) {

  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  if (!self->moments)
    Py_RETURN_NONE;

  if (self->moments->order < 2) {
    PyErr_Format(PyExc_ValueError,
      "%s needs sigma_0 through sigma_2, but these moments only reach order %u",
      what, self->moments->order);
    return NULL;
  }

  sif_real* values = compute(self->moments);
  if (!values) {
    PyErr_Format(
      PyExc_RuntimeError, "failed to compute %s; see the sif log", what);
    return NULL;
  }

  return py_sif_owned_array(values, self->moments->n_radii);
}

static PyObject* sifDeltaMoments_get_gamma(PyObject* self_obj, void* closure) {
  return derived_array(self_obj, sif_bbks_gamma, "gamma");
}

static PyObject* sifDeltaMoments_get_r_star(PyObject* self_obj, void* closure) {
  return derived_array(self_obj, sif_bbks_r_star, "r_star");
}

/*
 * The per-order accessor, mirroring sif_delta_moments_sigma. Equivalent to
 * sigma[order], but names the available range instead of raising IndexError.
 */
static PyObject* sifDeltaMoments_moment(PyObject* self_obj, PyObject* args) {
  sifDeltaMomentsObject* self = (sifDeltaMomentsObject*)self_obj;
  int order;

  if (!PyArg_ParseTuple(args, "i", &order))
    return NULL;

  if (!self->moments || !self->moments->sigma)
    Py_RETURN_NONE;

  if (order < 0 || order > (int)self->moments->order) {
    PyErr_Format(PyExc_ValueError,
      "order %d is out of range; these moments reach order %u", order,
      self->moments->order);
    return NULL;
  }

  const sif_real* values =
    sif_delta_moments_sigma(self->moments, (uint8_t)order);
  if (!values) {
    PyErr_Format(PyExc_RuntimeError, "order %d was not computed", order);
    return NULL;
  }

  npy_intp dims[1] = {self->moments->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, (void*)values);
}

static PyMethodDef sifDeltaMoments_methods[] = {
  {"moment", sifDeltaMoments_moment, METH_VARARGS,
    "sigma_order across radii, as a read-only view into the moment set.\n"
    "Equivalent to sigma[order], but rejects an order that was not computed "
    "rather than indexing past the end."},
  {NULL}};

static PyGetSetDef sifDeltaMoments_getset[] = {
  {"n_radii", sifDeltaMoments_get_n_radii, NULL, "Number of smoothing radii",
    NULL},
  {"order", sifDeltaMoments_get_order, NULL, "Highest moment order computed",
    NULL},
  {"n_moments", sifDeltaMoments_get_n_moments, NULL, "order + 1", NULL},
  {"radii", sifDeltaMoments_get_radii, NULL, "1D array of smoothing radii",
    NULL},
  {"sigma", sifDeltaMoments_get_sigma, NULL,
    "(n_moments, n_radii) array; sigma[j] is sigma_j across radii", NULL},
  {"high_k_fraction", sifDeltaMoments_get_high_k_fraction, NULL,
    "(n_moments, n_radii) fraction of each sum from the top half of the "
    "available k range; a large value means that moment is resolution- or "
    "truncation-limited",
    NULL},
  {"offsets", sifDeltaMoments_get_offsets, NULL,
    "n_moments + 1 offsets into the flat C arrays, one block per order", NULL},
  {"gamma", sifDeltaMoments_get_gamma, NULL,
    "BBKS spectral parameter sigma_1^2 / (sigma_0 sigma_2); needs order >= 2",
    NULL},
  {"r_star", sifDeltaMoments_get_r_star, NULL,
    "BBKS coherence scale sqrt(3) sigma_1 / sigma_2; needs order >= 2", NULL},
  {NULL}};

PyTypeObject sifDeltaMomentsType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.measure.DeltaMoments",
  .tp_basicsize = sizeof(sifDeltaMomentsObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifDeltaMoments_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "DeltaMoments()\n"
    "--\n\n"
    "Spectral moments sigma_0..sigma_order at several smoothing\n"
    "radii.\n\n"
    "Returned by pysif.measure.delta_moments_grid() (measured from a\n"
    "field) or pysif.model.delta_moments_pk() (integrated from a model\n"
    "spectrum). The orders are held together because they are only\n"
    "meaningful together: sigma_0 and sigma_2 describe the same field\n"
    "only if they came from the same radii and the same window.\n\n"
    "Ask the producer for order 0 if only sigma_0 is wanted.\n\n"
    "Not constructed directly.",
  .tp_methods = sifDeltaMoments_methods,
  .tp_getset = sifDeltaMoments_getset,
  .tp_init = sifDeltaMoments_init,
  .tp_new = PyType_GenericNew,
};

PyObject* py_sif_wrap_moments(sif_delta_moments_t* moments) {
  sifDeltaMomentsObject* obj =
    (sifDeltaMomentsObject*)sifDeltaMomentsType.tp_alloc(
      &sifDeltaMomentsType, 0);
  if (!obj) {
    sif_delta_moments_free(moments);
    return PyErr_NoMemory();
  }
  obj->moments = moments;
  return (PyObject*)obj;
}
