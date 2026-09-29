/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The background cosmology from Python. There is no Cosmology type: the five
 * parameters are keyword arguments wherever one is needed, with the same
 * defaults everywhere -- a cosmological constant, no radiation, and flat
 * unless omega_de says otherwise -- built into the C struct by one function,
 * so that comoving_distance() and Field.convert_sky_coordinates() cannot
 * disagree about what a set of arguments means.
 */

#include "model/py_model.h"

#include "sif/model/cosmology.h"

#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

int py_sif_cosmology_from(double omega_m, PyObject* omega_de, double omega_r,
  double w0, double wa, sif_cosmology_t* out) {

  double de;
  if (!omega_de || omega_de == Py_None) {
    de = 1.0 - omega_m - omega_r;
  } else {
    de = PyFloat_AsDouble(omega_de);
    if (de == -1.0 && PyErr_Occurred())
      return -1;
  }

  /* The C side refuses these too, but only says why in the log. */
  char detail[192];
  if (!isfinite(omega_m) || !isfinite(de) || !isfinite(omega_r) ||
      !isfinite(w0) || !isfinite(wa)) {
    PyErr_SetString(
      PyExc_ValueError, "cosmological parameters must be finite numbers");
    return -1;
  }
  if (omega_m < 0.0 || de < 0.0 || omega_r < 0.0) {
    snprintf(detail, sizeof(detail),
      "densities cannot be negative: omega_m = %g, omega_de = %g%s, "
      "omega_r = %g",
      omega_m, de, (!omega_de || omega_de == Py_None) ? " (flat)" : "",
      omega_r);
    PyErr_SetString(PyExc_ValueError, detail);
    return -1;
  }

  out->omega_m = omega_m;
  out->omega_de = de;
  out->omega_r = omega_r;
  out->w0 = w0;
  out->wa = wa;
  return 0;
}

PyObject* py_sif_cosmology_range_error(double z) {
  char detail[160];
  snprintf(detail, sizeof(detail),
    "this cosmology has no expansion history out to z = %g: E(z)^2 is not "
    "positive before it",
    z);
  PyErr_SetString(PyExc_ValueError, detail);
  return NULL;
}

PyObject* py_sif_comoving_distance(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject* z_obj;
  double omega_m;
  PyObject* omega_de = Py_None;
  double omega_r = 0.0, w0 = -1.0, wa = 0.0;
  static char* kwlist[] = {
    "z", "omega_m", "omega_de", "omega_r", "w0", "wa", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|Oddd", kwlist, &z_obj,
        &omega_m, &omega_de, &omega_r, &w0, &wa))
    return NULL;

  sif_cosmology_t cosmo;
  if (py_sif_cosmology_from(omega_m, omega_de, omega_r, w0, wa, &cosmo) < 0)
    return NULL;

  PyArrayObject* z = (PyArrayObject*)PyArray_FROM_OTF(
    z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!z)
    return NULL;

  PyArrayObject* d = (PyArrayObject*)PyArray_SimpleNew(
    PyArray_NDIM(z), PyArray_SHAPE(z), NPY_DOUBLE);
  if (!d) {
    Py_DECREF(z);
    return NULL;
  }

  const npy_intp n = PyArray_SIZE(z);
  const double* zv = (const double*)PyArray_DATA(z);
  double* dv = (double*)PyArray_DATA(d);

  for (npy_intp i = 0; i < n; i++) {
    if (!(zv[i] >= 0.0) || !isfinite(zv[i])) {
      char detail[128];
      snprintf(detail, sizeof(detail),
        "redshifts must be finite and at least 0, not %g", zv[i]);
      PyErr_SetString(PyExc_ValueError, detail);
      Py_DECREF(z);
      Py_DECREF(d);
      return NULL;
    }
    const int status = sif_cosmology_comoving_distance(&cosmo, zv[i], &dv[i]);
    if (status != SIF_OK) {
      Py_DECREF(z);
      Py_DECREF(d);
      return py_sif_cosmology_range_error(zv[i]);
    }
  }

  Py_DECREF(z);

  /* A number in, a number out. */
  if (PyArray_NDIM(d) == 0) {
    const double v = dv[0];
    Py_DECREF(d);
    return PyFloat_FromDouble(v);
  }
  return (PyObject*)d;
}
