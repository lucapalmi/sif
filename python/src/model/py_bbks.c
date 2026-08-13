/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_model.h"

#include "py_delta_common.h"
#include "sif/model/bbks.h"
#include "structures/py_delta_moments.h"
#include "structures/py_size_function.h"
#include <numpy/arrayobject.h>
#include <string.h>

/* Which G(gamma, w) to evaluate. Shared by every entry point here. */
static int py_sif_bbks_g_option(const char* g, sif_option* opt) {
  if (!g || strcmp(g, "fitted") == 0) {
    *opt = SIF_BBKS_G_FITTED;
    return 0;
  }
  if (strcmp(g, "exact") == 0) {
    *opt = SIF_BBKS_G_EXACT;
    return 0;
  }
  PyErr_Format(PyExc_ValueError, "g must be 'fitted' or 'exact', got '%s'", g);
  return -1;
}

PyObject* py_sif_bbks_g(PyObject* self, PyObject* args, PyObject* kwds) {
  double gamma, w;
  const char* g_mode = NULL;
  static char* kwlist[] = {"gamma", "w", "g", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "dd|z", kwlist, &gamma, &w, &g_mode))
    return NULL;

  if (gamma <= 0.0 || gamma >= 1.0) {
    PyErr_SetString(PyExc_ValueError, "gamma must lie strictly in (0, 1)");
    return NULL;
  }

  sif_option options;
  if (py_sif_bbks_g_option(g_mode, &options) != 0)
    return NULL;
  return PyFloat_FromDouble(
    (double)sif_bbks_g((sif_real)gamma, (sif_real)w, options));
}

PyObject* py_sif_bbks_number_density_differential(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* nu_obj;
  PyObject* gamma_obj;
  PyObject* r_star_obj;
  const char* g_mode = NULL;

  static char* kwlist[] = {"nu", "gamma", "r_star", "g", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|z", kwlist, &nu_obj,
        &gamma_obj, &r_star_obj, &g_mode)) {
    return NULL;
  }

  sif_option options;
  if (py_sif_bbks_g_option(g_mode, &options) != 0)
    return NULL;

  PyArrayObject* nu_arr = py_sif_as_real_array(nu_obj, "nu");
  if (!nu_arr)
    return NULL;

  PyArrayObject* gamma_arr = py_sif_as_real_array(gamma_obj, "gamma");
  if (!gamma_arr) {
    Py_DECREF(nu_arr);
    return NULL;
  }

  PyArrayObject* r_star_arr = py_sif_as_real_array(r_star_obj, "r_star");
  if (!r_star_arr) {
    Py_DECREF(nu_arr);
    Py_DECREF(gamma_arr);
    return NULL;
  }

  const npy_intp size = PyArray_SIZE(nu_arr);

  if (PyArray_SIZE(gamma_arr) != size || PyArray_SIZE(r_star_arr) != size) {
    Py_DECREF(nu_arr);
    Py_DECREF(gamma_arr);
    Py_DECREF(r_star_arr);
    PyErr_SetString(
      PyExc_ValueError, "nu, gamma and r_star must have the same length");
    return NULL;
  }

  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_bbks_number_density_differential((const sif_real*)PyArray_DATA(nu_arr),
      (const sif_real*)PyArray_DATA(gamma_arr),
      (const sif_real*)PyArray_DATA(r_star_arr), (uint32_t)size, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(nu_arr);
  Py_DECREF(gamma_arr);
  Py_DECREF(r_star_arr);

  if (!values) {
    PyErr_SetString(PyExc_ValueError,
      "Failed to evaluate the BBKS number density; see the sif log for the "
      "cause.");
    return NULL;
  }

  return py_sif_owned_array(values, size);
}

PyObject* py_sif_bbks_number_density_cumulative(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* moments_obj;
  double delta;
  const char* g_mode = NULL;

  static char* kwlist[] = {"moments", "delta", "g", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|z", kwlist,
        &sifDeltaMomentsType, &moments_obj, &delta, &g_mode)) {
    return NULL;
  }

  const sif_delta_moments_t* moments =
    ((sifDeltaMomentsObject*)moments_obj)->moments;

  if (!moments) {
    PyErr_SetString(PyExc_ValueError, "these moments are empty");
    return NULL;
  }

  if (moments->order < 2) {
    PyErr_Format(PyExc_ValueError,
      "the cumulative density needs sigma_0 through sigma_2, but these "
      "moments only reach order %u",
      moments->order);
    return NULL;
  }

  sif_option options;
  if (py_sif_bbks_g_option(g_mode, &options) != 0)
    return NULL;
  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_bbks_number_density_cumulative((sif_real)delta, moments, options);
  Py_END_ALLOW_THREADS

    if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to compute the cumulative number density; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, moments->n_radii);
}

PyObject* py_sif_size_function_bbks(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* moments_obj;
  double delta;
  const char* units = NULL;
  const char* g_mode = NULL;

  static char* kwlist[] = {"moments", "delta", "units", "g", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|zz", kwlist,
        &sifDeltaMomentsType, &moments_obj, &delta, &units, &g_mode)) {
    return NULL;
  }

  sif_option options;
  if (py_sif_bbks_g_option(g_mode, &options) != 0)
    return NULL;

  if (units == NULL || strcmp(units, "ln_r") == 0) {
    options |= SIF_VSF_BIN_LN;
  } else if (strcmp(units, "r") == 0) {
    options |= SIF_VSF_BIN_LINEAR;
  } else {
    PyErr_Format(PyExc_ValueError,
      "units must be None, 'ln_r' (dC/dlnR) or 'r' (dC/dR), not '%s'", units);
    return NULL;
  }

  const sif_delta_moments_t* moments =
    ((sifDeltaMomentsObject*)moments_obj)->moments;

  if (!moments) {
    PyErr_SetString(PyExc_ValueError, "these moments are empty");
    return NULL;
  }

  if (moments->order < 2) {
    PyErr_Format(PyExc_ValueError,
      "the size function needs sigma_0 through sigma_2, but these moments "
      "only reach order %u",
      moments->order);
    return NULL;
  }

  if (moments->n_radii < 2) {
    PyErr_SetString(PyExc_ValueError,
      "the size function is a derivative in R and needs at least two radii");
    return NULL;
  }

  sif_size_function_t* vsf = NULL;

  Py_BEGIN_ALLOW_THREADS vsf =
    sif_size_function_bbks((sif_real)delta, moments, options);
  Py_END_ALLOW_THREADS

    if (!vsf) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to compute the BBKS size function; see the sif log");
    return NULL;
  }

  return py_sif_wrap_size_function(vsf);
}
