#include "py_model.h"

#include "py_delta_common.h"
#include "structures/py_delta_moments.h"
#include "structures/py_size_function.h"
#include "sif/model/bbks.h"
#include <numpy/arrayobject.h>
#include <string.h>

PyObject* py_sif_g_bbks(PyObject* self, PyObject* args, PyObject* kwds) {
  double gamma, w;
  int exact = 0;
  static char* kwlist[] = {"gamma", "w", "exact", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "dd|p", kwlist, &gamma, &w, &exact))
    return NULL;

  if (gamma <= 0.0 || gamma >= 1.0) {
    PyErr_SetString(PyExc_ValueError, "gamma must lie strictly in (0, 1)");
    return NULL;
  }

  const sif_option_t options = exact ? SIF_BBKS_G_EXACT : SIF_BBKS_G_FITTED;
  return PyFloat_FromDouble(
    (double)sif_g_bbks((real_t)gamma, (real_t)w, options));
}

PyObject* py_sif_differential_number_density_bbks(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* nu_obj;
  PyObject* gamma_obj;
  PyObject* r_star_obj;
  int exact = 0;

  static char* kwlist[] = {"nu", "gamma", "r_star", "exact", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|p", kwlist, &nu_obj,
        &gamma_obj, &r_star_obj, &exact)) {
    return NULL;
  }

  const sif_option_t options = exact ? SIF_BBKS_G_EXACT : SIF_BBKS_G_FITTED;

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

  real_t* values = NULL;

  Py_BEGIN_ALLOW_THREADS values = sif_differential_number_density_bbks(
    (const real_t*)PyArray_DATA(nu_arr),
    (const real_t*)PyArray_DATA(gamma_arr),
    (const real_t*)PyArray_DATA(r_star_arr), (uint32_t)size, options);
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

PyObject* py_sif_cumulative_number_density_bbks(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* moments_obj;
  double delta;
  int exact = 0;

  static char* kwlist[] = {"moments", "delta", "exact", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|p", kwlist,
        &sifDeltaMomentsType, &moments_obj, &delta, &exact)) {
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

  const sif_option_t options = exact ? SIF_BBKS_G_EXACT : SIF_BBKS_G_FITTED;
  real_t* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_cumulative_number_density_bbks((real_t)delta, moments, options);
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
  int exact = 0;

  static char* kwlist[] = {"moments", "delta", "units", "exact", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|zp", kwlist,
        &sifDeltaMomentsType, &moments_obj, &delta, &units, &exact)) {
    return NULL;
  }

  sif_option_t options = exact ? SIF_BBKS_G_EXACT : SIF_BBKS_G_FITTED;

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
    sif_size_function_bbks((real_t)delta, moments, options);
  Py_END_ALLOW_THREADS

    if (!vsf) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to compute the BBKS size function; see the sif log");
    return NULL;
  }

  return py_sif_wrap_size_function(vsf);
}
