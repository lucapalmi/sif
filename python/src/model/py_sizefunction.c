#include "py_model.h"

#include "py_delta_common.h"
#include "sif/model/deltamoments.h"
#include "sif/model/sizefunction.h"
#include "structures/py_size_function.h"
#include <numpy/arrayobject.h>
#include <string.h>

/* Both size functions take the same arguments; only the model differs. */
typedef sif_size_function_t* (*__vsf_fn)(const real_t*, const real_t*, uint32_t,
  const real_t*, uint32_t, real_t, real_t, real_t, sif_option_t);

static PyObject* __size_function(
  PyObject* args, PyObject* kwds, __vsf_fn compute, const char* name) {
  PyObject* k_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  double delta_v = -2.7;
  double delta_c = 1.686;
  double expansion_factor = 0.0;
  const char* units = NULL;

  static char* kwlist[] = {"k", "pk", "radii", "delta_v", "delta_c",
    "expansion_factor", "units", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|dddz", kwlist, &k_obj,
        &pk_obj, &radii_obj, &delta_v, &delta_c, &expansion_factor, &units)) {
    return NULL;
  }

  sif_option_t options = SIF_DEFAULT;
  if (units == NULL || strcmp(units, "ln_r") == 0) {
    options |= SIF_VSF_BIN_LN;
  } else if (strcmp(units, "r") == 0) {
    options |= SIF_VSF_BIN_LINEAR;
  } else {
    PyErr_Format(PyExc_ValueError,
      "units must be None, 'ln_r' (dn/dlnR) or 'r' (dn/dR), not '%s'", units);
    return NULL;
  }

  if (!(delta_v < 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_v must be strictly negative");
    return NULL;
  }
  if (!(delta_c > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_c must be strictly positive");
    return NULL;
  }

  PyArrayObject* k_arr = py_sif_as_real_array(k_obj, "k");
  if (!k_arr)
    return NULL;

  PyArrayObject* pk_arr = py_sif_as_real_array(pk_obj, "pk");
  if (!pk_arr) {
    Py_DECREF(k_arr);
    return NULL;
  }

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr) {
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    return NULL;
  }

  if (PyArray_SIZE(k_arr) != PyArray_SIZE(pk_arr)) {
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "k and pk must have the same length");
    return NULL;
  }

  const npy_intp n_points = PyArray_SIZE(k_arr);
  const npy_intp n_radii = PyArray_SIZE(radii_arr);

  sif_size_function_t* vsf = NULL;

  Py_BEGIN_ALLOW_THREADS vsf = compute((const real_t*)PyArray_DATA(k_arr),
    (const real_t*)PyArray_DATA(pk_arr), (uint32_t)n_points,
    (const real_t*)PyArray_DATA(radii_arr), (uint32_t)n_radii,
    (real_t)delta_v, (real_t)delta_c, (real_t)expansion_factor, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!vsf) {
    PyErr_Format(PyExc_RuntimeError,
      "failed to evaluate the %s size function; see the sif log", name);
    return NULL;
  }

  return py_sif_wrap_size_function(vsf);
}

PyObject* py_sif_size_function_svdw(
  PyObject* self, PyObject* args, PyObject* kwds) {
  return __size_function(args, kwds, sif_size_function_svdw, "SvdW");
}

PyObject* py_sif_size_function_vdn(
  PyObject* self, PyObject* args, PyObject* kwds) {
  return __size_function(args, kwds, sif_size_function_vdn, "Vdn");
}

PyObject* py_sif_sigma_slope_pk(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* k_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  const char* window = NULL;

  static char* kwlist[] = {"k", "pk", "radii", "window", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "OOO|z", kwlist, &k_obj, &pk_obj, &radii_obj, &window)) {
    return NULL;
  }

  sif_option_t options = SIF_DEFAULT;
  if (py_sif_delta_parse_options(NULL, window, 0, &options) < 0)
    return NULL;

  PyArrayObject* k_arr = py_sif_as_real_array(k_obj, "k");
  if (!k_arr)
    return NULL;

  PyArrayObject* pk_arr = py_sif_as_real_array(pk_obj, "pk");
  if (!pk_arr) {
    Py_DECREF(k_arr);
    return NULL;
  }

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr) {
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    return NULL;
  }

  if (PyArray_SIZE(k_arr) != PyArray_SIZE(pk_arr)) {
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "k and pk must have the same length");
    return NULL;
  }

  const npy_intp n_points = PyArray_SIZE(k_arr);
  const npy_intp n_radii = PyArray_SIZE(radii_arr);

  real_t* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_sigma_slope_pk((const real_t*)PyArray_DATA(k_arr),
      (const real_t*)PyArray_DATA(pk_arr), (uint32_t)n_points,
      (const real_t*)PyArray_DATA(radii_arr), (uint32_t)n_radii, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the sigma slope; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n_radii);
}

PyObject* py_sif_expansion_factor(
  PyObject* self, PyObject* args, PyObject* kwds) {
  double delta_v;
  static char* kwlist[] = {"delta_v", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &delta_v))
    return NULL;

  if (!(delta_v < 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_v must be strictly negative");
    return NULL;
  }

  return PyFloat_FromDouble((double)sif_expansion_factor((real_t)delta_v));
}

PyObject* py_sif_multiplicity_function_svdw(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* sigma_obj;
  double delta_v = -2.7;
  double delta_c = 1.686;

  static char* kwlist[] = {"sigma", "delta_v", "delta_c", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O|dd", kwlist, &sigma_obj, &delta_v, &delta_c)) {
    return NULL;
  }

  if (!(delta_v < 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_v must be strictly negative");
    return NULL;
  }
  if (!(delta_c > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_c must be strictly positive");
    return NULL;
  }

  PyArrayObject* sigma_arr = py_sif_as_real_array(sigma_obj, "sigma");
  if (!sigma_arr)
    return NULL;

  const npy_intp n = PyArray_SIZE(sigma_arr);
  real_t* values = NULL;

  Py_BEGIN_ALLOW_THREADS values = sif_multiplicity_function_svdw(
    (const real_t*)PyArray_DATA(sigma_arr), (uint32_t)n, (real_t)delta_v,
    (real_t)delta_c);
  Py_END_ALLOW_THREADS

    Py_DECREF(sigma_arr);

  if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the multiplicity function; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n);
}
