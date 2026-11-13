/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_model.h"

#include "py_delta_common.h"
#include "py_variance_kernel.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/model/size_function.h"
#include "structures/py_size_function.h"
#include <numpy/arrayobject.h>
#include <string.h>

/* Both size functions take the same arguments; only the model differs. */
typedef sif_size_function_t* (*vsf_fn)(const sif_real*, const sif_real*,
  uint32_t, const sif_real*, uint32_t, sif_real, sif_real, sif_real,
  sif_option);

/* Forward-declared: shared with the mapping entry points further down. */
static int py_sif_spherical_option(const char* method, PyObject* omega_m_obj,
  sif_option* opt, double* omega_m);

/* The units=, method= and omega_m= every model size function takes. */
static int py_sif_vsf_options(const char* units, const char* method,
  PyObject* omega_m_obj, sif_option* options, double* omega_m) {

  *options = SIF_DEFAULT;
  if (units == NULL || strcmp(units, "ln_r") == 0) {
    *options |= SIF_VSF_BIN_LN;
  } else if (strcmp(units, "r") == 0) {
    *options |= SIF_VSF_BIN_LINEAR;
  } else {
    PyErr_Format(PyExc_ValueError,
      "units must be None, 'ln_r' (dn/dlnR) or 'r' (dn/dR), not '%s'", units);
    return -1;
  }

  sif_option spherical;
  if (py_sif_spherical_option(method, omega_m_obj, &spherical, omega_m) != 0)
    return -1;
  *options |= spherical;
  return 0;
}

static PyObject* size_function(
  PyObject* args, PyObject* kwds, vsf_fn compute, const char* name) {
  PyObject* k_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  double delta_v = -2.7;
  double delta_c = 1.686;
  const char* units = NULL;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;

  static char* kwlist[] = {"k", "pk", "radii", "delta_v", "delta_c", "units",
    "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|ddzzO", kwlist, &k_obj,
        &pk_obj, &radii_obj, &delta_v, &delta_c, &units, &method,
        &omega_m_obj)) {
    return NULL;
  }

  sif_option options;
  double omega_m;
  if (py_sif_vsf_options(units, method, omega_m_obj, &options, &omega_m) != 0)
    return NULL;

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

  Py_BEGIN_ALLOW_THREADS vsf = compute((const sif_real*)PyArray_DATA(k_arr),
    (const sif_real*)PyArray_DATA(pk_arr), (uint32_t)n_points,
    (const sif_real*)PyArray_DATA(radii_arr), (uint32_t)n_radii,
    (sif_real)delta_v, (sif_real)delta_c, (sif_real)omega_m, options);
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
  return size_function(args, kwds, sif_size_function_svdw, "SvdW");
}

PyObject* py_sif_size_function_vdn(
  PyObject* self, PyObject* args, PyObject* kwds) {
  return size_function(args, kwds, sif_size_function_vdn, "Vdn");
}

/* The checks both EVM entry points make on their common arguments. */
static int evm_check_scalars(double delta_v, double h) {
  if (!(delta_v < 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_v must be strictly negative");
    return -1;
  }
  if (!(h > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "h must be strictly positive");
    return -1;
  }
  return 0;
}

PyObject* py_sif_size_function_evm(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* k_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  double delta_v;
  double h;
  const char* units = NULL;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;

  static char* kwlist[] = {
    "k", "pk", "radii", "delta_v", "h", "units", "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOdd|zzO", kwlist, &k_obj,
        &pk_obj, &radii_obj, &delta_v, &h, &units, &method, &omega_m_obj)) {
    return NULL;
  }

  sif_option options;
  double omega_m;
  if (py_sif_vsf_options(units, method, omega_m_obj, &options, &omega_m) != 0 ||
      evm_check_scalars(delta_v, h) != 0) {
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

  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS vsf = sif_size_function_evm(
    (const sif_real*)PyArray_DATA(k_arr), (const sif_real*)PyArray_DATA(pk_arr),
    (uint32_t)n_points, (const sif_real*)PyArray_DATA(radii_arr),
    (uint32_t)n_radii, (sif_real)delta_v, (sif_real)h, (sif_real)omega_m,
    options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!vsf)
    return py_sif_raise(sif_error_status(), "size_function_evm");

  return py_sif_wrap_size_function(vsf);
}

PyObject* py_sif_size_function_evm_kernel(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* kernel_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  double delta_v;
  double h;
  const char* units = NULL;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;

  static char* kwlist[] = {"kernel", "pk", "radii", "delta_v", "h", "units",
    "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OOdd|zzO", kwlist,
        &sifVarianceKernelType, &kernel_obj, &pk_obj, &radii_obj, &delta_v, &h,
        &units, &method, &omega_m_obj)) {
    return NULL;
  }

  const sif_variance_kernel_t* kernel =
    ((sifVarianceKernelObject*)kernel_obj)->kernel;
  if (!kernel) {
    PyErr_SetString(PyExc_RuntimeError, "VarianceKernel: not initialised");
    return NULL;
  }

  sif_option options;
  double omega_m;
  if (py_sif_vsf_options(units, method, omega_m_obj, &options, &omega_m) != 0 ||
      evm_check_scalars(delta_v, h) != 0) {
    return NULL;
  }

  PyArrayObject* pk_arr = py_sif_as_real_array(pk_obj, "pk");
  if (!pk_arr)
    return NULL;

  if (PyArray_SIZE(pk_arr) != (npy_intp)kernel->n_k) {
    PyErr_Format(PyExc_ValueError, "pk: %zd values (expected one per node, %u)",
      PyArray_SIZE(pk_arr), kernel->n_k);
    Py_DECREF(pk_arr);
    return NULL;
  }

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr) {
    Py_DECREF(pk_arr);
    return NULL;
  }

  const npy_intp n_radii = PyArray_SIZE(radii_arr);

  sif_size_function_t* vsf = NULL;

  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS vsf =
    sif_size_function_evm_kernel(kernel, (const sif_real*)PyArray_DATA(pk_arr),
      (const sif_real*)PyArray_DATA(radii_arr), (uint32_t)n_radii,
      (sif_real)delta_v, (sif_real)h, (sif_real)omega_m, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!vsf)
    return py_sif_raise(sif_error_status(), "size_function_evm");

  return py_sif_wrap_size_function(vsf);
}

PyObject* py_sif_size_function_evm_walk(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* walk_obj;
  PyObject* radii_obj;
  double delta_v;
  double h;
  const char* units = NULL;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;

  static char* kwlist[] = {
    "walk", "radii", "delta_v", "h", "units", "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Odd|zzO", kwlist,
        &sifWalkType, &walk_obj, &radii_obj, &delta_v, &h, &units, &method,
        &omega_m_obj)) {
    return NULL;
  }

  const sif_walk_t* walk = ((sifWalkObject*)walk_obj)->walk;
  if (!walk) {
    PyErr_SetString(PyExc_RuntimeError, "Walk: not initialised");
    return NULL;
  }

  sif_option options;
  double omega_m;
  if (py_sif_vsf_options(units, method, omega_m_obj, &options, &omega_m) != 0 ||
      evm_check_scalars(delta_v, h) != 0) {
    return NULL;
  }

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr)
    return NULL;

  const npy_intp n_radii = PyArray_SIZE(radii_arr);

  sif_size_function_t* vsf = NULL;

  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS vsf =
    sif_size_function_evm_walk(walk, (const sif_real*)PyArray_DATA(radii_arr),
      (uint32_t)n_radii, (sif_real)delta_v, (sif_real)h, (sif_real)omega_m,
      options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!vsf)
    return py_sif_raise(sif_error_status(), "size_function_evm");

  return py_sif_wrap_size_function(vsf);
}

PyObject* py_sif_delta_sigma_slope_pk(
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

  sif_option options = SIF_DEFAULT;
  if (py_sif_delta_parse_options(NULL, window, 0, NULL, &options) < 0)
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

  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_delta_sigma_slope_pk((const sif_real*)PyArray_DATA(k_arr),
      (const sif_real*)PyArray_DATA(pk_arr), (uint32_t)n_points,
      (const sif_real*)PyArray_DATA(radii_arr), (uint32_t)n_radii, options);
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

/* method= and omega_m=: omega_m is required by 'lcdm' and refused by the
 * others, so no cosmology is ever assumed. */
static int py_sif_spherical_option(const char* method, PyObject* omega_m_obj,
  sif_option* opt, double* omega_m) {
  *omega_m = 1.0;
  if (!method || strcmp(method, "b94") == 0) {
    *opt = SIF_SPHERICAL_B94;
  } else if (strcmp(method, "eds") == 0) {
    *opt = SIF_SPHERICAL_EDS;
  } else if (strcmp(method, "lcdm") == 0) {
    *opt = SIF_SPHERICAL_LCDM;
  } else {
    PyErr_Format(PyExc_ValueError,
      "method: '%s' (expected 'b94', 'eds' or 'lcdm')", method);
    return -1;
  }

  if (*opt != SIF_SPHERICAL_LCDM) {
    if (omega_m_obj != Py_None) {
      PyErr_SetString(
        PyExc_ValueError, "omega_m: given, but only method='lcdm' reads it");
      return -1;
    }
    return 0;
  }

  if (omega_m_obj == Py_None) {
    PyErr_SetString(PyExc_ValueError,
      "omega_m: required by method='lcdm' (Omega_m at the epoch of the voids)");
    return -1;
  }
  *omega_m = PyFloat_AsDouble(omega_m_obj);
  if (*omega_m == -1.0 && PyErr_Occurred())
    return -1;
  if (!(*omega_m > 0.0 && *omega_m <= 1.0)) {
    PyErr_Format(
      PyExc_ValueError, "omega_m: %g (expected 0 < omega_m <= 1)", *omega_m);
    return -1;
  }
  return 0;
}

PyObject* py_sif_spherical_map_nonlinear(
  PyObject* self, PyObject* args, PyObject* kwds) {
  double delta_linear;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;
  static char* kwlist[] = {"delta_linear", "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "d|zO", kwlist, &delta_linear, &method, &omega_m_obj))
    return NULL;

  if (!(delta_linear < 0.0)) {
    PyErr_SetString(PyExc_ValueError, "delta_linear must be strictly negative");
    return NULL;
  }

  sif_option opt;
  double omega_m;
  if (py_sif_spherical_option(method, omega_m_obj, &opt, &omega_m) != 0)
    return NULL;

  return PyFloat_FromDouble((double)sif_spherical_map_nonlinear(
    (sif_real)delta_linear, (sif_real)omega_m, opt));
}

PyObject* py_sif_spherical_map_linear(
  PyObject* self, PyObject* args, PyObject* kwds) {
  double delta_nonlinear;
  const char* method = NULL;
  PyObject* omega_m_obj = Py_None;
  static char* kwlist[] = {"delta_nonlinear", "method", "omega_m", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "d|zO", kwlist, &delta_nonlinear, &method, &omega_m_obj))
    return NULL;

  if (!(delta_nonlinear < 0.0) || !(delta_nonlinear > -1.0)) {
    PyErr_SetString(
      PyExc_ValueError, "delta_nonlinear must lie strictly between -1 and 0");
    return NULL;
  }

  sif_option opt;
  double omega_m;
  if (py_sif_spherical_option(method, omega_m_obj, &opt, &omega_m) != 0)
    return NULL;

  return PyFloat_FromDouble((double)sif_spherical_map_linear(
    (sif_real)delta_nonlinear, (sif_real)omega_m, opt));
}

PyObject* py_sif_svdw_multiplicity_function(
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
  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_svdw_multiplicity_function((const sif_real*)PyArray_DATA(sigma_arr),
      (uint32_t)n, (sif_real)delta_v, (sif_real)delta_c);
  Py_END_ALLOW_THREADS

    Py_DECREF(sigma_arr);

  if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the multiplicity function; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n);
}
