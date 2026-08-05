#include "py_model.h"

#include "py_delta_common.h"
#include "sif/model/deltamoments.h"
#include "sif/model/excursionset.h"
#include <numpy/arrayobject.h>
#include <string.h>

/* The covariance and the walk run in double whatever real_t is, so these
 * mirror the real_t helpers in py_delta_common.h at fixed precision. */

static PyArrayObject* py_sif_as_double_array(PyObject* obj, const char* name) {
  PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(
    obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!arr) {
    PyErr_Format(PyExc_TypeError, "%s must be a 1D array of numbers", name);
    return NULL;
  }
  if (PyArray_NDIM(arr) != 1 || PyArray_SIZE(arr) == 0) {
    Py_DECREF(arr);
    PyErr_Format(PyExc_ValueError, "%s must be a non-empty 1D array", name);
    return NULL;
  }
  return arr;
}

static PyObject* py_sif_owned_double_array(double* values, npy_intp n) {
  if (!values)
    return NULL;

  PyObject* array = PyArray_SimpleNew(1, &n, NPY_FLOAT64);
  if (!array) {
    sif_free_aligned(values);
    return NULL;
  }

  memcpy(PyArray_DATA((PyArrayObject*)array), values,
    (size_t)n * sizeof(double));
  sif_free_aligned(values);

  return array;
}

static PyObject* py_sif_owned_u64_array(uint64_t* values, npy_intp n) {
  if (!values)
    return NULL;

  PyObject* array = PyArray_SimpleNew(1, &n, NPY_UINT64);
  if (!array) {
    sif_free_aligned(values);
    return NULL;
  }

  memcpy(PyArray_DATA((PyArrayObject*)array), values,
    (size_t)n * sizeof(uint64_t));
  sif_free_aligned(values);

  return array;
}

PyObject* py_sif_delta_covariance_pk(
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

  PyObject* sigma_arr = PyArray_SimpleNew(1, (npy_intp*)&n_radii, NPY_REAL_T);
  PyObject* high_arr = PyArray_SimpleNew(1, (npy_intp*)&n_radii, NPY_REAL_T);

  if (!sigma_arr || !high_arr) {
    Py_XDECREF(sigma_arr);
    Py_XDECREF(high_arr);
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    Py_DECREF(radii_arr);
    return NULL;
  }

  double* cov = NULL;

  Py_BEGIN_ALLOW_THREADS cov = sif_delta_covariance_pk(
    (const real_t*)PyArray_DATA(k_arr), (const real_t*)PyArray_DATA(pk_arr),
    (uint32_t)n_points, (const real_t*)PyArray_DATA(radii_arr),
    (uint32_t)n_radii, (real_t*)PyArray_DATA((PyArrayObject*)sigma_arr),
    (real_t*)PyArray_DATA((PyArrayObject*)high_arr), options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!cov) {
    Py_DECREF(sigma_arr);
    Py_DECREF(high_arr);
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the covariance; see the sif log");
    return NULL;
  }

  PyObject* cov_arr = py_sif_owned_double_array(
    cov, (npy_intp)SIF_COV_SIZE((uint32_t)n_radii));
  if (!cov_arr) {
    Py_DECREF(sigma_arr);
    Py_DECREF(high_arr);
    return NULL;
  }

  return Py_BuildValue("(NNN)", cov_arr, sigma_arr, high_arr);
}

PyObject* py_sif_barrier_smt(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* sigma_obj;
  double alpha, beta, gamma;

  static char* kwlist[] = {"sigma", "alpha", "beta", "gamma", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "Oddd", kwlist, &sigma_obj, &alpha, &beta, &gamma)) {
    return NULL;
  }

  if (!(alpha > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "alpha must be strictly positive");
    return NULL;
  }
  if (!(beta > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "beta must be strictly positive");
    return NULL;
  }

  PyArrayObject* sigma_arr = py_sif_as_real_array(sigma_obj, "sigma");
  if (!sigma_arr)
    return NULL;

  const npy_intp n = PyArray_SIZE(sigma_arr);
  real_t* values = sif_barrier_smt((const real_t*)PyArray_DATA(sigma_arr),
    (uint32_t)n, (real_t)alpha, (real_t)beta, (real_t)gamma);

  Py_DECREF(sigma_arr);

  if (!values) {
    PyErr_SetString(
      PyExc_RuntimeError, "failed to build the barrier; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n);
}

/* Shared argument handling for the two first-crossing entry points. */
static int __parse_walk_args(PyObject* args, PyObject* kwds,
  PyArrayObject** radii_arr, PyArrayObject** cov_arr,
  PyArrayObject** barrier_arr, unsigned long long* n_paths,
  unsigned long long* seed) {

  PyObject* radii_obj;
  PyObject* cov_obj;
  PyObject* barrier_obj;
  *n_paths = 1000000;
  *seed = 0;

  static char* kwlist[] = {
    "radii", "cov", "barrier", "n_paths", "seed", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|KK", kwlist, &radii_obj,
        &cov_obj, &barrier_obj, n_paths, seed)) {
    return -1;
  }

  if (*n_paths == 0) {
    PyErr_SetString(PyExc_ValueError, "n_paths must be strictly positive");
    return -1;
  }

  *radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!*radii_arr)
    return -1;

  *cov_arr = py_sif_as_double_array(cov_obj, "cov");
  if (!*cov_arr) {
    Py_DECREF(*radii_arr);
    return -1;
  }

  *barrier_arr = py_sif_as_real_array(barrier_obj, "barrier");
  if (!*barrier_arr) {
    Py_DECREF(*radii_arr);
    Py_DECREF(*cov_arr);
    return -1;
  }

  const npy_intp n = PyArray_SIZE(*radii_arr);

  if (PyArray_SIZE(*barrier_arr) != n) {
    PyErr_SetString(
      PyExc_ValueError, "barrier must have one entry per radius");
    goto fail;
  }

  if (PyArray_SIZE(*cov_arr) != (npy_intp)SIF_COV_SIZE((uint32_t)n)) {
    PyErr_Format(PyExc_ValueError,
      "cov must be the packed lower triangle of %zd radii, i.e. %zd entries, "
      "not %zd",
      n, (npy_intp)SIF_COV_SIZE((uint32_t)n), PyArray_SIZE(*cov_arr));
    goto fail;
  }

  return 0;

fail:
  Py_DECREF(*radii_arr);
  Py_DECREF(*cov_arr);
  Py_DECREF(*barrier_arr);
  return -1;
}

PyObject* py_sif_first_crossing_counts_ep(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyArrayObject *radii_arr, *cov_arr, *barrier_arr;
  unsigned long long n_paths, seed;

  if (__parse_walk_args(
        args, kwds, &radii_arr, &cov_arr, &barrier_arr, &n_paths, &seed) < 0) {
    return NULL;
  }

  const npy_intp n = PyArray_SIZE(radii_arr);
  uint64_t* counts = NULL;

  Py_BEGIN_ALLOW_THREADS counts = sif_first_crossing_counts_ep(
    (const real_t*)PyArray_DATA(radii_arr), (uint32_t)n,
    (const double*)PyArray_DATA(cov_arr),
    (const real_t*)PyArray_DATA(barrier_arr), (uint64_t)n_paths, (uint64_t)seed,
    SIF_DEFAULT);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);
  Py_DECREF(cov_arr);
  Py_DECREF(barrier_arr);

  if (!counts) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the first-crossing counts; see the sif log");
    return NULL;
  }

  return py_sif_owned_u64_array(counts, n);
}

PyObject* py_sif_multiplicity_function_ep(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyArrayObject *radii_arr, *cov_arr, *barrier_arr;
  unsigned long long n_paths, seed;

  if (__parse_walk_args(
        args, kwds, &radii_arr, &cov_arr, &barrier_arr, &n_paths, &seed) < 0) {
    return NULL;
  }

  const npy_intp n = PyArray_SIZE(radii_arr);
  real_t* values = NULL;

  Py_BEGIN_ALLOW_THREADS values = sif_multiplicity_function_ep(
    (const real_t*)PyArray_DATA(radii_arr), (uint32_t)n,
    (const double*)PyArray_DATA(cov_arr),
    (const real_t*)PyArray_DATA(barrier_arr), (uint64_t)n_paths, (uint64_t)seed,
    SIF_DEFAULT);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);
  Py_DECREF(cov_arr);
  Py_DECREF(barrier_arr);

  if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the multiplicity function; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n - 1);
}
