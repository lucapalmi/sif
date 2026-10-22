/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_model.h"

#include "py_delta_common.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>

/* The covariance and the walk run in double whatever sif_real is, so these
 * mirror the sif_real helpers in py_delta_common.h at fixed precision. */

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

  memcpy(
    PyArray_DATA((PyArrayObject*)array), values, (size_t)n * sizeof(double));
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

  memcpy(
    PyArray_DATA((PyArrayObject*)array), values, (size_t)n * sizeof(uint64_t));
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

  PyObject* sigma_arr = PyArray_SimpleNew(1, (npy_intp*)&n_radii, NPY_REAL_T);
  PyObject* high_arr = PyArray_SimpleNew(1, (npy_intp*)&n_radii, NPY_REAL_T);

  /* Double, not sif_real: the derivative variance follows the covariance, which
   * is double for the same reason. */
  PyObject* dvar_arr = PyArray_SimpleNew(1, (npy_intp*)&n_radii, NPY_FLOAT64);

  if (!sigma_arr || !high_arr || !dvar_arr) {
    Py_XDECREF(sigma_arr);
    Py_XDECREF(high_arr);
    Py_XDECREF(dvar_arr);
    Py_DECREF(k_arr);
    Py_DECREF(pk_arr);
    Py_DECREF(radii_arr);
    return NULL;
  }

  double* cov = NULL;

  Py_BEGIN_ALLOW_THREADS cov = sif_delta_covariance_pk(
    (const sif_real*)PyArray_DATA(k_arr), (const sif_real*)PyArray_DATA(pk_arr),
    (uint32_t)n_points, (const sif_real*)PyArray_DATA(radii_arr),
    (uint32_t)n_radii, (sif_real*)PyArray_DATA((PyArrayObject*)sigma_arr),
    (sif_real*)PyArray_DATA((PyArrayObject*)high_arr),
    (double*)PyArray_DATA((PyArrayObject*)dvar_arr), options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!cov) {
    Py_DECREF(sigma_arr);
    Py_DECREF(high_arr);
    Py_DECREF(dvar_arr);
    PyErr_SetString(
      PyExc_RuntimeError, "failed to evaluate the covariance; see the sif log");
    return NULL;
  }

  PyObject* cov_arr =
    py_sif_owned_double_array(cov, (npy_intp)SIF_COV_SIZE((uint32_t)n_radii));
  if (!cov_arr) {
    Py_DECREF(sigma_arr);
    Py_DECREF(high_arr);
    Py_DECREF(dvar_arr);
    return NULL;
  }

  return Py_BuildValue("(NNNN)", cov_arr, sigma_arr, high_arr, dvar_arr);
}

PyObject* py_sif_ep_barrier_smt(
  PyObject* self, PyObject* args, PyObject* kwds) {
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
  sif_real* values =
    sif_ep_barrier_smt((const sif_real*)PyArray_DATA(sigma_arr), (uint32_t)n,
      (sif_real)alpha, (sif_real)beta, (sif_real)gamma);

  Py_DECREF(sigma_arr);

  if (!values) {
    PyErr_SetString(
      PyExc_RuntimeError, "failed to build the barrier; see the sif log");
    return NULL;
  }

  return py_sif_owned_array(values, n);
}

/* The collapse barrier: None for none, a number for a constant delta_c, or
 * one entry per radius for a moving one. Sets *out to NULL for None. */
static int as_collapse(PyObject* obj, npy_intp n, PyArrayObject** out) {
  *out = NULL;
  if (obj == NULL || obj == Py_None)
    return 0;

  if (PyFloat_Check(obj) || PyLong_Check(obj)) {
    const double v = PyFloat_AsDouble(obj);
    if (v == -1.0 && PyErr_Occurred())
      return -1;
    *out = (PyArrayObject*)PyArray_SimpleNew(1, &n, NPY_REAL_T);
    if (!*out)
      return -1;
    sif_real* d = (sif_real*)PyArray_DATA(*out);
    for (npy_intp i = 0; i < n; i++)
      d[i] = (sif_real)v;
    return 0;
  }

  *out = py_sif_as_real_array(obj, "collapse");
  if (!*out)
    return -1;
  if (PyArray_SIZE(*out) != n) {
    PyErr_Format(PyExc_ValueError,
      "collapse: %zd entries (expected a number or one per radius, %zd)",
      PyArray_SIZE(*out), n);
    Py_CLEAR(*out);
    return -1;
  }
  return 0;
}

/* delta_c is a magnitude; said here with the index rather than left to the C
 * log behind a RuntimeError. */
static int collapse_positive(PyArrayObject* c) {
  const sif_real* d = (const sif_real*)PyArray_DATA(c);
  for (npy_intp i = 0; i < PyArray_SIZE(c); i++) {
    if (!(d[i] > 0.0f && isfinite((double)d[i]))) {
      char msg[160];
      snprintf(msg, sizeof msg,
        "collapse[%zd]: %g (expected positive and finite: delta_c is a "
        "magnitude, like the void barrier)",
        (Py_ssize_t)i, (double)d[i]);
      PyErr_SetString(PyExc_ValueError, msg);
      return -1;
    }
  }
  return 0;
}

/* Shared argument handling for the two first-crossing entry points. Each has
 * its own keywords: the counts entry point returns the counts either way and
 * can add the collapse counts; the multiplicity can add the counts. */
typedef struct {
  PyArrayObject* radii;
  PyArrayObject* cov;
  PyArrayObject* barrier;
  PyArrayObject* collapse; /* NULL for the single void barrier */
  unsigned long long n_paths;
  unsigned long long seed;
  int return_extra;
} walk_args_t;

static void walk_args_release(walk_args_t* w) {
  Py_XDECREF(w->radii);
  Py_XDECREF(w->cov);
  Py_XDECREF(w->barrier);
  Py_XDECREF(w->collapse);
}

static int parse_walk_args(
  PyObject* args, PyObject* kwds, int counts_entry, walk_args_t* w) {

  PyObject* radii_obj;
  PyObject* cov_obj;
  PyObject* barrier_obj;
  PyObject* collapse_obj = Py_None;
  w->radii = w->cov = w->barrier = w->collapse = NULL;
  w->n_paths = 1000000;
  w->seed = 0;
  w->return_extra = 0;

  static char* kw_counts[] = {"radii", "cov", "barrier", "n_paths", "seed",
    "collapse", "return_collapse_counts", NULL};
  static char* kw_mult[] = {"radii", "cov", "barrier", "n_paths", "seed",
    "return_counts", "collapse", NULL};

  int ok = counts_entry
             ? PyArg_ParseTupleAndKeywords(args, kwds, "OOO|KKOp", kw_counts,
                 &radii_obj, &cov_obj, &barrier_obj, &w->n_paths, &w->seed,
                 &collapse_obj, &w->return_extra)
             : PyArg_ParseTupleAndKeywords(args, kwds, "OOO|KKpO", kw_mult,
                 &radii_obj, &cov_obj, &barrier_obj, &w->n_paths, &w->seed,
                 &w->return_extra, &collapse_obj);
  if (!ok)
    return -1;

  if (w->n_paths == 0) {
    PyErr_SetString(PyExc_ValueError, "n_paths must be strictly positive");
    return -1;
  }

  w->radii = py_sif_as_real_array(radii_obj, "radii");
  if (!w->radii)
    goto fail;
  w->cov = py_sif_as_double_array(cov_obj, "cov");
  if (!w->cov)
    goto fail;
  w->barrier = py_sif_as_real_array(barrier_obj, "barrier");
  if (!w->barrier)
    goto fail;

  const npy_intp n = PyArray_SIZE(w->radii);

  if (PyArray_SIZE(w->barrier) != n) {
    PyErr_SetString(PyExc_ValueError, "barrier must have one entry per radius");
    goto fail;
  }

  if (PyArray_SIZE(w->cov) != (npy_intp)SIF_COV_SIZE((uint32_t)n)) {
    PyErr_Format(PyExc_ValueError,
      "cov must be the packed lower triangle of %zd radii, i.e. %zd entries, "
      "not %zd",
      n, (npy_intp)SIF_COV_SIZE((uint32_t)n), PyArray_SIZE(w->cov));
    goto fail;
  }

  if (as_collapse(collapse_obj, n, &w->collapse) < 0)
    goto fail;
  if (w->collapse && collapse_positive(w->collapse) < 0)
    goto fail;

  return 0;

fail:
  walk_args_release(w);
  return -1;
}

static const sif_real* collapse_data(const walk_args_t* w) {
  return w->collapse ? (const sif_real*)PyArray_DATA(w->collapse) : NULL;
}

PyObject* py_sif_ep_first_crossing_counts(
  PyObject* self, PyObject* args, PyObject* kwds) {
  walk_args_t w;
  if (parse_walk_args(args, kwds, 1, &w) < 0)
    return NULL;

  npy_intp n = PyArray_SIZE(w.radii);

  PyObject* collapse_arr = NULL;
  uint64_t* collapse_counts = NULL;
  if (w.return_extra) {
    collapse_arr = PyArray_ZEROS(1, &n, NPY_UINT64, 0);
    if (!collapse_arr) {
      walk_args_release(&w);
      return NULL;
    }
    collapse_counts = (uint64_t*)PyArray_DATA((PyArrayObject*)collapse_arr);
  }

  uint64_t* counts = NULL;

  Py_BEGIN_ALLOW_THREADS counts =
    sif_ep_first_crossing_counts((const sif_real*)PyArray_DATA(w.radii),
      (uint32_t)n, (const double*)PyArray_DATA(w.cov),
      (const sif_real*)PyArray_DATA(w.barrier), collapse_data(&w),
      (uint64_t)w.n_paths, (uint64_t)w.seed, collapse_counts, SIF_DEFAULT);
  Py_END_ALLOW_THREADS

    walk_args_release(&w);

  if (!counts) {
    Py_XDECREF(collapse_arr);
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the first-crossing counts; see the sif log");
    return NULL;
  }

  PyObject* counts_arr = py_sif_owned_u64_array(counts, n);
  if (!counts_arr) {
    Py_XDECREF(collapse_arr);
    return NULL;
  }

  if (!w.return_extra)
    return counts_arr;

  return Py_BuildValue("(NN)", counts_arr, collapse_arr);
}

PyObject* py_sif_ep_multiplicity_function(
  PyObject* self, PyObject* args, PyObject* kwds) {
  walk_args_t w;
  if (parse_walk_args(args, kwds, 0, &w) < 0)
    return NULL;

  npy_intp n = PyArray_SIZE(w.radii);

  /* Allocated up front so the walk can write straight into it: the whole
   * point of the out-parameter is that the counts cost nothing extra. */
  PyObject* counts_arr = NULL;
  uint64_t* counts_data = NULL;

  if (w.return_extra) {
    counts_arr = PyArray_SimpleNew(1, &n, NPY_UINT64);
    if (!counts_arr) {
      walk_args_release(&w);
      return NULL;
    }
    counts_data = (uint64_t*)PyArray_DATA((PyArrayObject*)counts_arr);
  }

  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_ep_multiplicity_function((const sif_real*)PyArray_DATA(w.radii),
      (uint32_t)n, (const double*)PyArray_DATA(w.cov),
      (const sif_real*)PyArray_DATA(w.barrier), collapse_data(&w),
      (uint64_t)w.n_paths, (uint64_t)w.seed, counts_data, SIF_DEFAULT);
  Py_END_ALLOW_THREADS

    walk_args_release(&w);

  if (!values) {
    Py_XDECREF(counts_arr);
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the multiplicity function; see the sif log");
    return NULL;
  }

  PyObject* mult_arr = py_sif_owned_array(values, n - 1);
  if (!mult_arr) {
    Py_XDECREF(counts_arr);
    return NULL;
  }

  if (!w.return_extra)
    return mult_arr;

  return Py_BuildValue("(NN)", mult_arr, counts_arr);
}

/*
 * The emulator. Deliberately a different signature from the walk above: it
 * reads only the covariance DIAGONAL, so it takes sigma where the Monte Carlo
 * takes the packed triangle, and its cost is linear rather than quadratic in
 * the radius count.
 */
PyObject* py_sif_ep_multiplicity_function_emu(
  PyObject* self, PyObject* args, PyObject* kwds) {

  (void)self;

  PyObject *radii_obj, *sigma_obj, *barrier_obj, *dvar_obj;
  PyObject* collapse_obj = Py_None;
  PyObject* cov_obj = Py_None;
  int return_domain = 0;
  int binned = 0;

  static char* kwlist[] = {"radii", "sigma", "barrier", "deriv_variance",
    "return_domain", "binned", "collapse", "cov", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|ppOO", kwlist, &radii_obj,
        &sigma_obj, &barrier_obj, &dvar_obj, &return_domain, &binned,
        &collapse_obj, &cov_obj)) {
    return NULL;
  }

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr)
    return NULL;

  /* One at a time, and bail on the first failure: calling into the C API
   * again with an exception already set can mask the original error and
   * trips the assertions in a debug build of Python. */
  PyArrayObject* sigma_arr = NULL;
  PyArrayObject* barrier_arr = NULL;
  PyArrayObject* dvar_arr = NULL;
  PyArrayObject* collapse_arr = NULL;
  PyArrayObject* cov_arr = NULL;

  sigma_arr = py_sif_as_real_array(sigma_obj, "sigma");
  if (!sigma_arr)
    goto fail;

  barrier_arr = py_sif_as_real_array(barrier_obj, "barrier");
  if (!barrier_arr)
    goto fail;

  dvar_arr = py_sif_as_double_array(dvar_obj, "deriv_variance");
  if (!dvar_arr)
    goto fail;

  const npy_intp n = PyArray_SIZE(radii_arr);

  if (PyArray_SIZE(sigma_arr) != n || PyArray_SIZE(barrier_arr) != n ||
      PyArray_SIZE(dvar_arr) != n) {
    PyErr_SetString(PyExc_ValueError,
      "sigma, barrier and deriv_variance must each have one entry per radius");
    goto fail;
  }

  if (as_collapse(collapse_obj, n, &collapse_arr) < 0)
    goto fail;
  if (collapse_arr) {
    if (collapse_positive(collapse_arr) < 0)
      goto fail;
    if (cov_obj == Py_None) {
      PyErr_SetString(PyExc_ValueError,
        "collapse needs cov: the packed covariance from delta_covariance_pk, "
        "which void-in-cloud conditions on");
      goto fail;
    }
    cov_arr = py_sif_as_double_array(cov_obj, "cov");
    if (!cov_arr)
      goto fail;
    if (PyArray_SIZE(cov_arr) != (npy_intp)SIF_COV_SIZE((uint32_t)n)) {
      PyErr_Format(PyExc_ValueError,
        "cov: %zd entries (expected the packed triangle of %zd radii, %zd)",
        PyArray_SIZE(cov_arr), n, (npy_intp)SIF_COV_SIZE((uint32_t)n));
      goto fail;
    }
  }

  sif_emu_domain_t domain;
  sif_real* values = NULL;

  Py_BEGIN_ALLOW_THREADS values =
    sif_ep_multiplicity_function_emu((const sif_real*)PyArray_DATA(radii_arr),
      (uint32_t)n, (const sif_real*)PyArray_DATA(sigma_arr),
      (const sif_real*)PyArray_DATA(barrier_arr),
      (const double*)PyArray_DATA(dvar_arr),
      collapse_arr ? (const sif_real*)PyArray_DATA(collapse_arr) : NULL,
      cov_arr ? (const double*)PyArray_DATA(cov_arr) : NULL, &domain,
      binned ? SIF_EP_EMU_BIN_CENTRES : SIF_EP_EMU_AT_RADII);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);
  Py_DECREF(sigma_arr);
  Py_DECREF(barrier_arr);
  Py_DECREF(dvar_arr);
  Py_XDECREF(collapse_arr);
  Py_XDECREF(cov_arr);

  if (!values) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to evaluate the emulated multiplicity function; see the sif log");
    return NULL;
  }

  PyObject* mult_arr = py_sif_owned_array(values, binned ? n - 1 : n);
  if (!mult_arr)
    return NULL;

  if (!return_domain)
    return mult_arr;

  /* A dict rather than a tuple: these are five unrelated numbers, and a
   * caller reading domain["in_domain"] cannot get the order wrong. */
  PyObject* d = Py_BuildValue("{s:O,s:I,s:d,s:d,s:d}", "in_domain",
    domain.in_domain ? Py_True : Py_False, "n_bins_outside",
    (unsigned int)domain.n_bins_outside, "nu_origin", (double)domain.nu_origin,
    "first_step_mass", (double)domain.first_step_mass, "expected_error",
    (double)domain.expected_error);

  if (!d) {
    Py_DECREF(mult_arr);
    return NULL;
  }

  return Py_BuildValue("(NN)", mult_arr, d);

fail:
  Py_XDECREF(radii_arr);
  Py_XDECREF(sigma_arr);
  Py_XDECREF(barrier_arr);
  Py_XDECREF(dvar_arr);
  Py_XDECREF(collapse_arr);
  Py_XDECREF(cov_arr);
  return NULL;
}
