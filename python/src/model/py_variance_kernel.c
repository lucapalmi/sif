/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_variance_kernel.h"

#include "py_delta_common.h"
#include "sif/utils/align.h"

#include <math.h>
#include <numpy/arrayobject.h>

/* The nodes a kernel gets when the caller names none: enough for 3e-6 in the
 * derivative variance of a CDM spectrum, from horizon scales to well past
 * where a void-scale window has any weight. */
#define DEFAULT_N_K  600
#define DEFAULT_K_LO 1e-4
#define DEFAULT_K_HI 100.0

/* A double buffer owned by `owner`, as a read-only array that keeps it
 * alive: py_sif_wrap_borrowed, at fixed precision. */
static PyObject* wrap_borrowed_double(
  PyObject* owner, npy_intp n, double* data) {

  PyObject* array = PyArray_SimpleNewFromData(1, &n, NPY_FLOAT64, data);
  if (!array)
    return NULL;

  Py_INCREF(owner);
  if (PyArray_SetBaseObject((PyArrayObject*)array, owner) < 0) {
    Py_DECREF(array);
    return NULL;
  }

  PyArray_CLEARFLAGS((PyArrayObject*)array, NPY_ARRAY_WRITEABLE);
  return array;
}

static const char* window_name(sif_option window) {
  return window == SIF_DELTA_FILTER_GAUSSIAN ? "gaussian" : "top_hat";
}

/* --- Walk --- */

static void sifWalk_dealloc(PyObject* self_obj) {
  sifWalkObject* self = (sifWalkObject*)self_obj;
  sif_walk_free(self->walk);
  self->walk = NULL;
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifWalk_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  static char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;
  ((sifWalkObject*)self_obj)->walk = NULL;
  return 0;
}

static PyObject* sifWalk_get_n_radii(PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(w->n_radii);
}

static PyObject* sifWalk_get_radii(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  npy_intp dims[1] = {w->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, w->radii);
}

static PyObject* sifWalk_get_sigma(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  npy_intp dims[1] = {w->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, w->sigma);
}

static PyObject* sifWalk_get_deriv_variance(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  return wrap_borrowed_double(self_obj, w->n_radii, w->deriv_variance);
}

static PyObject* sifWalk_get_k(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  npy_intp dims[1] = {w->n_k};
  return py_sif_wrap_borrowed(self_obj, 1, dims, w->k);
}

static PyObject* sifWalk_get_pk(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  npy_intp dims[1] = {w->n_k};
  return py_sif_wrap_borrowed(self_obj, 1, dims, w->pk);
}

static PyObject* sifWalk_get_window(PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_walk_t* w = ((sifWalkObject*)self_obj)->walk;
  if (!w)
    Py_RETURN_NONE;
  return PyUnicode_FromString(window_name(w->window));
}

static PyGetSetDef sifWalk_getset[] = {
  {"n_radii", sifWalk_get_n_radii, NULL, "Number of lattice radii", NULL},
  {"radii", sifWalk_get_radii, NULL,
    "The kernel's radius lattice, ascending, in Mpc/h", NULL},
  {"sigma", sifWalk_get_sigma, NULL,
    "sigma(R) at each lattice radius; the one to build the barrier from", NULL},
  {"deriv_variance", sifWalk_get_deriv_variance, NULL,
    "<(d delta / dS)^2>, S = sigma^2, at each lattice radius (float64)", NULL},
  {"k", sifWalk_get_k, NULL, "The kernel's nodes", NULL},
  {"pk", sifWalk_get_pk, NULL,
    "The spectrum the walk was built from, at the nodes", NULL},
  {"window", sifWalk_get_window, NULL, "'top_hat' or 'gaussian'", NULL},
  {NULL}};

PyTypeObject sifWalkType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.model.Walk",
  .tp_basicsize = sizeof(sifWalkObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifWalk_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Walk()\n"
            "--\n\n"
            "sigma(R) and the derivative variance on a variance kernel's\n"
            "radius lattice, for one power spectrum: what\n"
            "ep_multiplicity_function_emu takes.\n\n"
            "Returned by calling a VarianceKernel on a P(k). Build the\n"
            "barrier from walk.sigma, so the barrier and the walk share one\n"
            "variance. The arrays are read-only views that keep the walk\n"
            "alive.\n\n"
            "Not constructed directly.",
  .tp_getset = sifWalk_getset,
  .tp_init = sifWalk_init,
  .tp_new = PyType_GenericNew,
};

/* --- VarianceKernel --- */

static void sifVarianceKernel_dealloc(PyObject* self_obj) {
  sifVarianceKernelObject* self = (sifVarianceKernelObject*)self_obj;
  sif_variance_kernel_free(self->kernel);
  self->kernel = NULL;
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifVarianceKernel_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {

  sifVarianceKernelObject* self = (sifVarianceKernelObject*)self_obj;

  PyObject* k_obj = Py_None;
  double r_min = 0.5, r_max = 400.0, dln_r = 0.02;
  const char* window = NULL;

  static char* kwlist[] = {"k", "r_min", "r_max", "dln_r", "window", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|Odddz", kwlist, &k_obj, &r_min, &r_max, &dln_r, &window))
    return -1;

  sif_option options = SIF_DEFAULT;
  if (py_sif_delta_parse_options(NULL, window, 0, NULL, &options) < 0)
    return -1;

  PyArrayObject* k_arr = NULL;
  if (k_obj == Py_None) {
    npy_intp n = DEFAULT_N_K;
    k_arr = (PyArrayObject*)PyArray_SimpleNew(1, &n, NPY_REAL_T);
    if (!k_arr)
      return -1;
    sif_real* d = (sif_real*)PyArray_DATA(k_arr);
    for (npy_intp m = 0; m < n; m++)
      d[m] = (sif_real)(DEFAULT_K_LO * pow(DEFAULT_K_HI / DEFAULT_K_LO,
                                         (double)m / (double)(n - 1)));
  } else {
    k_arr = py_sif_as_real_array(k_obj, "k");
    if (!k_arr)
      return -1;
  }

  sif_variance_kernel_t* kernel = NULL;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS kernel = sif_variance_kernel_alloc(
    (const sif_real*)PyArray_DATA(k_arr), (uint32_t)PyArray_SIZE(k_arr),
    (sif_real)r_min, (sif_real)r_max, (sif_real)dln_r, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);

  if (!kernel) {
    py_sif_raise(sif_error_status(), "VarianceKernel");
    return -1;
  }

  sif_variance_kernel_free(self->kernel);
  self->kernel = kernel;
  return 0;
}

static PyObject* sifVarianceKernel_call(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {

  sifVarianceKernelObject* self = (sifVarianceKernelObject*)self_obj;
  if (!self->kernel) {
    PyErr_SetString(PyExc_RuntimeError, "VarianceKernel: not initialised");
    return NULL;
  }

  PyObject* pk_obj;
  static char* kwlist[] = {"pk", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &pk_obj))
    return NULL;

  PyArrayObject* pk_arr = py_sif_as_real_array(pk_obj, "pk");
  if (!pk_arr)
    return NULL;

  if (PyArray_SIZE(pk_arr) != (npy_intp)self->kernel->n_k) {
    PyErr_Format(PyExc_ValueError, "pk: %zd values (expected one per node, %u)",
      PyArray_SIZE(pk_arr), self->kernel->n_k);
    Py_DECREF(pk_arr);
    return NULL;
  }

  sif_walk_t* walk = NULL;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS walk =
    sif_walk_pk(self->kernel, (const sif_real*)PyArray_DATA(pk_arr));
  Py_END_ALLOW_THREADS

    Py_DECREF(pk_arr);

  if (!walk)
    return py_sif_raise(sif_error_status(), "VarianceKernel");

  sifWalkObject* out = PyObject_New(sifWalkObject, &sifWalkType);
  if (!out) {
    sif_walk_free(walk);
    return NULL;
  }
  out->walk = walk;
  return (PyObject*)out;
}

static PyObject* sifVarianceKernel_get_k(PyObject* self_obj, void* closure) {
  (void)closure;
  sif_variance_kernel_t* kn = ((sifVarianceKernelObject*)self_obj)->kernel;
  if (!kn)
    Py_RETURN_NONE;
  npy_intp dims[1] = {kn->n_k};
  return py_sif_wrap_borrowed(self_obj, 1, dims, kn->k);
}

static PyObject* sifVarianceKernel_get_radii(
  PyObject* self_obj, void* closure) {
  (void)closure;
  sif_variance_kernel_t* kn = ((sifVarianceKernelObject*)self_obj)->kernel;
  if (!kn)
    Py_RETURN_NONE;
  npy_intp dims[1] = {kn->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, kn->radii);
}

static PyObject* sifVarianceKernel_get_dln_r(
  PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_variance_kernel_t* kn =
    ((sifVarianceKernelObject*)self_obj)->kernel;
  if (!kn)
    Py_RETURN_NONE;
  return PyFloat_FromDouble(kn->dln_r);
}

static PyObject* sifVarianceKernel_get_window(
  PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_variance_kernel_t* kn =
    ((sifVarianceKernelObject*)self_obj)->kernel;
  if (!kn)
    Py_RETURN_NONE;
  return PyUnicode_FromString(window_name(kn->window));
}

static PyGetSetDef sifVarianceKernel_getset[] = {
  {"k", sifVarianceKernel_get_k, NULL,
    "The nodes, ascending: where the P(k) passed to the kernel is sampled",
    NULL},
  {"radii", sifVarianceKernel_get_radii, NULL,
    "The radius lattice, r_min e^(j dln_r), ascending, in Mpc/h", NULL},
  {"dln_r", sifVarianceKernel_get_dln_r, NULL, "Lattice spacing in ln R", NULL},
  {"window", sifVarianceKernel_get_window, NULL, "'top_hat' or 'gaussian'",
    NULL},
  {NULL}};

PyTypeObject sifVarianceKernelType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.model.VarianceKernel",
  .tp_basicsize = sizeof(sifVarianceKernelObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifVarianceKernel_dealloc,
  .tp_call = sifVarianceKernel_call,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "VarianceKernel(k=None, r_min=0.5, r_max=400.0, dln_r=0.02, "
    "window='top_hat')\n"
    "--\n\n"
    "sigma(R) and the derivative variance as a precomputed linear map from\n"
    "P(k): built once, then kernel(pk) gives a Walk in tens of\n"
    "microseconds.\n\n"
    "sigma^2, dS/dR and <(d delta / dR)^2> are each linear in P(k), and\n"
    "every transcendental sits in the window terms, which do not depend on\n"
    "the spectrum. The kernel integrates those once, resolving the\n"
    "window's oscillation at every radius, so it is also more accurate\n"
    "than delta_covariance_pk on a table: 3e-6 in the derivative variance\n"
    "of a CDM spectrum at the default 600 nodes, against 1.5e-4 for an\n"
    "8000-point table at R = 150.\n\n"
    "Args:\n"
    "    k: The nodes P(k) will be given at, ascending, at least 4; the\n"
    "        integrals stop at the first and last. A Boltzmann code's or a\n"
    "        P(k) emulator's own array can be passed as it is. None, the\n"
    "        default, is 600 log-spaced from 1e-4 to 100 h/Mpc.\n"
    "    r_min, r_max: The radius lattice, in Mpc/h. It must reach out to\n"
    "        where the barrier is ~3.5 sigma, and in to the smallest radius\n"
    "        the multiplicity is wanted at.\n"
    "    dln_r: Lattice spacing in ln R; 0.02 holds the emulated\n"
    "        multiplicity within 2e-4 of a lattice twice as fine.\n"
    "    window: 'top_hat' (default) or 'gaussian'.\n\n"
    "Raises:\n"
    "    ValueError: For nodes that are not positive and increasing, or a\n"
    "        lattice of fewer than 4 radii.",
  .tp_getset = sifVarianceKernel_getset,
  .tp_init = sifVarianceKernel_init,
  .tp_new = PyType_GenericNew,
};
