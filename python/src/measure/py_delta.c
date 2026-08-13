/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_delta.h"

#include "py_delta_common.h"
#include "sif/measure/delta_distribution.h"
#include "sif/measure/delta_moments.h"
#include "structures/py_delta_distribution.h"
#include "structures/py_delta_moments.h"
#include "structures/py_grid.h"
#include <numpy/arrayobject.h>

/* Both entry points take a grid, so they share the unwrapping. */
static const sif_grid_t* grid_of(PyObject* grid_obj) {
  const sif_grid_t* grid = ((sifGridObject*)grid_obj)->grid;
  if (!grid || !grid->values) {
    PyErr_SetString(PyExc_ValueError,
      "the grid holds no density field; call assign_cic and "
      "compute_overdensity first");
    return NULL;
  }
  return grid;
}

PyObject* py_sif_delta_distribution_grid(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* grid_obj;
  PyObject* radii_obj;
  uint32_t n_bins;
  double delta_min;
  double delta_max;

  const char* shuffle = NULL;
  const char* window = NULL;
  unsigned long long seed = 42;
  int keep_cic_window = 0;

  static char* kwlist[] = {"grid", "radii", "n_bins", "delta_min", "delta_max",
    "shuffle", "window", "seed", "keep_cic_window", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OIdd|zzKp", kwlist,
        &sifGridType, &grid_obj, &radii_obj, &n_bins, &delta_min, &delta_max,
        &shuffle, &window, &seed, &keep_cic_window)) {
    return NULL;
  }

  if (delta_min >= delta_max) {
    PyErr_SetString(
      PyExc_ValueError, "delta_min must be strictly less than delta_max.");
    return NULL;
  }

  sif_option options = SIF_DEFAULT;
  if (py_sif_delta_parse_options(shuffle, window, keep_cic_window, &options) <
      0)
    return NULL;

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr)
    return NULL;

  const sif_grid_t* c_grid = grid_of(grid_obj);
  if (!c_grid) {
    Py_DECREF(radii_arr);
    return NULL;
  }

  const npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  const sif_real* c_radii = (const sif_real*)PyArray_DATA(radii_arr);

  sif_real delta_bounds[2] = {(sif_real)delta_min, (sif_real)delta_max};
  sif_delta_distribution_t* tmp = NULL;

  /* Release the GIL allowing seamless multi-threading in the C backend */
  Py_BEGIN_ALLOW_THREADS tmp = sif_delta_distribution_grid(c_grid, c_radii,
    (uint32_t)n_radii, n_bins, delta_bounds, (uint64_t)seed, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError,
      "Failed to compute the delta distribution; see the sif log for the "
      "cause.");
    return NULL;
  }

  return py_sif_wrap_distribution(tmp);
}

PyObject* py_sif_delta_moments_grid(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* grid_obj;
  PyObject* radii_obj;
  unsigned char order = 2;

  const char* shuffle = NULL;
  const char* window = NULL;
  unsigned long long n_tracers = 0;
  unsigned long long seed = 42;
  int keep_cic_window = 0;

  static char* kwlist[] = {"grid", "radii", "order", "shuffle", "window",
    "n_tracers", "seed", "keep_cic_window", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O|bzzKKp", kwlist,
        &sifGridType, &grid_obj, &radii_obj, &order, &shuffle, &window,
        &n_tracers, &seed, &keep_cic_window)) {
    return NULL;
  }

  sif_option options = SIF_DEFAULT;
  if (py_sif_delta_parse_options(shuffle, window, keep_cic_window, &options) <
      0)
    return NULL;

  PyArrayObject* radii_arr = py_sif_as_real_array(radii_obj, "radii");
  if (!radii_arr)
    return NULL;

  const sif_grid_t* c_grid = grid_of(grid_obj);
  if (!c_grid) {
    Py_DECREF(radii_arr);
    return NULL;
  }

  const npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  const sif_real* c_radii = (const sif_real*)PyArray_DATA(radii_arr);

  sif_delta_moments_t* tmp = NULL;

  Py_BEGIN_ALLOW_THREADS tmp =
    sif_delta_moments_grid(c_grid, c_radii, (uint32_t)n_radii, (uint8_t)order,
      (uint64_t)n_tracers, (uint64_t)seed, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError,
      "Failed to compute the moments from the field; see the sif log for the "
      "cause.");
    return NULL;
  }

  return py_sif_wrap_moments(tmp);
}
