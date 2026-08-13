/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_finders.h"

#include "structures/py_catalog.h"
#include "structures/py_chain_mesh.h"
#include "structures/py_grid.h"
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/finder/rescaled_spherical_finder.h"
#include "sif/finder/spherical_finder.h"

/* --- Mesh Sizing Helper --- */
PyObject* py_sif_finder_suggest_mesh_cells(
  PyObject* self, PyObject* args, PyObject* kwds) {

  unsigned long long n_particles;
  double box_length;
  double max_radius = 0.0;

  static char* kwlist[] = {"n_particles", "box_length", "max_radius", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "Kd|d", kwlist, &n_particles, &box_length, &max_radius)) {
    return NULL;
  }

  /* The C helper folds every failure into a 0 return, which as a Python value
   * is a footgun: it would reach ChainMesh and fail there with an unrelated
   * message. The two ways it can happen are separated out here so each says
   * what is actually wrong. */
  /* PyErr_Format has no float conversion, so anything carrying a double has to
   * be rendered before it gets there. */
  char detail[256];

  if (n_particles == 0 || !(box_length > 0.0)) {
    snprintf(detail, sizeof(detail),
      "need n_particles > 0 and box_length > 0, got n_particles=%llu, "
      "box_length=%g",
      n_particles, box_length);
    PyErr_SetString(PyExc_ValueError, detail);
    return NULL;
  }

  if (max_radius > 0.0 && 2.0 * max_radius >= box_length) {
    snprintf(detail, sizeof(detail),
      "a search sphere of %g (twice max_radius) does not fit in a box of %g, "
      "so no mesh resolution can hold it",
      2.0 * max_radius, box_length);
    PyErr_SetString(PyExc_ValueError, detail);
    return NULL;
  }

  const uint32_t n_cells = sif_finder_suggest_mesh_cells(
    (uint64_t)n_particles, (sif_real)box_length, (sif_real)max_radius);

  if (n_cells == 0) {
    PyErr_SetString(
      PyExc_ValueError, "no usable mesh resolution for this geometry");
    return NULL;
  }

  return PyLong_FromUnsignedLong(n_cells);
}

/* --- Rescaled Spherical Finder Wrapper --- */
PyObject* py_sif_finder_rescaled_spherical(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *grid_obj = NULL, *mesh_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;

  /* Optional keyword arguments with safe defaults */
  int consume_grid = 0;

  static char* kwlist[] = {"grid", "mesh", "radii", "threshold",
    "overlap_fraction", "consume_grid", NULL};

  /* The '|' character denotes that everything after it is optional.
   * 'p' safely converts a Python boolean to a C int (1 or 0). */
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!Od|dp", kwlist,
        &sifGridType, &grid_obj, &sifChainMeshType, &mesh_obj, &radii_obj,
        &threshold, &overlap_frac, &consume_grid)) {
    return NULL;
  }

  PyArrayObject* radii_arr = (PyArrayObject*)PyArray_FROM_OTF(
    radii_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!radii_arr) {
    PyErr_Format(PyExc_TypeError,
      "radii must be a 1D contiguous %s NumPy array",
      sizeof(sif_real) == 8 ? "float64" : "float32");
    return NULL;
  }

  if (PyArray_NDIM(radii_arr) != 1 || PyArray_SIZE(radii_arr) == 0) {
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "radii must be a non-empty 1D array");
    return NULL;
  }

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  sif_real* radii_data = (sif_real*)PyArray_DATA(radii_arr);

  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;
  const sif_chain_mesh_t* c_mesh = ((sifChainMeshObject*)mesh_obj)->mesh;

  /* --- Parse Options to Bitmask --- */
  sif_option options = 0;

  if (consume_grid)
    options |= SIF_FINDER_CONSUME_GRID;

  /* Execute the algorithm. The GIL is released: this runs for minutes to
   * hours across every core, and holding it would freeze the interpreter and
   * swallow Ctrl-C. Nothing below touches Python state.
   *
   * The mesh is borrowed for the duration; mesh_obj is kept alive by the
   * caller's reference for the whole call, so it cannot be collected here. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS res_catalog =
    sif_finder_rescaled_spherical(c_grid, c_mesh, radii_data, (uint32_t)n_radii,
      (sif_real)threshold, (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res_catalog) {
    PyErr_SetString(PyExc_RuntimeError,
      "Rescaled spherical finder execution failed. Check system logs.");
    return NULL;
  }

  sifCatalogObject* out_cat =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!out_cat) {
    sif_catalog_free(res_catalog);
    return PyErr_NoMemory();
  }

  out_cat->catalog = res_catalog;
  return (PyObject*)out_cat;
}

/* --- Spherical Finder Wrapper --- */
PyObject* py_sif_finder_spherical(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *grid_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;

  /* Optional keyword arguments with safe defaults */
  int consume_grid = 0;

  static char* kwlist[] = {
    "grid", "radii", "threshold", "overlap_fraction", "consume_grid", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Od|dp", kwlist, &sifGridType,
        &grid_obj, &radii_obj, &threshold, &overlap_frac, &consume_grid)) {
    return NULL;
  }

  PyArrayObject* radii_arr = (PyArrayObject*)PyArray_FROM_OTF(
    radii_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!radii_arr)
    return NULL;

  if (PyArray_NDIM(radii_arr) != 1 || PyArray_SIZE(radii_arr) == 0) {
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "radii must be a non-empty 1D array");
    return NULL;
  }

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  sif_real* radii_data = (sif_real*)PyArray_DATA(radii_arr);
  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;

  /* --- Parse Options to Bitmask --- */
  sif_option options = 0;

  if (consume_grid)
    options |= SIF_FINDER_CONSUME_GRID;

  /* See the note in the rescaled wrapper: the GIL is released for the run. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS res_catalog = sif_finder_spherical(c_grid, radii_data,
    (uint32_t)n_radii, (sif_real)threshold, (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res_catalog) {
    PyErr_SetString(PyExc_RuntimeError, "Spherical finder execution failed.");
    return NULL;
  }

  sifCatalogObject* out_cat =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!out_cat) {
    sif_catalog_free(res_catalog);
    return PyErr_NoMemory();
  }

  out_cat->catalog = res_catalog;
  return (PyObject*)out_cat;
}
