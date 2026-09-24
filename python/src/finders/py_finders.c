/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_finders.h"

#include "structures/py_catalog.h"
#include "structures/py_chain_mesh.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
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

/*
 * The search factor as an option bit. The C side carries the reach as two
 * bits of the option word rather than a float, so anything asked for here
 * lands on the nearest of the four it can represent. Snapping rather than
 * rejecting keeps the argument a physical quantity the caller can sweep: 1.4
 * and 1.6 both mean "about one and a half", and neither is worth an
 * exception. The value actually used is reported back by the finder's own log
 * line. Shared by every entry point that takes one, so a survey box is always
 * sized for the reach the survey finder will then use.
 *
 * @return 0 with *out set, -1 with a ValueError.
 */
static int search_factor_option(double search_factor, sif_option* out) {
  static const double allowed[4] = {1.25, 1.5, 1.75, 2.0};
  static const sif_option flag[4] = {SIF_FINDER_SEARCH_1_25,
    SIF_FINDER_SEARCH_1_50, SIF_FINDER_SEARCH_1_75, SIF_FINDER_SEARCH_2_00};

  if (!(search_factor > 1.0) || !(search_factor < 1e6)) {
    PyErr_SetString(PyExc_ValueError,
      "search_factor must be greater than 1 (it is a multiple of the rung "
      "radius); it is snapped to the nearest of 1.25, 1.5, 1.75, 2.0");
    return -1;
  }

  int best = 0;
  for (int k = 1; k < 4; k++) {
    const double d = search_factor - allowed[k];
    const double b = search_factor - allowed[best];
    if ((d < 0 ? -d : d) < (b < 0 ? -b : b))
      best = k;
  }

  *out = flag[best];
  return 0;
}

/* The radii argument as a contiguous sif_real array; NULL with an exception
 * set. */
static PyArrayObject* radii_from_object(PyObject* radii_obj) {
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

  return radii_arr;
}

/* A new Catalog owning `cat`, or NULL with an exception set (and `cat`
 * freed). */
static PyObject* wrap_catalog(sif_catalog_t* cat) {
  sifCatalogObject* out =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!out) {
    sif_catalog_free(cat);
    return PyErr_NoMemory();
  }

  out->catalog = cat;
  return (PyObject*)out;
}

/* --- Exodus Finder Wrapper --- */
PyObject* py_sif_finder_exodus(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *grid_obj = NULL, *mesh_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;

  /* Optional keyword arguments with safe defaults */
  int consume_grid = 0;
  double search_factor = 1.5;

  static char* kwlist[] = {"grid", "mesh", "radii", "threshold",
    "overlap_fraction", "consume_grid", "search_factor", NULL};

  /* The '|' character denotes that everything after it is optional.
   * 'p' safely converts a Python boolean to a C int (1 or 0). */
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!Od|dpd", kwlist,
        &sifGridType, &grid_obj, &sifChainMeshType, &mesh_obj, &radii_obj,
        &threshold, &overlap_frac, &consume_grid, &search_factor)) {
    return NULL;
  }

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr)
    return NULL;

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  sif_real* radii_data = (sif_real*)PyArray_DATA(radii_arr);

  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;
  const sif_chain_mesh_t* c_mesh = ((sifChainMeshObject*)mesh_obj)->mesh;

  /* --- Parse Options to Bitmask --- */
  sif_option options = 0;

  if (consume_grid)
    options |= SIF_FINDER_CONSUME_GRID;

  sif_option reach;
  if (search_factor_option(search_factor, &reach) < 0) {
    Py_DECREF(radii_arr);
    return NULL;
  }
  options |= reach;

  /* Execute the algorithm. The GIL is released: this runs for minutes to
   * hours across every core, and holding it would freeze the interpreter and
   * swallow Ctrl-C. Nothing below touches Python state.
   *
   * The mesh is borrowed for the duration; mesh_obj is kept alive by the
   * caller's reference for the whole call, so it cannot be collected here. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS res_catalog =
    sif_finder_exodus(c_grid, c_mesh, radii_data, (uint32_t)n_radii,
      (sif_real)threshold, (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res_catalog) {
    PyErr_SetString(
      PyExc_RuntimeError, "Exodus finder execution failed. Check system logs.");
    return NULL;
  }

  return wrap_catalog(res_catalog);
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

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr)
    return NULL;

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  sif_real* radii_data = (sif_real*)PyArray_DATA(radii_arr);
  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;

  /* --- Parse Options to Bitmask --- */
  sif_option options = 0;

  if (consume_grid)
    options |= SIF_FINDER_CONSUME_GRID;

  /* See the note in the exodus wrapper: the GIL is released for the run. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS res_catalog = sif_finder_spherical(c_grid, radii_data,
    (uint32_t)n_radii, (sif_real)threshold, (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res_catalog) {
    PyErr_SetString(PyExc_RuntimeError, "Spherical finder execution failed.");
    return NULL;
  }

  return wrap_catalog(res_catalog);
}

/* --- Survey Finder Wrapper --- */
PyObject* py_sif_finder_exodus_survey(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject *data_grid_obj = NULL, *random_grid_obj = NULL;
  PyObject *data_mesh_obj = NULL, *random_mesh_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;
  int consume_grid = 0;
  double search_factor = 1.5;

  static char* kwlist[] = {"data_grid", "random_grid", "data_mesh",
    "random_mesh", "radii", "threshold", "overlap_fraction", "consume_grid",
    "search_factor", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!Od|dpd", kwlist,
        &sifGridType, &data_grid_obj, &sifGridType, &random_grid_obj,
        &sifChainMeshType, &data_mesh_obj, &sifChainMeshType, &random_mesh_obj,
        &radii_obj, &threshold, &overlap_frac, &consume_grid, &search_factor)) {
    return NULL;
  }

  /* One grid passed as both would be smoothed twice in two workspaces, and
   * restored twice over itself. */
  if (data_grid_obj == random_grid_obj || data_mesh_obj == random_mesh_obj) {
    PyErr_SetString(PyExc_ValueError,
      "the data and the randoms need a grid and a mesh each, not a shared one");
    return NULL;
  }

  sif_option options = consume_grid ? SIF_FINDER_CONSUME_GRID : SIF_DEFAULT;
  sif_option reach;
  if (search_factor_option(search_factor, &reach) < 0)
    return NULL;
  options |= reach;

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr)
    return NULL;

  const npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  const sif_real* radii_data = (const sif_real*)PyArray_DATA(radii_arr);

  sif_grid_t* data_grid = ((sifGridObject*)data_grid_obj)->grid;
  sif_grid_t* random_grid = ((sifGridObject*)random_grid_obj)->grid;
  const sif_chain_mesh_t* data_mesh =
    ((sifChainMeshObject*)data_mesh_obj)->mesh;
  const sif_chain_mesh_t* random_mesh =
    ((sifChainMeshObject*)random_mesh_obj)->mesh;

  /* See the note in the exodus wrapper: the GIL is released for the run, and
   * the four borrowed objects are kept alive by the caller's references. */
  sif_catalog_t* res = NULL;
  Py_BEGIN_ALLOW_THREADS res = sif_finder_exodus_survey(data_grid, random_grid,
    data_mesh, random_mesh, radii_data, (uint32_t)n_radii, (sif_real)threshold,
    (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res) {
    PyErr_SetString(PyExc_RuntimeError,
      "Survey finder execution failed. Check system logs: the usual cause is "
      "a survey too close to the box faces, and the log says how much padding "
      "it needs -- survey_box() works it out.");
    return NULL;
  }

  return wrap_catalog(res);
}

/* --- Survey Box Helper --- */
PyObject* py_sif_finder_survey_box(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject *randoms_obj = NULL, *radii_obj = NULL;
  unsigned int n_cells;
  double search_factor = 1.5;

  static char* kwlist[] = {
    "randoms", "radii", "n_cells", "search_factor", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OI|d", kwlist, &sifFieldType,
        &randoms_obj, &radii_obj, &n_cells, &search_factor)) {
    return NULL;
  }

  sif_option options;
  if (search_factor_option(search_factor, &options) < 0)
    return NULL;

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr)
    return NULL;

  const sif_field_t* randoms = ((sifFieldObject*)randoms_obj)->field;
  sif_real offset[3];
  sif_real box_length = 0.0f;

  const int status = sif_finder_exodus_survey_box(randoms,
    (const sif_real*)PyArray_DATA(radii_arr),
    (uint32_t)PyArray_SHAPE(radii_arr)[0], (uint32_t)n_cells, options, offset,
    &box_length);
  Py_DECREF(radii_arr);

  if (status != SIF_OK) {
    PyErr_SetString(PyExc_ValueError,
      "could not size a survey box: the randoms must hold positions and "
      "n_cells must be at least 16");
    return NULL;
  }

  npy_intp dims[1] = {3};
  PyObject* offset_arr = PyArray_SimpleNew(1, dims, NPY_REAL_T);
  if (!offset_arr)
    return NULL;
  sif_real* o = (sif_real*)PyArray_DATA((PyArrayObject*)offset_arr);
  o[0] = offset[0];
  o[1] = offset[1];
  o[2] = offset[2];

  return Py_BuildValue("(Nd)", offset_arr, (double)box_length);
}
