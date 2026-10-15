/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_finders.h"

#include "structures/py_catalogue.h"
#include "structures/py_chain_mesh.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/finder/spherical_finder.h"

/*
 * The optional volume the sizing helpers take: None for the whole box, or the
 * volume the tracers occupy. 0 means the box to the C side, so it is refused
 * here rather than read as a second way to say None.
 *
 * @return 0 with *out set (0 for None), -1 with an exception.
 */
static int volume_argument(PyObject* obj, double* out) {
  *out = 0.0;
  if (obj == Py_None)
    return 0;

  const double v = PyFloat_AsDouble(obj);
  if (v == -1.0 && PyErr_Occurred())
    return -1;
  if (!(v > 0.0) || !isfinite(v)) {
    char detail[128];
    snprintf(detail, sizeof(detail),
      "volume: %g, expected positive and finite (None for the box)", v);
    PyErr_SetString(PyExc_ValueError, detail);
    return -1;
  }
  *out = v;
  return 0;
}

/* --- Mesh Sizing Helper --- */
PyObject* py_sif_finder_suggest_mesh_cells(
  PyObject* self, PyObject* args, PyObject* kwds) {

  unsigned long long n_particles;
  double box_length;
  double max_radius = 0.0;
  PyObject* volume_obj = Py_None;

  static char* kwlist[] = {
    "n_particles", "box_length", "max_radius", "volume", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kd|dO", kwlist, &n_particles,
        &box_length, &max_radius, &volume_obj)) {
    return NULL;
  }

  double volume;
  if (volume_argument(volume_obj, &volume) < 0)
    return NULL;

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

  const uint32_t n_cells = sif_finder_suggest_mesh_cells((uint64_t)n_particles,
    (sif_real)box_length, (sif_real)max_radius, (sif_real)volume);

  if (n_cells == 0) {
    PyErr_SetString(
      PyExc_ValueError, "no usable mesh resolution for this geometry");
    return NULL;
  }

  return PyLong_FromUnsignedLong(n_cells);
}

/* --- Grid Sizing Helper --- */
PyObject* py_sif_finder_grid_cells_from_size(
  PyObject* self, PyObject* args, PyObject* kwds) {

  unsigned long long n_particles;
  double box_length;
  double cell_size;
  const char* units = "mps";
  PyObject* volume_obj = Py_None;

  static char* kwlist[] = {
    "n_particles", "box_length", "cell_size", "units", "volume", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kdd|sO", kwlist, &n_particles,
        &box_length, &cell_size, &units, &volume_obj))
    return NULL;

  double volume;
  if (volume_argument(volume_obj, &volume) < 0)
    return NULL;

  /* The same spellings sif-exodus takes for its radii. */
  sif_finder_units_t u;
  if (strcmp(units, "mps") == 0)
    u = SIF_FINDER_UNITS_MPS;
  else if (strcmp(units, "physical") == 0)
    u = SIF_FINDER_UNITS_PHYSICAL;
  else {
    PyErr_Format(
      PyExc_ValueError, "units: '%s', expected 'mps' or 'physical'", units);
    return NULL;
  }

  sif_error_clear();
  const uint32_t n_cells =
    sif_finder_grid_cells_from_size((uint64_t)n_particles, (sif_real)box_length,
      (sif_real)cell_size, u, (sif_real)volume);
  if (n_cells == 0)
    return py_sif_raise(SIF_ERR_INVALID, "grid_cells_from_size");

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

/* A new Catalogue owning `cat`, or NULL with an exception set (and `cat`
 * freed). */
static PyObject* wrap_catalogue(sif_catalogue_t* cat) {
  sifCatalogueObject* out =
    (sifCatalogueObject*)sifCatalogueType.tp_alloc(&sifCatalogueType, 0);
  if (!out) {
    sif_catalogue_free(cat);
    return PyErr_NoMemory();
  }

  out->catalogue = cat;
  return (PyObject*)out;
}

/*
 * What the exodus entry points take beyond the finder's own arguments:
 * checkpoint= -- None for a name of the finder's own, False for none, or a
 * path (str or os.PathLike) -- and prior=, None or a Catalogue. The path's
 * bytes are held here, since the run reads them with the GIL released.
 */
typedef struct {
  PyObject* path_bytes;
  sif_finder_exodus_extras_t extras;
  sif_option options;
} exodus_extras_t;

static int exodus_extras(
  PyObject* checkpoint, PyObject* prior, exodus_extras_t* out) {

  memset(out, 0, sizeof(*out));

  if (checkpoint == Py_False) {
    out->options = SIF_FINDER_NO_CHECKPOINT;
  } else if (checkpoint == Py_True) {
    PyErr_SetString(PyExc_ValueError,
      "checkpoint: True (expected a path, None for a name of the run's own, "
      "or False for none)");
    return -1;
  } else if (checkpoint != Py_None) {
    if (!PyUnicode_FSConverter(checkpoint, &out->path_bytes))
      return -1;
    out->extras.checkpoint = PyBytes_AS_STRING(out->path_bytes);
  }

  if (prior != Py_None) {
    if (!PyObject_TypeCheck(prior, &sifCatalogueType)) {
      Py_CLEAR(out->path_bytes);
      PyErr_Format(PyExc_TypeError,
        "prior: expected a Catalogue or None, got %s", Py_TYPE(prior)->tp_name);
      return -1;
    }
    out->extras.prior = ((sifCatalogueObject*)prior)->catalogue;
  }
  return 0;
}

/* --- Exodus Finder Wrapper --- */
PyObject* py_sif_finder_exodus(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *grid_obj = NULL, *mesh_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;

  /* Optional keyword arguments with safe defaults */
  int consume_grid = 0;
  double search_factor = 1.5;
  PyObject* checkpoint = Py_None;
  PyObject* prior = Py_None;

  static char* kwlist[] = {"grid", "mesh", "radii", "threshold",
    "overlap_fraction", "consume_grid", "search_factor", "checkpoint", "prior",
    NULL};

  /* The '|' character denotes that everything after it is optional.
   * 'p' safely converts a Python boolean to a C int (1 or 0). */
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!Od|dpd$OO", kwlist,
        &sifGridType, &grid_obj, &sifChainMeshType, &mesh_obj, &radii_obj,
        &threshold, &overlap_frac, &consume_grid, &search_factor, &checkpoint,
        &prior)) {
    return NULL;
  }

  exodus_extras_t x;
  if (exodus_extras(checkpoint, prior, &x) < 0)
    return NULL;

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr) {
    Py_XDECREF(x.path_bytes);
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

  sif_option reach;
  if (search_factor_option(search_factor, &reach) < 0) {
    Py_DECREF(radii_arr);
    Py_XDECREF(x.path_bytes);
    return NULL;
  }
  options |= reach | x.options;

  /* Execute the algorithm. The GIL is released: this runs for minutes to
   * hours across every core, and holding it would freeze the interpreter and
   * swallow Ctrl-C. Nothing below touches Python state.
   *
   * The mesh is borrowed for the duration; mesh_obj is kept alive by the
   * caller's reference for the whole call, so it cannot be collected here. */
  sif_catalogue_t* res_catalogue = NULL;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS res_catalogue =
    sif_finder_exodus_ex(c_grid, c_mesh, radii_data, (uint32_t)n_radii,
      (sif_real)threshold, (sif_real)overlap_frac, options, &x.extras);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);
  Py_XDECREF(x.path_bytes);

  if (!res_catalogue)
    return py_sif_raise(sif_error_status(), "exodus");

  return wrap_catalogue(res_catalogue);
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
  sif_catalogue_t* res_catalogue = NULL;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS res_catalogue = sif_finder_spherical(c_grid, radii_data,
    (uint32_t)n_radii, (sif_real)threshold, (sif_real)overlap_frac, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);

  if (!res_catalogue)
    return py_sif_raise(sif_error_status(), "spherical");

  return wrap_catalogue(res_catalogue);
}

/* --- Survey Finder Wrapper --- */
PyObject* py_sif_finder_exodus_survey(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject *data_grid_obj = NULL, *random_grid_obj = NULL;
  PyObject *data_mesh_obj = NULL, *random_mesh_obj = NULL, *radii_obj = NULL;
  double ox, oy, oz;
  double threshold;
  double overlap_frac = 0.0;
  int consume_grid = 0;
  double search_factor = 1.5;
  PyObject* checkpoint = Py_None;
  PyObject* prior = Py_None;

  static char* kwlist[] = {"data_grid", "random_grid", "data_mesh",
    "random_mesh", "offset", "radii", "threshold", "overlap_fraction",
    "consume_grid", "search_factor", "checkpoint", "prior", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!(ddd)Od|dpd$OO", kwlist,
        &sifGridType, &data_grid_obj, &sifGridType, &random_grid_obj,
        &sifChainMeshType, &data_mesh_obj, &sifChainMeshType, &random_mesh_obj,
        &ox, &oy, &oz, &radii_obj, &threshold, &overlap_frac, &consume_grid,
        &search_factor, &checkpoint, &prior)) {
    return NULL;
  }
  const sif_real offset[3] = {(sif_real)ox, (sif_real)oy, (sif_real)oz};

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

  exodus_extras_t x;
  if (exodus_extras(checkpoint, prior, &x) < 0)
    return NULL;
  options |= reach | x.options;

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr) {
    Py_XDECREF(x.path_bytes);
    return NULL;
  }

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
  sif_catalogue_t* res = NULL;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS res = sif_finder_exodus_survey_ex(data_grid,
    random_grid, data_mesh, random_mesh, offset, radii_data, (uint32_t)n_radii,
    (sif_real)threshold, (sif_real)overlap_frac, options, &x.extras);
  Py_END_ALLOW_THREADS

    Py_DECREF(radii_arr);
  Py_XDECREF(x.path_bytes);

  if (!res)
    return py_sif_raise(sif_error_status(), "exodus_survey");

  return wrap_catalogue(res);
}

/* --- Survey Box Helper --- */
PyObject* py_sif_finder_survey_box(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject *data_obj = NULL, *randoms_obj = NULL, *radii_obj = NULL;
  double search_factor = 1.5;

  static char* kwlist[] = {"data", "randoms", "radii", "search_factor", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O|d", kwlist, &sifFieldType,
        &data_obj, &sifFieldType, &randoms_obj, &radii_obj, &search_factor)) {
    return NULL;
  }

  sif_option options;
  if (search_factor_option(search_factor, &options) < 0)
    return NULL;
  if (py_sif_field_check_cartesian(
        (sifFieldObject*)data_obj, "size a survey box around it") < 0 ||
      py_sif_field_check_cartesian(
        (sifFieldObject*)randoms_obj, "size a survey box around it") < 0)
    return NULL;

  PyArrayObject* radii_arr = radii_from_object(radii_obj);
  if (!radii_arr)
    return NULL;

  const sif_field_t* data = ((sifFieldObject*)data_obj)->field;
  const sif_field_t* randoms = ((sifFieldObject*)randoms_obj)->field;
  sif_real offset[3];
  sif_real box_length = 0.0f;

  sif_error_clear();
  const int status = sif_finder_exodus_survey_box(data, randoms,
    (const sif_real*)PyArray_DATA(radii_arr),
    (uint32_t)PyArray_SHAPE(radii_arr)[0], options, offset, &box_length);
  Py_DECREF(radii_arr);

  if (status != SIF_OK)
    return py_sif_raise(status, "survey_box");

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

/* --- Checkpoint Reader --- */
PyObject* py_sif_finder_read_checkpoint(
  PyObject* self, PyObject* args, PyObject* kwds) {

  PyObject* path_bytes = NULL;
  static char* kwlist[] = {"path", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&", kwlist, PyUnicode_FSConverter, &path_bytes))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_bytes);

  sif_finder_exodus_checkpoint_t h;
  sif_error_clear();
  sif_catalogue_t* cat = sif_finder_exodus_checkpoint_read(path, &h);
  if (!cat) {
    Py_DECREF(path_bytes);
    return py_sif_raise(sif_error_status(), "read_checkpoint");
  }

  /* The ladder follows the header; the reader above has already checked the
   * file whole, size and checksum, so it is there to be read. */
  npy_intp dims[1] = {(npy_intp)h.n_radii};
  PyObject* radii = PyArray_SimpleNew(1, dims, NPY_REAL_T);
  FILE* f = radii ? fopen(path, "rb") : NULL;
  const int ok = f && fseek(f, (long)sizeof(h), SEEK_SET) == 0 &&
                 fread(PyArray_DATA((PyArrayObject*)radii), sizeof(sif_real),
                   h.n_radii, f) == h.n_radii;
  if (f)
    fclose(f);
  Py_DECREF(path_bytes);
  if (!ok) {
    Py_XDECREF(radii);
    sif_catalogue_free(cat);
    if (!PyErr_Occurred())
      PyErr_SetFromErrno(PyExc_OSError);
    return NULL;
  }

  PyObject* catalogue = wrap_catalogue(cat);
  if (!catalogue) {
    Py_DECREF(radii);
    return NULL;
  }

  PyObject* info = Py_BuildValue(
    "{s:s,s:O,s:N,s:I,s:K,s:K,s:d,s:d,s:d,s:O,s:I,s:d,s:K,s:d,s:K,s:d,"
    "s:(ddd)}",
    "mode", h.is_survey ? "survey" : "box", "weighted",
    h.is_weighted ? Py_True : Py_False, "radii", radii, "rungs_done",
    (unsigned)h.next_rung, "n_voids", (unsigned long long)h.n_voids, "n_prior",
    (unsigned long long)h.n_prior, "threshold", h.threshold, "overlap_fraction",
    h.overlap_fraction, "search_factor",
    (double)sif__finder_search_factor(h.options), "keep_cic_window",
    (h.options & SIF_FINDER_KEEP_CIC_WINDOW) ? Py_True : Py_False,
    "grid_n_cells", (unsigned)h.grid_n_cells, "box_length", h.box_length,
    "n_particles", (unsigned long long)h.n_particles, "total_weight",
    h.total_weight, "n_randoms", (unsigned long long)h.random_n_particles,
    "random_total_weight", h.random_total_weight, "offset", h.offset[0],
    h.offset[1], h.offset[2]);
  if (!info) {
    Py_DECREF(catalogue);
    return NULL;
  }

  return Py_BuildValue("(NN)", catalogue, info);
}
