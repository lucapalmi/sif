#include "py_finders.h"

#include "structures/py_catalog.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"
#include <numpy/arrayobject.h>

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/finder/rescaled_spherical_finder.h"
#include "sif/finder/spherical_finder.h"

/* --- Rescaled Spherical Finder Wrapper --- */
PyObject* py_sif_finder_rescaled_spherical(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *grid_obj = NULL, *field_obj = NULL, *radii_obj = NULL;
  double threshold;
  double overlap_frac = 0.0;

  /* Optional keyword arguments with safe defaults */
  int center_is_min = 0;
  int refine_hessian = 0;
  int preserve_grid = 0;

  static char* kwlist[] = {"grid", "field", "radii", "threshold",
    "center_is_minimum", "refine_center_hessian", "overlap_fraction",
    "preserve_grid", NULL};

  /* The '|' character denotes that everything after it is optional.
   * 'p' safely converts a Python boolean to a C int (1 or 0). */
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!Od|ppdp", kwlist,
        &sifGridType, &grid_obj, &sifFieldType, &field_obj, &radii_obj,
        &threshold, &center_is_min, &refine_hessian, &overlap_frac,
        &preserve_grid)) {
    return NULL;
  }

  PyArrayObject* radii_arr =
    (PyArrayObject*)PyArray_FROM_OTF(radii_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!radii_arr) {
    PyErr_Format(PyExc_TypeError,
      "radii must be a 1D contiguous %s NumPy array",
      sizeof(real_t) == 8 ? "float64" : "float32");
    return NULL;
  }

  if (PyArray_NDIM(radii_arr) != 1 || PyArray_SIZE(radii_arr) == 0) {
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "radii must be a non-empty 1D array");
    return NULL;
  }

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  real_t* radii_data = (real_t*)PyArray_DATA(radii_arr);

  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;
  sif_field_t* c_field = ((sifFieldObject*)field_obj)->field;

  /* --- Parse Options to Bitmask --- */
  sif_option_t options = 0;

  if (center_is_min)
    options |= SIF_FINDER_CENTER_IS_MINIMUM;
  if (refine_hessian)
    options |= SIF_FINDER_REFINE_CENTER_HESSIAN;
  if (preserve_grid)
    options |= SIF_FINDER_PRESERVE_GRID;

  /* Execute the algorithm. The GIL is released: this runs for minutes to
   * hours across every core, and holding it would freeze the interpreter and
   * swallow Ctrl-C. Nothing below touches Python state. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS
  res_catalog = sif_finder_rescaled_spherical(c_grid, c_field, radii_data,
    (uint32_t)n_radii, (real_t)threshold, (real_t)overlap_frac, options);
  Py_END_ALLOW_THREADS

  Py_DECREF(radii_arr);

  if (!res_catalog) {
    PyErr_SetString(
      PyExc_RuntimeError, "Rescaled spherical finder execution failed. Check system logs.");
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
  int center_is_min = 0;
  int refine_hessian = 0;
  int preserve_grid = 0;

  static char* kwlist[] = {"grid", "radii", "threshold", "center_is_minimum",
    "refine_center_hessian", "overlap_fraction", "preserve_grid", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Od|ppdp", kwlist,
        &sifGridType, &grid_obj, &radii_obj, &threshold, &center_is_min,
        &refine_hessian, &overlap_frac, &preserve_grid)) {
    return NULL;
  }

  PyArrayObject* radii_arr =
    (PyArrayObject*)PyArray_FROM_OTF(radii_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!radii_arr)
    return NULL;

  if (PyArray_NDIM(radii_arr) != 1 || PyArray_SIZE(radii_arr) == 0) {
    Py_DECREF(radii_arr);
    PyErr_SetString(PyExc_ValueError, "radii must be a non-empty 1D array");
    return NULL;
  }

  npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];
  real_t* radii_data = (real_t*)PyArray_DATA(radii_arr);
  sif_grid_t* c_grid = ((sifGridObject*)grid_obj)->grid;

  /* --- Parse Options to Bitmask --- */
  sif_option_t options = 0;

  if (center_is_min)
    options |= SIF_FINDER_CENTER_IS_MINIMUM;
  if (refine_hessian)
    options |= SIF_FINDER_REFINE_CENTER_HESSIAN;
  if (preserve_grid)
    options |= SIF_FINDER_PRESERVE_GRID;

  /* See the note in the rescaled wrapper: the GIL is released for the run. */
  sif_catalog_t* res_catalog = NULL;
  Py_BEGIN_ALLOW_THREADS
  res_catalog = sif_finder_spherical(c_grid, radii_data, (uint32_t)n_radii,
    (real_t)threshold, (real_t)overlap_frac, options);
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
