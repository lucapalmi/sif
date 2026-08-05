#include "py_size_function.h"

#include "structures/py_catalog.h"
#include "structures/py_size_function.h"
#include "sif/measure/sizefunction.h"
#include <numpy/arrayobject.h>

/* --- Functional API Implementation --- */

PyObject* py_sif_size_function_catalog(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* cat_obj;
  double box_length;
  uint32_t n_bins;
  const char* bin_str = "ln";
  double r_min = 0.0;
  double r_max = 0.0;

  static char* kwlist[] = {
    "catalog", "box_length", "n_bins", "bins", "r_min", "r_max", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dI|sdd", kwlist,
        &sifCatalogType, &cat_obj, &box_length, &n_bins, &bin_str, &r_min,
        &r_max)) {
    return NULL;
  }

  sif_option_t options = 0;

  if (strcmp(bin_str, "linear") == 0) {
    options |= SIF_VSF_BIN_LINEAR;
  } else if (strcmp(bin_str, "ln") != 0) {
    PyErr_SetString(
      PyExc_ValueError, "Invalid bin type. Expected 'ln' or 'linear'.");
    return NULL;
  }

  sifCatalogObject* cat = (sifCatalogObject*)cat_obj;

  sif_size_function_t* tmp = NULL;
  Py_BEGIN_ALLOW_THREADS
  tmp = sif_size_function_catalog(cat->catalog, (real_t)box_length, n_bins,
    options, (real_t)r_min, (real_t)r_max);
  Py_END_ALLOW_THREADS

  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to compute size function");
    return NULL;
  }

  sifSizeFunctionObject* obj =
    (sifSizeFunctionObject*)sifSizeFunctionType.tp_alloc(
      &sifSizeFunctionType, 0);
  if (!obj) {
    sif_size_function_free(tmp);
    return PyErr_NoMemory();
  }
  obj->vsf = tmp;

  return (PyObject*)obj;
}

/* --- NEW: Combination API Implementation --- */

PyObject* py_sif_size_function_combine(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* vsfs_seq;
  uint32_t master_bins;
  PyObject* domains_seq = Py_None;
  const char* method_str = "mean";
  const char* bin_str = "ln";

  static char* kwlist[] = {
    "vsfs", "master_bins", "domains", "method", "bin", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI|Oss", kwlist, &vsfs_seq,
        &master_bins, &domains_seq, &method_str, &bin_str)) {
    return NULL;
  }

  if (!PySequence_Check(vsfs_seq)) {
    PyErr_SetString(PyExc_TypeError,
      "vsfs must be a sequence (list or tuple) of SizeFunction objects.");
    return NULL;
  }

  Py_ssize_t n_vsfs = PySequence_Size(vsfs_seq);
  if (n_vsfs == 0) {
    PyErr_SetString(PyExc_ValueError, "vsfs sequence cannot be empty.");
    return NULL;
  }

  /* 1. Extract the underlying C VSF pointers */
  const sif_size_function_t** vsfs_arr =
    malloc(n_vsfs * sizeof(sif_size_function_t*));
  if (!vsfs_arr)
    return PyErr_NoMemory();

  for (Py_ssize_t i = 0; i < n_vsfs; i++) {
    PyObject* item = PySequence_GetItem(vsfs_seq, i);
    if (!item) {
      free((void*)vsfs_arr);
      return NULL;
    }
    if (!PyObject_TypeCheck(item, &sifSizeFunctionType)) {
      Py_DECREF(item);
      free((void*)vsfs_arr);
      PyErr_SetString(PyExc_TypeError,
        "All elements in vsfs must be pysif.measure.SizeFunction objects.");
      return NULL;
    }
    vsfs_arr[i] = ((sifSizeFunctionObject*)item)->vsf;
    Py_DECREF(item);
  }

  /* 2. Extract the Domain Interval Tuples (if provided) */
  sif_interval_t* domains_arr = NULL;
  if (domains_seq != Py_None) {
    if (!PySequence_Check(domains_seq) ||
        PySequence_Size(domains_seq) != n_vsfs) {
      free((void*)vsfs_arr);
      PyErr_SetString(PyExc_ValueError,
        "domains must be a sequence of the same length as vsfs.");
      return NULL;
    }

    domains_arr = malloc(n_vsfs * sizeof(sif_interval_t));
    if (!domains_arr) {
      free((void*)vsfs_arr);
      return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < n_vsfs; i++) {
      PyObject* tup = PySequence_GetItem(domains_seq, i);
      if (!PySequence_Check(tup) || PySequence_Size(tup) != 2) {
        Py_DECREF(tup);
        free((void*)vsfs_arr);
        free(domains_arr);
        PyErr_SetString(PyExc_ValueError,
          "Each domain must be a sequence of 2 floats: (min, max).");
        return NULL;
      }

      PyObject* pmin = PySequence_GetItem(tup, 0);
      PyObject* pmax = PySequence_GetItem(tup, 1);

      domains_arr[i].min = (real_t)PyFloat_AsDouble(pmin);
      domains_arr[i].max = (real_t)PyFloat_AsDouble(pmax);

      Py_DECREF(pmin);
      Py_DECREF(pmax);
      Py_DECREF(tup);

      if (PyErr_Occurred()) {
        free((void*)vsfs_arr);
        free(domains_arr);
        return NULL;
      }
    }
  }

  /* 3. Parse Options */
  sif_option_t options = 0;

  if (strcmp(method_str, "mean") == 0)
    options |= SIF_VSF_MERGE_MEAN;
  else if (strcmp(method_str, "median") == 0)
    options |= SIF_VSF_MERGE_MEDIAN;
  else if (strcmp(method_str, "stitch") == 0)
    options |= SIF_VSF_MERGE_STITCH;
  else {
    free((void*)vsfs_arr);
    if (domains_arr)
      free(domains_arr);
    PyErr_SetString(PyExc_ValueError,
      "Invalid method. Expected 'mean', 'median', or 'stitch'.");
    return NULL;
  }

  if (strcmp(bin_str, "linear") == 0)
    options |= SIF_VSF_BIN_LINEAR;
  else if (strcmp(bin_str, "ln") == 0)
    options |= SIF_VSF_BIN_LN;
  else {
    free((void*)vsfs_arr);
    if (domains_arr)
      free(domains_arr);
    PyErr_SetString(
      PyExc_ValueError, "Invalid bin type. Expected 'ln' or 'linear'.");
    return NULL;
  }

  /* 4. Execute C Engine */
  sif_size_function_t* tmp = sif_size_function_combine(
    vsfs_arr, (uint32_t)n_vsfs, master_bins, domains_arr, options);

  /* Cleanup temporary C arrays */
  free((void*)vsfs_arr);
  if (domains_arr)
    free(domains_arr);

  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError,
      "Failed to combine size functions. Check bounds and overlap.");
    return NULL;
  }

  /* 5. Wrap and Return */
  sifSizeFunctionObject* obj =
    (sifSizeFunctionObject*)sifSizeFunctionType.tp_alloc(
      &sifSizeFunctionType, 0);
  if (!obj) {
    sif_size_function_free(tmp);
    return PyErr_NoMemory();
  }
  obj->vsf = tmp;

  return (PyObject*)obj;
}
