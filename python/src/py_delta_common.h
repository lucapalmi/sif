#ifndef __SIF_PY_DELTA_COMMON_H__
#define __SIF_PY_DELTA_COMMON_H__

/*
 * Helpers shared by the measure-side and model-side delta bindings. Both
 * produce the same container types, so the option parsing and the array
 * plumbing have to agree between them.
 */

#include "py_common.h"
#include "sif/utils/align.h"
#include <string.h>

/*
 * @brief Maps the shuffle/window strings onto the options bitmask.
 *
 * @return 0, or -1 with a Python exception set on an unrecognized name.
 */
static inline int py_sif_delta_parse_options(const char* shuffle,
  const char* window, int keep_cic_window, sif_option_t* out) {

  sif_option_t options = SIF_DEFAULT;

  if (shuffle == NULL || strcmp(shuffle, "none") == 0) {
    options |= SIF_DELTA_SHUFFLE_NONE;
  } else if (strcmp(shuffle, "phases") == 0) {
    options |= SIF_DELTA_SHUFFLE_PHASES;
  } else if (strcmp(shuffle, "gaussian") == 0) {
    options |= SIF_DELTA_SHUFFLE_GAUSSIAN;
  } else {
    PyErr_Format(PyExc_ValueError,
      "shuffle must be None, 'none', 'phases' or 'gaussian', not '%s'",
      shuffle);
    return -1;
  }

  if (window == NULL || strcmp(window, "top_hat") == 0) {
    options |= SIF_DELTA_FILTER_TOP_HAT;
  } else if (strcmp(window, "gaussian") == 0) {
    options |= SIF_DELTA_FILTER_GAUSSIAN;
  } else {
    PyErr_Format(PyExc_ValueError,
      "window must be None, 'top_hat' or 'gaussian', not '%s'", window);
    return -1;
  }

  if (keep_cic_window)
    options |= SIF_DELTA_KEEP_CIC_WINDOW;

  *out = options;
  return 0;
}

/*
 * @brief Converts a 1D sequence to a contiguous real_t array.
 *
 * @return A new reference, or NULL with a Python exception set.
 */
static inline PyArrayObject* py_sif_as_real_array(
  PyObject* obj, const char* name) {

  PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(
    obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
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

/*
 * @brief Copies a C-owned real_t array into a fresh NumPy array and releases
 * the original.
 *
 * Takes ownership of `values` on every path, including failure.
 *
 * @return A new reference, or NULL with a Python exception set.
 */
static inline PyObject* py_sif_owned_array(real_t* values, npy_intp n) {
  if (!values)
    return NULL;

  PyObject* array = PyArray_SimpleNew(1, &n, NPY_REAL_T);
  if (!array) {
    sif_free_aligned(values);
    return NULL;
  }

  memcpy(PyArray_DATA((PyArrayObject*)array), values,
    (size_t)n * sizeof(real_t));
  sif_free_aligned(values);

  return array;
}

#endif /* __SIF_PY_DELTA_COMMON_H__ */
