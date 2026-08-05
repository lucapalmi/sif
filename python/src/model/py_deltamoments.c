#include "py_model.h"

#include "py_delta_common.h"
#include "structures/py_delta_moments.h"
#include "sif/model/deltamoments.h"
#include <numpy/arrayobject.h>

PyObject* py_sif_delta_moments_pk(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* k_obj;
  PyObject* pk_obj;
  PyObject* radii_obj;
  unsigned char order = 2;

  const char* window = NULL;

  static char* kwlist[] = {"k", "pk", "radii", "order", "window", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|bz", kwlist, &k_obj,
        &pk_obj, &radii_obj, &order, &window)) {
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
  const npy_intp n_radii = PyArray_SHAPE(radii_arr)[0];

  sif_delta_moments_t* tmp = NULL;

  Py_BEGIN_ALLOW_THREADS tmp = sif_delta_moments_pk(
    (const real_t*)PyArray_DATA(k_arr), (const real_t*)PyArray_DATA(pk_arr),
    (uint32_t)n_points, (const real_t*)PyArray_DATA(radii_arr),
    (uint32_t)n_radii, (uint8_t)order, options);
  Py_END_ALLOW_THREADS

    Py_DECREF(k_arr);
  Py_DECREF(pk_arr);
  Py_DECREF(radii_arr);

  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError,
      "Failed to evaluate the moments from P(k); see the sif log for the "
      "cause.");
    return NULL;
  }

  return py_sif_wrap_moments(tmp);
}
