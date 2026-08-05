#include "py_delta_distribution.h"

#include "py_delta_common.h"
#include <numpy/arrayobject.h>

static void sifDeltaDistribution_dealloc(PyObject* self_obj) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (self->dist != NULL) {
    sif_delta_distribution_free(self->dist);
    self->dist = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifDeltaDistribution_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;

  static char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;

  self->dist = NULL;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifDeltaDistribution_get_n_radii(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->dist->n_radii);
}

static PyObject* sifDeltaDistribution_get_n_bins(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(self->dist->n_bins);
}

static PyObject* sifDeltaDistribution_get_n_samples(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(self->dist->n_samples);
}

static PyObject* sifDeltaDistribution_get_radii(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist || !self->dist->radii)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->dist->n_radii};
  return py_sif_wrap_borrowed(self_obj, 1, dims, self->dist->radii);
}

static PyObject* sifDeltaDistribution_get_delta_edges(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist || !self->dist->delta_edges)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->dist->n_bins + 1};
  return py_sif_wrap_borrowed(self_obj, 1, dims, self->dist->delta_edges);
}

static PyObject* sifDeltaDistribution_get_delta_centers(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist || !self->dist->delta_edges)
    Py_RETURN_NONE;

  /* Derived rather than borrowed, so this one is a genuine new array. */
  npy_intp dims[1] = {self->dist->n_bins};
  PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_REAL_T);
  if (!array)
    return NULL;

  real_t* out = (real_t*)PyArray_DATA(array);
  const real_t* edges = self->dist->delta_edges;
  for (uint32_t i = 0; i < self->dist->n_bins; i++) {
    out[i] = (real_t)(0.5 * ((double)edges[i] + (double)edges[i + 1]));
  }

  return (PyObject*)array;
}

static PyObject* sifDeltaDistribution_get_distributions(
  PyObject* self_obj, void* closure) {
  sifDeltaDistributionObject* self = (sifDeltaDistributionObject*)self_obj;
  if (!self->dist || !self->dist->distributions)
    Py_RETURN_NONE;

  npy_intp dims[2] = {self->dist->n_radii, self->dist->n_bins};
  return py_sif_wrap_borrowed(self_obj, 2, dims, self->dist->distributions);
}

static PyGetSetDef sifDeltaDistribution_getset[] = {
  {"n_radii", sifDeltaDistribution_get_n_radii, NULL,
    "Number of smoothing radii", NULL},
  {"n_bins", sifDeltaDistribution_get_n_bins, NULL, "Number of delta bins",
    NULL},
  {"n_samples", sifDeltaDistribution_get_n_samples, NULL,
    "Grid cells sampled per radius", NULL},
  {"radii", sifDeltaDistribution_get_radii, NULL,
    "1D array of smoothing radii", NULL},
  {"delta_edges", sifDeltaDistribution_get_delta_edges, NULL,
    "1D array of delta bin edges", NULL},
  {"delta_centers", sifDeltaDistribution_get_delta_centers, NULL,
    "1D array of delta bin centers", NULL},
  {"distributions", sifDeltaDistribution_get_distributions, NULL,
    "2D array of delta PDFs, one row per radius", NULL},
  {NULL}};

PyTypeObject sifDeltaDistributionType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.measure.DeltaDistribution",
  .tp_basicsize = sizeof(sifDeltaDistributionObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifDeltaDistribution_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "PDF of the smoothed density contrast, one row per radius.",
  .tp_getset = sifDeltaDistribution_getset,
  .tp_init = sifDeltaDistribution_init,
  .tp_new = PyType_GenericNew,
};

PyObject* py_sif_wrap_distribution(sif_delta_distribution_t* dist) {
  sifDeltaDistributionObject* obj =
    (sifDeltaDistributionObject*)sifDeltaDistributionType.tp_alloc(
      &sifDeltaDistributionType, 0);
  if (!obj) {
    sif_delta_distribution_free(dist);
    return PyErr_NoMemory();
  }
  obj->dist = dist;
  return (PyObject*)obj;
}
