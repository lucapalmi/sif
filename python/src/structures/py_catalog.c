#include "py_catalog.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifCatalog_dealloc(PyObject* self_obj) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  if (self->catalog != NULL) {
    sif_catalog_free(self->catalog);
    self->catalog = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifCatalog_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint64_t initial_capacity = 1024; // Sensible default

  static char* kwlist[] = {"capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|K", kwlist, &initial_capacity)) {
    return -1;
  }

  sif_catalog_t* tmp = sif_catalog_alloc(initial_capacity);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sif.catalog");
    return -1;
  }

  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  self->catalog = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifCatalog_get_centers(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;

  npy_intp dims[2] = {self->catalog->n_voids, 3};
  PyObject* array = PyArray_SimpleNew(2, dims, NPY_REAL_T);
  if (!array)
    return NULL;

  real_t* data = (real_t*)PyArray_DATA((PyArrayObject*)array);

  /* Interleave the C Struct-of-Arrays into a Python Nx3 array */
  for (uint64_t i = 0; i < self->catalog->n_voids; i++) {
    data[i * 3 + 0] = self->catalog->cx[i];
    data[i * 3 + 1] = self->catalog->cy[i];
    data[i * 3 + 2] = self->catalog->cz[i];
  }

  return array;
}

static PyObject* sifCatalog_get_radii(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;

  npy_intp dims[1] = {self->catalog->n_voids};

  /* Create an array referencing the underlying C memory (Zero-Copy) */
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->catalog->radii);
  if (!array)
    return NULL;

  /* Tie the lifecycle of the array to the catalog object */
  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);

  return array;
}

static PyObject* sifCatalog_get_n_voids(PyObject* self_obj, void* closure) {
  sifCatalogObject* self = (sifCatalogObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->catalog->n_voids);
}

static PyGetSetDef sifCatalog_getset[] = {
  {"centers", sifCatalog_get_centers, NULL,
    "Nx3 NumPy array of void centers (x, y, z)", NULL},
  {"radii", sifCatalog_get_radii, NULL, "1D NumPy array of void radii", NULL},
  {"n_voids", sifCatalog_get_n_voids, NULL,
    "The number of voids in the catalog", NULL},
  {NULL}};

/* --- Methods --- */

static PyMethodDef sifCatalog_methods[] = {
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifCatalogType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pysif.structures.Catalog", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifCatalogObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifCatalog_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF void catalog object.",
  .tp_methods = sifCatalog_methods,
  .tp_getset = sifCatalog_getset,
  .tp_init = sifCatalog_init,
  .tp_new = PyType_GenericNew,
};