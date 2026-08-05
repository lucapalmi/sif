#include "py_tessellation.h"
#include "py_field.h"
#include <numpy/arrayobject.h>
#include <string.h>

/* --- Lifecycle Methods --- */

static void sifTessellation_dealloc(PyObject* self_obj) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (self->tess != NULL) {
    sif_tessellation_free(self->tess);
    self->tess = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifTessellation_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  self->tess = NULL;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifTessellation_get_num_particles(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(self->tess->num_particles);
}

static PyObject* sifTessellation_get_num_edges(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(self->tess->num_edges);
}

static PyObject* sifTessellation_get_volumes(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess || !self->tess->volumes)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->tess->num_particles};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->tess->volumes);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifTessellation_get_neighbor_offsets(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess || !self->tess->neighbor_offsets)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->tess->num_particles + 1};
  PyObject* array = PyArray_SimpleNewFromData(
    1, dims, NPY_UINT64, self->tess->neighbor_offsets);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifTessellation_get_neighbor_indices(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess || !self->tess->neighbor_indices)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->tess->num_edges};
  PyObject* array = PyArray_SimpleNewFromData(
    1, dims, NPY_UINT64, self->tess->neighbor_indices);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyGetSetDef sifTessellation_getset[] = {
  {"num_particles", sifTessellation_get_num_particles, NULL,
    "Number of particles", NULL},
  {"num_edges", sifTessellation_get_num_edges, NULL,
    "Number of total graph edges", NULL},
  {"volumes", sifTessellation_get_volumes, NULL,
    "Contiguous array of particle volumes", NULL},
  {"neighbor_offsets", sifTessellation_get_neighbor_offsets, NULL,
    "CSR offsets for neighbors", NULL},
  {"neighbor_indices", sifTessellation_get_neighbor_indices, NULL,
    "CSR indices for neighbors", NULL},
  {NULL}};

/* --- The Free Function (Dispatcher) --- */

PyObject* py_sif_tessellation_build(
  PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* field_obj;
  char* method = "random";
  uint32_t supersample_factor = 10;
  uint32_t options = 0;

  static char* kwlist[] = {
    "field", "method", "supersample_factor", "options", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|sII", kwlist, &sifFieldType,
        &field_obj, &method, &supersample_factor, &options)) {
    return NULL;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  sif_tessellation_t* tmp_tess = NULL;

  /* String Dispatch Logic */
  if (strcmp(method, "random") == 0) {
    options |= SIF_TESS_METHOD_RANDOM;
    tmp_tess =
      sif_tessellation_build_approx(field->field, supersample_factor, options);
  } else if (strcmp(method, "voxel") == 0) {
    options |= SIF_TESS_METHOD_VOXEL;
    tmp_tess =
      sif_tessellation_build_approx(field->field, supersample_factor, options);
  } else {
    PyErr_SetString(PyExc_ValueError,
      "Invalid method. Choose 'random' or 'voxel'. For exact "
      "geometry, use the dedicated builder.");
    return NULL;
  }

  if (!tmp_tess) {
    PyErr_SetString(
      PyExc_RuntimeError, "Failed to build tessellation. Check C logs.");
    return NULL;
  }

  /* Allocate and wrap the Python object */
  sifTessellationObject* obj =
    (sifTessellationObject*)sifTessellationType.tp_alloc(
      &sifTessellationType, 0);
  if (!obj) {
    sif_tessellation_free(tmp_tess);
    return PyErr_NoMemory();
  }

  obj->tess = tmp_tess;
  return (PyObject*)obj;
}

/* --- Type Object --- */
PyTypeObject sifTessellationType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.structures.Tessellation",
  .tp_basicsize = sizeof(sifTessellationObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifTessellation_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF Generalized volume and topology tessellation (CSR Graph).",
  .tp_getset = sifTessellation_getset,
  .tp_init = sifTessellation_init,
  .tp_new = PyType_GenericNew,
};
