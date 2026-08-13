/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

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
  PyObject* field_obj;
  char* method = "random";
  uint32_t supersample_factor = 10;
  uint32_t options = 0;

  static char* kwlist[] = {
    "field", "method", "supersample_factor", "options", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|sII", kwlist, &sifFieldType,
        &field_obj, &method, &supersample_factor, &options)) {
    return -1;
  }

  if (strcmp(method, "random") == 0) {
    options |= SIF_TESS_METHOD_RANDOM;
  } else if (strcmp(method, "voxel") == 0) {
    options |= SIF_TESS_METHOD_VOXEL;
  } else {
    PyErr_SetString(
      PyExc_ValueError, "method must be either 'random' or 'voxel'");
    return -1;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  sif_tessellation_t* tmp = NULL;

  Py_BEGIN_ALLOW_THREADS tmp =
    sif_tessellation_approx(field->field, supersample_factor, options);
  Py_END_ALLOW_THREADS

    if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError,
      "failed to build the tessellation; see the sif log for details");
    return -1;
  }

  if (self->tess)
    sif_tessellation_free(self->tess);
  self->tess = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifTessellation_get_num_particles(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(self->tess->n_particles);
}

static PyObject* sifTessellation_get_num_edges(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(self->tess->n_edges);
}

static PyObject* sifTessellation_get_volumes(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess || !self->tess->volumes)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->tess->n_particles};
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

  npy_intp dims[1] = {self->tess->n_particles + 1};
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

  npy_intp dims[1] = {self->tess->n_edges};
  PyObject* array = PyArray_SimpleNewFromData(
    1, dims, NPY_UINT64, self->tess->neighbor_indices);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyGetSetDef sifTessellation_getset[] = {
  {"n_particles", sifTessellation_get_num_particles, NULL,
    "Number of particles", NULL},
  {"n_edges", sifTessellation_get_num_edges, NULL,
    "Number of total graph edges", NULL},
  {"volumes", sifTessellation_get_volumes, NULL,
    "Contiguous array of particle volumes", NULL},
  {"neighbor_offsets", sifTessellation_get_neighbor_offsets, NULL,
    "CSR offsets for neighbors", NULL},
  {"neighbor_indices", sifTessellation_get_neighbor_indices, NULL,
    "CSR indices for neighbors", NULL},
  {NULL}};

/* --- The Free Function (Dispatcher) --- */

/* --- Type Object --- */
PyTypeObject sifTessellationType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.Tessellation",
  .tp_basicsize = sizeof(sifTessellationObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifTessellation_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Tessellation(field, method='random', supersample_factor=10, options=0)\n"
    "--\n\n"
    "Per-particle volumes and adjacency, as a sparse graph.\n\n"
    "Gives every tracer a volume, and so a density estimate that\n"
    "adapts to the local sampling instead of to a fixed grid -- which\n"
    "is what a void finder wants, since voids are where a fixed grid\n"
    "has fewest tracers per cell.\n\n"
    "Approximate: the volumes come from sampling rather than from\n"
    "constructing the cells, trading a controllable error for a cost\n"
    "linear in the particle count.\n\n"
    "Args:\n"
    "    field: Particle field to tessellate.\n"
    "    method: 'random' or 'voxel', for how sample points are\n"
    "        placed.\n"
    "    supersample_factor: Sample points per real particle. Higher\n"
    "        is more accurate and proportionally slower.\n"
    "    options: Extra option bits.",
  .tp_getset = sifTessellation_getset,
  .tp_init = sifTessellation_init,
  .tp_new = PyType_GenericNew,
};
