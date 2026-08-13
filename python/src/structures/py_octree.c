/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_octree.h"
#include "py_field.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifOctree_dealloc(PyObject* self_obj) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  if (self->tree != NULL) {
    sif_octree_free(self->tree);
    self->tree = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifOctree_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  PyObject* field_obj;
  uint32_t max_per_leaf = 16;

  static char* kwlist[] = {"field", "max_per_leaf", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!|I", kwlist, &sifFieldType, &field_obj, &max_per_leaf)) {
    return -1;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  sif_octree_t* tmp = sif_octree_alloc(field->field, max_per_leaf);
  if (!tmp) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to build sif.octree");
    return -1;
  }

  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  self->tree = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifOctree_get_capacity(PyObject* self_obj, void* closure) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  return PyLong_FromUnsignedLong(self->tree->capacity);
}

static PyObject* sifOctree_get_count(PyObject* self_obj, void* closure) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  return PyLong_FromUnsignedLong(self->tree->count);
}

static PyGetSetDef sifOctree_getset[] = {
  {"capacity", sifOctree_get_capacity, NULL, "Capacity of the tree", NULL},
  {"count", sifOctree_get_count, NULL, "Number of nodes", NULL}, {NULL}};

/* --- Methods --- */

static PyObject* sifOctree_find_nearest(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  double px, py, pz;

  static char* kwlist[] = {"field", "px", "py", "pz", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!ddd", kwlist, &sifFieldType,
        &field_obj, &px, &py, &pz)) {
    return NULL;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t nearest = sif_octree_find_nearest(
    self->tree, field->field, (sif_real)px, (sif_real)py, (sif_real)pz);

  return PyLong_FromUnsignedLongLong(nearest);
}

static PyObject* sifOctree_search_radius(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  double px, py, pz, radius;
  uint64_t max_capacity = 1024;

  static char* kwlist[] = {
    "field", "px", "py", "pz", "radius", "max_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dddd|K", kwlist,
        &sifFieldType, &field_obj, &px, &py, &pz, &radius, &max_capacity)) {
    return NULL;
  }

  npy_intp dims[1] = {max_capacity};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_UINT64);
  if (!array)
    return NULL;
  uint64_t* data = (uint64_t*)PyArray_DATA((PyArrayObject*)array);

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t count =
    sif_octree_search_radius(self->tree, field->field, (sif_real)px,
      (sif_real)py, (sif_real)pz, (sif_real)radius, data, max_capacity);

  /* Resize the array to actual count if needed */
  dims[0] = count;
  PyArray_Dims new_dims = {dims, 1};
  PyArray_Resize((PyArrayObject*)array, &new_dims, 0, NPY_ANYORDER);

  return array;
}

static PyObject* sifOctree_search_box(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  double min_x, min_y, min_z, max_x, max_y, max_z;
  uint64_t max_capacity = 1024;

  static char* kwlist[] = {"field", "min_x", "min_y", "min_z", "max_x", "max_y",
    "max_z", "max_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dddddd|K", kwlist,
        &sifFieldType, &field_obj, &min_x, &min_y, &min_z, &max_x, &max_y,
        &max_z, &max_capacity)) {
    return NULL;
  }

  npy_intp dims[1] = {max_capacity};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_UINT64);
  if (!array)
    return NULL;
  uint64_t* data = (uint64_t*)PyArray_DATA((PyArrayObject*)array);

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t count = sif_octree_search_box(self->tree, field->field,
    (sif_real)min_x, (sif_real)min_y, (sif_real)min_z, (sif_real)max_x,
    (sif_real)max_y, (sif_real)max_z, data, max_capacity);

  dims[0] = count;
  PyArray_Dims new_dims = {dims, 1};
  PyArray_Resize((PyArrayObject*)array, &new_dims, 0, NPY_ANYORDER);

  return array;
}

static PyMethodDef sifOctree_methods[] = {
  {"find_nearest", (PyCFunction)sifOctree_find_nearest,
    METH_VARARGS | METH_KEYWORDS,
    "Find the nearest particle to given coordinates."},
  {"search_radius", (PyCFunction)sifOctree_search_radius,
    METH_VARARGS | METH_KEYWORDS, "Find all particles within a radius."},
  {"search_box", (PyCFunction)sifOctree_search_box,
    METH_VARARGS | METH_KEYWORDS,
    "Find all particles inside an axis-aligned box."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifOctreeType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.Octree", /* Updated */
  .tp_basicsize = sizeof(sifOctreeObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifOctree_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Octree(field, max_per_leaf=16)\n"
    "--\n\n"
    "An adaptive octree over a particle field.\n\n"
    "Subdivides only where particles are, so query cost tracks the\n"
    "local density rather than the box volume. That makes it the right\n"
    "structure for clustered data and the wrong one for a nearly\n"
    "uniform field, where ChainMesh is cheaper.\n\n"
    "Building the tree Morton-sorts the field, which physically\n"
    "reorders its arrays, and the tree stays valid only while the\n"
    "field is left alone.\n\n"
    "Args:\n"
    "    field: Field to index. Modified: it is sorted in place.\n"
    "    max_per_leaf: Particles a leaf may hold before it splits.",
  .tp_methods = sifOctree_methods,
  .tp_getset = sifOctree_getset,
  .tp_init = sifOctree_init,
  .tp_new = PyType_GenericNew,
};
