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
  uint32_t initial_capacity = 1024;

  static char* kwlist[] = {"initial_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|I", kwlist, &initial_capacity)) {
    return -1;
  }

  sif_octree_t* tmp = sif_octree_alloc(initial_capacity);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sif.octree");
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
  {"count", sifOctree_get_count, NULL, "Number of nodes", NULL},
  {NULL}};

/* --- Methods --- */

static PyObject* sifOctree_build(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  uint32_t max_per_leaf = 16;

  static char* kwlist[] = {"field", "max_per_leaf", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|I", kwlist, &sifFieldType, &field_obj, &max_per_leaf)) {
    return NULL;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  if (sif_octree_build(self->tree, field->field, max_per_leaf) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to build octree");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* sifOctree_find_nearest(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  double px, py, pz;

  static char* kwlist[] = {"field", "px", "py", "pz", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!ddd", kwlist, &sifFieldType, &field_obj, &px, &py, &pz)) {
    return NULL;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t nearest = sif_octree_find_nearest(self->tree, field->field, (real_t)px, (real_t)py, (real_t)pz);

  return PyLong_FromUnsignedLongLong(nearest);
}

static PyObject* sifOctree_search_radius(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifOctreeObject* self = (sifOctreeObject*)self_obj;
  PyObject* field_obj;
  double px, py, pz, radius;
  uint64_t max_capacity = 1024;

  static char* kwlist[] = {"field", "px", "py", "pz", "radius", "max_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dddd|K", kwlist, &sifFieldType, &field_obj, &px, &py, &pz, &radius, &max_capacity)) {
    return NULL;
  }

  npy_intp dims[1] = {max_capacity};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_UINT64);
  if (!array) return NULL;
  uint64_t* data = (uint64_t*)PyArray_DATA((PyArrayObject*)array);

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t count = sif_octree_search_radius(self->tree, field->field, (real_t)px, (real_t)py, (real_t)pz, (real_t)radius, data, max_capacity);

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

  static char* kwlist[] = {"field", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "max_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dddddd|K", kwlist, &sifFieldType, &field_obj, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z, &max_capacity)) {
    return NULL;
  }

  npy_intp dims[1] = {max_capacity};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_UINT64);
  if (!array) return NULL;
  uint64_t* data = (uint64_t*)PyArray_DATA((PyArrayObject*)array);

  sifFieldObject* field = (sifFieldObject*)field_obj;
  uint64_t count = sif_octree_search_box(self->tree, field->field, (real_t)min_x, (real_t)min_y, (real_t)min_z, (real_t)max_x, (real_t)max_y, (real_t)max_z, data, max_capacity);

  dims[0] = count;
  PyArray_Dims new_dims = {dims, 1};
  PyArray_Resize((PyArrayObject*)array, &new_dims, 0, NPY_ANYORDER);

  return array;
}

static PyMethodDef sifOctree_methods[] = {
  {"build", (PyCFunction)sifOctree_build, METH_VARARGS | METH_KEYWORDS, "Build an octree from a field."},
  {"find_nearest", (PyCFunction)sifOctree_find_nearest, METH_VARARGS | METH_KEYWORDS, "Find the nearest particle to given coordinates."},
  {"search_radius", (PyCFunction)sifOctree_search_radius, METH_VARARGS | METH_KEYWORDS, "Find all particles within a radius."},
  {"search_box", (PyCFunction)sifOctree_search_box, METH_VARARGS | METH_KEYWORDS, "Find all particles inside an axis-aligned box."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifOctreeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pysif.structures.Octree", /* Updated */
  .tp_basicsize = sizeof(sifOctreeObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifOctree_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF octree object.",
  .tp_methods = sifOctree_methods,
  .tp_getset = sifOctree_getset,
  .tp_init = sifOctree_init,
  .tp_new = PyType_GenericNew,
};
