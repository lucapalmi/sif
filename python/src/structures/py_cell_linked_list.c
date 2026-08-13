/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_cell_linked_list.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifCellLinkedList_dealloc(PyObject* self_obj) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  if (self->cll != NULL) {
    sif_cell_linked_list_free(self->cll);
    self->cll = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifCellLinkedList_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint32_t n_cells;
  /* "d" writes a full double, so this must not be a sif_real. */
  double box_length_in;
  unsigned long long initial_capacity = 1024;
  int periodic = 1;

  static char* kwlist[] = {
    "n_cells", "box_length", "initial_capacity", "periodic", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id|Kp", kwlist, &n_cells,
        &box_length_in, &initial_capacity, &periodic)) {
    return -1;
  }

  sif_option opt = periodic ? SIF_PBC_PERIODIC : SIF_PBC_OPEN;

  sif_cell_linked_list_t* tmp = sif_cell_linked_list_alloc(
    n_cells, (sif_real)box_length_in, (uint64_t)initial_capacity, opt);
  if (!tmp) {
    PyErr_SetString(PyExc_ValueError,
      "Failed to allocate sif.cell_linked_list (check n_cells, box_length and "
      "capacity)");
    return -1;
  }

  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  self->cll = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifCellLinkedList_get_n_cells(
  PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  return PyLong_FromUnsignedLong(self->cll->n_cells);
}

static PyObject* sifCellLinkedList_get_total_cells(
  PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->cll->total_cells);
}

static PyObject* sifCellLinkedList_get_inv_cell_length(
  PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  return PyFloat_FromDouble((double)self->cll->inv_cell_length);
}

static PyObject* sifCellLinkedList_get_capacity(
  PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->cll->capacity);
}

static PyObject* sifCellLinkedList_get_periodic(
  PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  return PyBool_FromLong((long)self->cll->periodic);
}

static PyObject* sifCellLinkedList_get_head(PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;

  npy_intp dims[1] = {self->cll->total_cells};

  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_INT32, self->cll->head);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);

  return array;
}

static PyObject* sifCellLinkedList_get_next(PyObject* self_obj, void* closure) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;

  npy_intp dims[1] = {self->cll->capacity};

  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_INT32, self->cll->next);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);

  return array;
}

static PyGetSetDef sifCellLinkedList_getset[] = {
  {"n_cells", sifCellLinkedList_get_n_cells, NULL,
    "Dimension of the coarse grid", NULL},
  {"total_cells", sifCellLinkedList_get_total_cells, NULL,
    "Total number of cells", NULL},
  {"inv_cell_length", sifCellLinkedList_get_inv_cell_length, NULL,
    "Inverse of the cell length", NULL},
  {"capacity", sifCellLinkedList_get_capacity, NULL,
    "Capacity of the item array", NULL},
  {"periodic", sifCellLinkedList_get_periodic, NULL,
    "True if out-of-box coordinates wrap, False if they clamp", NULL},
  {"head", sifCellLinkedList_get_head, NULL, "1D NumPy array of head indices",
    NULL},
  {"next", sifCellLinkedList_get_next, NULL, "1D NumPy array of next indices",
    NULL},
  {NULL}};

/* --- Methods --- */

static PyObject* sifCellLinkedList_ensure_capacity(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  unsigned long long required_capacity;

  static char* kwlist[] = {"required_capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "K", kwlist, &required_capacity)) {
    return NULL;
  }

  int status = sif_cell_linked_list_ensure_capacity(
    self->cll, (uint64_t)required_capacity);

  if (status == SIF_ERR_RANGE) {
    PyErr_SetString(
      PyExc_OverflowError, "required capacity exceeds the int32 item limit");
    return NULL;
  }
  if (status != SIF_OK) {
    PyErr_SetString(PyExc_MemoryError, "failed to grow the cell linked list");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* sifCellLinkedList_insert(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifCellLinkedListObject* self = (sifCellLinkedListObject*)self_obj;
  unsigned long long item_idx;
  double cx, cy, cz;

  static char* kwlist[] = {"item_idx", "cx", "cy", "cz", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "Kddd", kwlist, &item_idx, &cx, &cy, &cz)) {
    return NULL;
  }

  int status = sif_cell_linked_list_insert(
    self->cll, (uint64_t)item_idx, (sif_real)cx, (sif_real)cy, (sif_real)cz);

  if (status != SIF_OK) {
    PyErr_Format(PyExc_IndexError,
      "item_idx %llu is past the list capacity %llu", item_idx,
      (unsigned long long)self->cll->capacity);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef sifCellLinkedList_methods[] = {
  {"ensure_capacity", (PyCFunction)sifCellLinkedList_ensure_capacity,
    METH_VARARGS | METH_KEYWORDS, "Ensure capacity of the item array."},
  {"insert", (PyCFunction)sifCellLinkedList_insert,
    METH_VARARGS | METH_KEYWORDS, "Insert an item into the list."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifCellLinkedListType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.CellLinkedList", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifCellLinkedListObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifCellLinkedList_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "CellLinkedList(n_cells, box_length, initial_capacity=1, options=0)\n"
    "--\n\n"
    "A spatial bin built from linked lists, one per cell.\n\n"
    "Where ChainMesh sorts a fixed set of particles, this takes items\n"
    "one at a time and never moves an existing entry, which is what a\n"
    "finder needs while it is still accepting voids.\n\n"
    "Args:\n"
    "    n_cells: Cells per side.\n"
    "    box_length: Physical side length of the box.\n"
    "    initial_capacity: Items to make room for up front.\n"
    "    options: Boundary convention, SIF_PBC_PERIODIC (default) or\n"
    "        SIF_PBC_OPEN.",
  .tp_methods = sifCellLinkedList_methods,
  .tp_getset = sifCellLinkedList_getset,
  .tp_init = sifCellLinkedList_init,
  .tp_new = PyType_GenericNew,
};
