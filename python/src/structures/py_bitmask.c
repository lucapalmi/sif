/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_bitmask.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifBitmask_dealloc(PyObject* self_obj) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  if (self->bitmask != NULL) {
    sif_bitmask_free(self->bitmask);
    self->bitmask = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifBitmask_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint64_t n_bits;

  static char* kwlist[] = {"n_bits", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &n_bits)) {
    return -1;
  }

  sif_bitmask_t* tmp = sif_bitmask_alloc(n_bits);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sif.bitmask");
    return -1;
  }

  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  self->bitmask = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifBitmask_get_n_bits(PyObject* self_obj, void* closure) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->bitmask->n_bits);
}

static PyObject* sifBitmask_get_words(PyObject* self_obj, void* closure) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;

  npy_intp dims[1] = {(self->bitmask->n_bits + 63) / 64};

  /* Create an array referencing the underlying C memory (Zero-Copy) */
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_UINT64, self->bitmask->words);
  if (!array)
    return NULL;

  /* Tie the lifecycle of the array to the bitmask object */
  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);

  return array;
}

static PyGetSetDef sifBitmask_getset[] = {
  {"n_bits", sifBitmask_get_n_bits, NULL,
    "The total number of bits in the bitmask", NULL},
  {"words", sifBitmask_get_words, NULL,
    "1D NumPy array of underlying 64-bit words", NULL},
  {NULL}};

/* --- Methods --- */

static PyObject* sifBitmask_clear_all(PyObject* self_obj, PyObject* args) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  sif_bitmask_clear_all(self->bitmask);
  Py_RETURN_NONE;
}

static PyObject* sifBitmask_count_set(PyObject* self_obj, PyObject* args) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  uint64_t count = sif_bitmask_count_set(self->bitmask);
  return PyLong_FromUnsignedLongLong(count);
}

static PyObject* sifBitmask_set(PyObject* self_obj, PyObject* args) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  uint64_t ind;
  if (!PyArg_ParseTuple(args, "K", &ind)) {
    return NULL;
  }
  if (ind >= self->bitmask->n_bits) {
    PyErr_SetString(PyExc_IndexError, "Bit index out of range");
    return NULL;
  }
  sif_bitmask_set(self->bitmask, ind);
  Py_RETURN_NONE;
}

static PyObject* sifBitmask_unset(PyObject* self_obj, PyObject* args) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  uint64_t ind;
  if (!PyArg_ParseTuple(args, "K", &ind)) {
    return NULL;
  }
  if (ind >= self->bitmask->n_bits) {
    PyErr_SetString(PyExc_IndexError, "Bit index out of range");
    return NULL;
  }
  sif_bitmask_unset(self->bitmask, ind);
  Py_RETURN_NONE;
}

static PyObject* sifBitmask_get(PyObject* self_obj, PyObject* args) {
  sifBitmaskObject* self = (sifBitmaskObject*)self_obj;
  uint64_t ind;
  if (!PyArg_ParseTuple(args, "K", &ind)) {
    return NULL;
  }
  if (ind >= self->bitmask->n_bits) {
    PyErr_SetString(PyExc_IndexError, "Bit index out of range");
    return NULL;
  }
  uint8_t val = sif_bitmask_get(self->bitmask, ind);
  return PyBool_FromLong(val);
}

static PyMethodDef sifBitmask_methods[] = {
  {"clear_all", (PyCFunction)sifBitmask_clear_all, METH_NOARGS,
    "Clear all bits."},
  {"count_set", (PyCFunction)sifBitmask_count_set, METH_NOARGS,
    "Count the number of set bits."},
  {"set", (PyCFunction)sifBitmask_set, METH_VARARGS,
    "Set a bit at given index."},
  {"unset", (PyCFunction)sifBitmask_unset, METH_VARARGS,
    "Unset a bit at given index."},
  {"get", (PyCFunction)sifBitmask_get, METH_VARARGS,
    "Get value of bit at given index."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifBitmaskType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.Bitmask", /* Updated */
  .tp_basicsize = sizeof(sifBitmaskObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifBitmask_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Bitmask(n_bits)\n"
    "--\n\n"
    "A dense array of bits, packed 64 to a word.\n\n"
    "Used where one boolean per grid cell would otherwise cost a byte:\n"
    "at 1024**3 cells that is 128 MiB rather than 1 GiB.\n\n"
    "Args:\n"
    "    n_bits: Number of bits. Must be non-zero.",
  .tp_methods = sifBitmask_methods,
  .tp_getset = sifBitmask_getset,
  .tp_init = sifBitmask_init,
  .tp_new = PyType_GenericNew,
};
