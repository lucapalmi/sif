/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_field.h"
#include <numpy/arrayobject.h>

static void sifField_dealloc(PyObject* self_obj) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  if (self->field != NULL) {
    sif_field_free(self->field);
    self->field = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifField_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint64_t initial_capacity = 0;

  static char* kwlist[] = {"capacity", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|K", kwlist, &initial_capacity)) {
    return -1;
  }

  sif_field_t* tmp = sif_field_alloc(initial_capacity);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "failed to allocate sif.field");
    return -1;
  }

  sifFieldObject* self = (sifFieldObject*)self_obj;
  self->field = tmp;
  return 0;
}

/*
 * One per-particle column: a contiguous 1D sif_real array of length `n`, or,
 * for an optional column passed as None or not at all, NULL. `n` < 0 takes
 * the length from this array instead, which is how the first column sets it
 * for the rest.
 *
 * @return 0 with *out set (a new reference, or NULL), -1 with an exception.
 */
static int column_from_numpy(
  PyObject* obj, const char* name, npy_intp n, PyArrayObject** out) {

  *out = NULL;
  if (!obj || obj == Py_None)
    return 0;

  PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(
    obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!arr) {
    PyErr_Format(PyExc_TypeError, "%s must be convertible to a %s array", name,
      sizeof(sif_real) == 8 ? "float64" : "float32");
    return -1;
  }

  if (PyArray_NDIM(arr) != 1 || (n >= 0 && PyArray_SHAPE(arr)[0] != n)) {
    PyErr_Format(PyExc_ValueError,
      "%s must be a 1D array with one entry per particle", name);
    Py_DECREF(arr);
    return -1;
  }

  *out = arr;
  return 0;
}

static PyObject* sifField_from_numpy(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  PyObject *xs_obj, *ys_obj, *zs_obj;
  PyObject *vxs_obj = NULL, *vys_obj = NULL, *vzs_obj = NULL;
  PyObject* ws_obj = NULL;
  static char* kwlist[] = {"x", "y", "z", "vx", "vy", "vz", "weights", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|OOOO", kwlist, &xs_obj,
        &ys_obj, &zs_obj, &vxs_obj, &vys_obj, &vzs_obj, &ws_obj)) {
    return NULL;
  }

  /* None and absent mean the same thing for every optional column. */
  const int n_vel = (vxs_obj && vxs_obj != Py_None) +
                    (vys_obj && vys_obj != Py_None) +
                    (vzs_obj && vzs_obj != Py_None);
  if (n_vel != 0 && n_vel != 3) {
    PyErr_SetString(PyExc_ValueError,
      "If providing velocities, vx, vy, and vz must all be provided");
    return NULL;
  }

  if (xs_obj == Py_None || ys_obj == Py_None || zs_obj == Py_None) {
    PyErr_SetString(PyExc_ValueError, "x, y and z are required");
    return NULL;
  }

  /* Every column is held here until the field is built; x sets the length the
   * others are checked against. */
  enum { X, Y, Z, VX, VY, VZ, W, N_COLS };
  PyArrayObject* cols[N_COLS] = {NULL};
  sif_field_t* fresh = NULL;
  int status = SIF_OK;

  if (column_from_numpy(xs_obj, "x", -1, &cols[X]) < 0)
    goto fail;

  const npy_intp n = PyArray_SHAPE(cols[X])[0];

  if (column_from_numpy(ys_obj, "y", n, &cols[Y]) < 0 ||
      column_from_numpy(zs_obj, "z", n, &cols[Z]) < 0 ||
      column_from_numpy(vxs_obj, "vx", n, &cols[VX]) < 0 ||
      column_from_numpy(vys_obj, "vy", n, &cols[VY]) < 0 ||
      column_from_numpy(vzs_obj, "vz", n, &cols[VZ]) < 0 ||
      column_from_numpy(ws_obj, "weights", n, &cols[W]) < 0)
    goto fail;

  /*
   * Built as a new field and swapped in only once it is complete, rather than
   * assigned into the existing one. Assigning in place would leave behind any
   * column this call does not supply -- velocities or weights from a previous
   * from_numpy, sized for a different particle count and describing different
   * particles -- and a failure half-way would leave a field that is neither.
   */
  fresh = sif_field_alloc((uint64_t)n);
  if (!fresh) {
    PyErr_SetString(PyExc_MemoryError, "failed to allocate the field");
    goto fail;
  }

#define COL(c) ((const sif_real*)PyArray_DATA(cols[c]))
  status = sif_field_assign_positions(fresh, COL(X), COL(Y), COL(Z));
  if (status == SIF_OK && cols[VX])
    status = sif_field_assign_velocities(fresh, COL(VX), COL(VY), COL(VZ));
  if (status == SIF_OK && cols[W])
    status = sif_field_assign_weights(fresh, COL(W));
#undef COL

  if (status != SIF_OK) {
    PyErr_SetString(
      PyExc_MemoryError, "failed to copy particles into the field");
    goto fail;
  }

  for (int c = 0; c < N_COLS; c++)
    Py_XDECREF(cols[c]);

  sif_field_free(self->field);
  self->field = fresh;

  Py_RETURN_NONE;

fail:
  for (int c = 0; c < N_COLS; c++)
    Py_XDECREF(cols[c]);
  sif_field_free(fresh);
  return NULL;
}

static PyObject* sifField_wrap(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  double box_length;
  static char* kwlist[] = {"box_length", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &box_length)) {
    return NULL;
  }

  uint64_t boundary = 0, wrapped = 0;
  int status = SIF_OK;

  /* One pass over every coordinate: worth dropping the GIL at the particle
   * counts this is meant for. */
  Py_BEGIN_ALLOW_THREADS status = sif_field_wrap_periodic(
    self->field, (sif_real)box_length, &boundary, &wrapped);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    PyErr_SetString(PyExc_ValueError,
      "failed to wrap the field: it must hold positions and box_length must "
      "be positive");
    return NULL;
  }

  return Py_BuildValue("{s:K,s:K}", "boundary", (unsigned long long)boundary,
    "wrapped", (unsigned long long)wrapped);
}

static PyObject* sifField_sort_morton(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  sif_field_sort_morton(self->field);
  Py_RETURN_NONE;
}

static PyObject* sifField_refresh_bounds(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  sif_field_refresh_bounds(self->field);
  Py_RETURN_NONE;
}

/* --- Properties (Getters) --- */

static PyObject* sifField_get_n_particles(PyObject* self_obj, void* closure) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->field->n_particles);
}

static PyObject* sifField_get_has_weights(PyObject* self_obj, void* closure) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  return PyBool_FromLong(self->field->weights != NULL);
}

static PyObject* sifField_get_has_velocities(
  PyObject* self_obj, void* closure) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  return PyBool_FromLong(self->field->vx != NULL);
}

static PyGetSetDef sifField_getset[] = {
  {"n_particles", sifField_get_n_particles, NULL,
    "int: Number of particles the field holds.", NULL},
  {"has_weights", sifField_get_has_weights, NULL,
    "bool: Whether the field carries per-particle weights. Everything built\n"
    "from a weighted field -- grids, meshes, the finders, the profiles --\n"
    "uses them.",
    NULL},
  {"has_velocities", sifField_get_has_velocities, NULL,
    "bool: Whether the field carries velocities.", NULL},
  {NULL}};

/* --- Method Definition Array --- */
static PyMethodDef sifField_methods[] = {
  {"from_numpy", (PyCFunction)sifField_from_numpy, METH_VARARGS | METH_KEYWORDS,
    "from_numpy(x, y, z, vx=None, vy=None, vz=None, weights=None)\n"
    "--\n\n"
    "Copy positions, and optionally velocities and weights, out of NumPy\n"
    "arrays.\n\n"
    "All arrays must have the same length and dtype pysif.real; anything\n"
    "else is converted, which costs a copy of the whole field. Velocities\n"
    "are optional but must be given together.\n\n"
    "Replaces whatever the field held before, including any velocities or\n"
    "weights this call does not supply.\n\n"
    "Args:\n"
    "    x, y, z: Position components, one entry per particle.\n"
    "    vx, vy, vz: Velocity components, or None to store no velocities.\n"
    "    weights: Per-particle weight (a mass, a luminosity, a selection\n"
    "        weight), or None for an unweighted field, where every particle\n"
    "        counts as 1. The exodus finder requires them to be finite and\n"
    "        non-negative.\n\n"
    "Raises:\n"
    "    ValueError: If the arrays disagree in length.\n"
    "    MemoryError: If the field's buffers could not be allocated."},
  {"wrap", (PyCFunction)sifField_wrap, METH_VARARGS | METH_KEYWORDS,
    "wrap(box_length) -> dict\n\n"
    "Fold every coordinate into [0, box_length) periodically, in place.\n"
    "Returns {'boundary': n, 'wrapped': n}: 'boundary' counts coordinates\n"
    "that sat exactly on the box edge, which is the single-precision\n"
    "rounding artifact this exists for, and 'wrapped' counts coordinates\n"
    "that were genuinely outside. A large 'wrapped' means the box length is\n"
    "wrong and the folded field is meaningless -- check it rather than\n"
    "proceeding. Only correct for a field that is periodic in this box."},
  {"sort_morton", (PyCFunction)sifField_sort_morton, METH_NOARGS,
    "sort_morton()\n"
    "--\n\n"
    "Reorder the particles along a 3D Morton curve, in place.\n\n"
    "Puts particles that are close in space close in memory, which is what\n"
    "the octree requires and what makes the neighbour queries fast. The\n"
    "permutation is kept, so velocities and weights assigned afterwards are\n"
    "matched to their own particles automatically."},
  {"refresh_bounds", (PyCFunction)sifField_refresh_bounds, METH_NOARGS,
    "refresh_bounds()\n"
    "--\n\n"
    "Recompute the cached bounding box and centre.\n\n"
    "Only needed after writing into the position arrays directly; every\n"
    "operation that goes through this object keeps the bounds current."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */
PyTypeObject sifFieldType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.Field", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifFieldObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifField_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Field(capacity=0)\n"
    "--\n\n"
    "A particle field: positions, and optionally velocities and\n"
    "weights.\n\n"
    "The container every other structure is built from. Fill it with\n"
    "from_numpy(), or read one off disk with pysif.io.read_field().\n\n"
    "Args:\n"
    "    capacity: Particles to make room for up front. The arrays are\n"
    "        sized by from_numpy() anyway, so this only avoids a\n"
    "        reallocation.",
  .tp_methods = sifField_methods,
  .tp_getset = sifField_getset,
  .tp_init = sifField_init,
  .tp_new = PyType_GenericNew,
};
