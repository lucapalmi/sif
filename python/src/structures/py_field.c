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

static PyObject* sifField_from_numpy(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  PyObject *xs_obj, *ys_obj, *zs_obj;
  PyObject *vxs_obj = NULL, *vys_obj = NULL, *vzs_obj = NULL;
  static char* kwlist[] = {"x", "y", "z", "vx", "vy", "vz", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|OOO", kwlist, &xs_obj,
        &ys_obj, &zs_obj, &vxs_obj, &vys_obj, &vzs_obj)) {
    return NULL;
  }

  /* Safely cast Python objects to contiguous NumPy arrays matching sif_real */
  PyArrayObject* xs_arr = (PyArrayObject*)PyArray_FROM_OTF(
    xs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* ys_arr = (PyArrayObject*)PyArray_FROM_OTF(
    ys_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* zs_arr = (PyArrayObject*)PyArray_FROM_OTF(
    zs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

  if (!xs_arr || !ys_arr || !zs_arr) {
    Py_XDECREF(xs_arr);
    Py_XDECREF(ys_arr);
    Py_XDECREF(zs_arr);
    PyErr_Format(PyExc_TypeError,
      "x, y, and z must be 1D contiguous %s NumPy arrays",
      sizeof(sif_real) == 8 ? "float64" : "float32");
    return NULL;
  }

  npy_intp n_particles = PyArray_SHAPE(xs_arr)[0];
  if (PyArray_NDIM(xs_arr) != 1 || PyArray_NDIM(ys_arr) != 1 ||
      PyArray_NDIM(zs_arr) != 1 || PyArray_SHAPE(ys_arr)[0] != n_particles ||
      PyArray_SHAPE(zs_arr)[0] != n_particles) {
    Py_DECREF(xs_arr);
    Py_DECREF(ys_arr);
    Py_DECREF(zs_arr);
    PyErr_SetString(PyExc_ValueError,
      "x, y, and z must be 1D arrays of the exact same length");
    return NULL;
  }

  /* Handle Optional Velocities */
  PyArrayObject *vxs_arr = NULL, *vys_arr = NULL, *vzs_arr = NULL;
  int has_velocities = (vxs_obj && vys_obj && vzs_obj);

  if (!has_velocities && (vxs_obj || vys_obj || vzs_obj)) {
    Py_DECREF(xs_arr);
    Py_DECREF(ys_arr);
    Py_DECREF(zs_arr);
    PyErr_SetString(PyExc_ValueError,
      "If providing velocities, vx, vy, and vz must all be provided");
    return NULL;
  }

  if (has_velocities) {
    vxs_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vxs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vys_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vys_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vzs_arr = (PyArrayObject*)PyArray_FROM_OTF(
      vzs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

    if (!vxs_arr || !vys_arr || !vzs_arr) {
      Py_XDECREF(xs_arr);
      Py_XDECREF(ys_arr);
      Py_XDECREF(zs_arr);
      Py_XDECREF(vxs_arr);
      Py_XDECREF(vys_arr);
      Py_XDECREF(vzs_arr);
      PyErr_SetString(
        PyExc_TypeError, "vx, vy, and vz must be 1D contiguous NumPy arrays");
      return NULL;
    }

    if (PyArray_NDIM(vxs_arr) != 1 || PyArray_NDIM(vys_arr) != 1 ||
        PyArray_NDIM(vzs_arr) != 1 ||
        PyArray_SHAPE(vxs_arr)[0] != n_particles ||
        PyArray_SHAPE(vys_arr)[0] != n_particles ||
        PyArray_SHAPE(vzs_arr)[0] != n_particles) {
      Py_DECREF(xs_arr);
      Py_DECREF(ys_arr);
      Py_DECREF(zs_arr);
      Py_DECREF(vxs_arr);
      Py_DECREF(vys_arr);
      Py_DECREF(vzs_arr);
      PyErr_SetString(PyExc_ValueError,
        "Velocity arrays must exactly match the length of the position arrays");
      return NULL;
    }
  }

  self->field->n_particles = (uint64_t)n_particles;

  const sif_real* x_data = (const sif_real*)PyArray_DATA(xs_arr);
  const sif_real* y_data = (const sif_real*)PyArray_DATA(ys_arr);
  const sif_real* z_data = (const sif_real*)PyArray_DATA(zs_arr);

  int status = sif_field_assign_positions(self->field, x_data, y_data, z_data);

  if (status == SIF_OK && has_velocities) {
    const sif_real* vx_data = (const sif_real*)PyArray_DATA(vxs_arr);
    const sif_real* vy_data = (const sif_real*)PyArray_DATA(vys_arr);
    const sif_real* vz_data = (const sif_real*)PyArray_DATA(vzs_arr);
    status =
      sif_field_assign_velocities(self->field, vx_data, vy_data, vz_data);
  }

  if (has_velocities) {
    Py_DECREF(vxs_arr);
    Py_DECREF(vys_arr);
    Py_DECREF(vzs_arr);
  }

  Py_DECREF(xs_arr);
  Py_DECREF(ys_arr);
  Py_DECREF(zs_arr);

  /* The copy is what makes the field independent of the caller's arrays, so a
   * failure here has to surface rather than leave a half-populated field. */
  if (status != SIF_OK) {
    PyErr_SetString(
      PyExc_MemoryError, "failed to copy particles into the field");
    return NULL;
  }

  Py_RETURN_NONE;
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

static PyGetSetDef sifField_getset[] = {
  {"n_particles", sifField_get_n_particles, NULL,
    "int: Number of particles the field holds.", NULL},
  {NULL}};

/* --- Method Definition Array --- */
static PyMethodDef sifField_methods[] = {
  {"from_numpy", (PyCFunction)sifField_from_numpy, METH_VARARGS | METH_KEYWORDS,
    "from_numpy(x, y, z, vx=None, vy=None, vz=None)\n"
    "--\n\n"
    "Copy positions, and optionally velocities, out of NumPy arrays.\n\n"
    "All arrays must have the same length and dtype pysif.real; anything\n"
    "else is converted, which costs a copy of the whole field. Velocities\n"
    "are optional but must be given together.\n\n"
    "Args:\n"
    "    x, y, z: Position components, one entry per particle.\n"
    "    vx, vy, vz: Velocity components, or None to store no velocities.\n\n"
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
