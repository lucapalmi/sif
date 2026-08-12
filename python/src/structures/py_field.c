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

  /* Safely cast Python objects to contiguous NumPy arrays matching real_t */
  PyArrayObject* xs_arr =
    (PyArrayObject*)PyArray_FROM_OTF(xs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* ys_arr =
    (PyArrayObject*)PyArray_FROM_OTF(ys_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* zs_arr =
    (PyArrayObject*)PyArray_FROM_OTF(zs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

  if (!xs_arr || !ys_arr || !zs_arr) {
    Py_XDECREF(xs_arr);
    Py_XDECREF(ys_arr);
    Py_XDECREF(zs_arr);
    PyErr_Format(PyExc_TypeError,
      "x, y, and z must be 1D contiguous %s NumPy arrays",
      sizeof(real_t) == 8 ? "float64" : "float32");
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
    vxs_arr =
      (PyArrayObject*)PyArray_FROM_OTF(vxs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vys_arr =
      (PyArrayObject*)PyArray_FROM_OTF(vys_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vzs_arr =
      (PyArrayObject*)PyArray_FROM_OTF(vzs_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

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

  const real_t* x_data = (const real_t*)PyArray_DATA(xs_arr);
  const real_t* y_data = (const real_t*)PyArray_DATA(ys_arr);
  const real_t* z_data = (const real_t*)PyArray_DATA(zs_arr);

  int status = sif_field_assign_positions(self->field, x_data, y_data, z_data);

  if (status == SIF_OK && has_velocities) {
    const real_t* vx_data = (const real_t*)PyArray_DATA(vxs_arr);
    const real_t* vy_data = (const real_t*)PyArray_DATA(vys_arr);
    const real_t* vz_data = (const real_t*)PyArray_DATA(vzs_arr);
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
    PyErr_SetString(PyExc_MemoryError, "failed to copy particles into the field");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* sifField_sort_morton(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  sif_field_sort_morton(self->field);
  Py_RETURN_NONE;
}

static PyObject* sifField_compute_bounds(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;
  sif_field_compute_bounds(self->field);
  Py_RETURN_NONE;
}

/* --- Method Definition Array --- */
static PyMethodDef sifField_methods[] = {
  {"from_numpy", (PyCFunction)sifField_from_numpy, METH_VARARGS | METH_KEYWORDS,
    "Load particles and optional velocities from NumPy arrays."},
  {"sort_morton", (PyCFunction)sifField_sort_morton, METH_NOARGS,
    "Compute bounds and sort particles by Morton code."},
  {"compute_bounds", (PyCFunction)sifField_compute_bounds, METH_NOARGS,
    "Compute and cache the bounding box without sorting."}, /* <--- ADD THIS
                                                               LINE */
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */
PyTypeObject sifFieldType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.structures.Field", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifFieldObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifField_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF particle field object.",
  .tp_methods = sifField_methods,
  .tp_init = sifField_init,
  .tp_new = PyType_GenericNew,
};
