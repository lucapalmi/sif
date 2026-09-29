/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_field.h"

#include "model/py_model.h"

#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <string.h>

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

PyObject* pysif_field_from_numpy(
  PyObject* module, PyObject* args, PyObject* kwds) {
  (void)module;
  PyObject *xs_obj = Py_None, *ys_obj = Py_None, *zs_obj = Py_None;
  PyObject *ra_obj = Py_None, *dec_obj = Py_None;
  PyObject *vxs_obj = Py_None, *vys_obj = Py_None, *vzs_obj = Py_None;
  PyObject* ws_obj = Py_None;
  static char* kwlist[] = {
    "x", "y", "z", "ra", "dec", "vx", "vy", "vz", "weights", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO$OOOOOO", kwlist, &xs_obj,
        &ys_obj, &zs_obj, &ra_obj, &dec_obj, &vxs_obj, &vys_obj, &vzs_obj,
        &ws_obj))
    return NULL;

  /* The names say what the positions are: x y z, or ra dec z on the sky --
   * never some of each. */
  const bool xy = xs_obj != Py_None || ys_obj != Py_None;
  const bool radec = ra_obj != Py_None || dec_obj != Py_None;
  if (xy && radec) {
    PyErr_SetString(PyExc_ValueError,
      "give x, y and z for positions, or ra, dec and z for sky coordinates, "
      "not some of each");
    return NULL;
  }
  PyObject* first = radec ? ra_obj : xs_obj;
  PyObject* second = radec ? dec_obj : ys_obj;
  if (first == Py_None || second == Py_None || zs_obj == Py_None) {
    PyErr_SetString(
      PyExc_ValueError, radec ? "ra, dec and z are all required"
                              : "x, y and z are required (or ra, dec and z)");
    return NULL;
  }

  /* None and absent mean the same thing for every optional column. */
  const int n_vel =
    (vxs_obj != Py_None) + (vys_obj != Py_None) + (vzs_obj != Py_None);
  if (n_vel != 0 && n_vel != 3) {
    PyErr_SetString(PyExc_ValueError,
      "If providing velocities, vx, vy, and vz must all be provided");
    return NULL;
  }

  /* Every column is held here until the field is built; the first sets the
   * length the others are checked against. */
  enum { X, Y, Z, VX, VY, VZ, W, N_COLS };
  PyArrayObject* cols[N_COLS] = {NULL};
  sif_field_t* field = NULL;
  int status = SIF_OK;

  if (column_from_numpy(first, radec ? "ra" : "x", -1, &cols[X]) < 0)
    goto fail;

  const npy_intp n = PyArray_SHAPE(cols[X])[0];

  if (column_from_numpy(second, radec ? "dec" : "y", n, &cols[Y]) < 0 ||
      column_from_numpy(zs_obj, "z", n, &cols[Z]) < 0 ||
      column_from_numpy(vxs_obj, "vx", n, &cols[VX]) < 0 ||
      column_from_numpy(vys_obj, "vy", n, &cols[VY]) < 0 ||
      column_from_numpy(vzs_obj, "vz", n, &cols[VZ]) < 0 ||
      column_from_numpy(ws_obj, "weights", n, &cols[W]) < 0)
    goto fail;

  field = sif_field_alloc((uint64_t)n);
  if (!field) {
    PyErr_SetString(PyExc_MemoryError, "failed to allocate the field");
    goto fail;
  }

#define COL(c) ((const sif_real*)PyArray_DATA(cols[c]))
  status = sif_field_assign_positions(field, COL(X), COL(Y), COL(Z));
  if (status == SIF_OK && cols[VX])
    status = sif_field_assign_velocities(field, COL(VX), COL(VY), COL(VZ));
  if (status == SIF_OK && cols[W])
    status = sif_field_assign_weights(field, COL(W));
#undef COL

  if (status != SIF_OK) {
    PyErr_SetString(
      PyExc_MemoryError, "failed to copy particles into the field");
    goto fail;
  }
  field->units = radec ? SIF_COORDINATES_SKY : SIF_COORDINATES_CARTESIAN;

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    PyErr_NoMemory();
    goto fail;
  }
  obj->field = field;

  for (int c = 0; c < N_COLS; c++)
    Py_XDECREF(cols[c]);
  return (PyObject*)obj;

fail:
  for (int c = 0; c < N_COLS; c++)
    Py_XDECREF(cols[c]);
  sif_field_free(field);
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
  if (py_sif_field_check_cartesian(self, "wrap it") < 0)
    return NULL;

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

static PyObject* sifField_translate(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  double ox, oy, oz;
  static char* kwlist[] = {"offset", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ddd)", kwlist, &ox, &oy, &oz))
    return NULL;
  if (py_sif_field_check_cartesian((sifFieldObject*)self_obj, "translate it") <
      0)
    return NULL;

  const sif_real offset[3] = {(sif_real)ox, (sif_real)oy, (sif_real)oz};
  if (sif_field_translate(((sifFieldObject*)self_obj)->field, offset) !=
      SIF_OK) {
    PyErr_SetString(PyExc_ValueError, "cannot translate an empty field");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* sifField_sort_morton(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  if (py_sif_field_check_exports(self, "sort_morton()") < 0 ||
      py_sif_field_check_cartesian(self, "sort it") < 0)
    return NULL;

  const int status = sif_field_sort_morton(self->field);
  if (status == SIF_ERR_ALLOC)
    return PyErr_NoMemory();
  if (status != SIF_OK) {
    PyErr_SetString(PyExc_ValueError, "cannot sort an empty field");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* sifField_refresh_bounds(PyObject* self_obj, PyObject* args) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  if (py_sif_field_check_cartesian(self, "bound it") < 0)
    return NULL;
  if (sif_field_refresh_bounds(self->field) != SIF_OK) {
    PyErr_SetString(
      PyExc_ValueError, "cannot bound an empty or positionless field");
    return NULL;
  }

  Py_RETURN_NONE;
}

/* --- Array views --- */

int py_sif_field_check_exports(sifFieldObject* self, const char* action) {
  if (self->n_exports == 0)
    return 0;
  PyErr_Format(PyExc_BufferError,
    "cannot %s: the field has %zd live array view%s (field.x and the like), "
    "and this reallocates the arrays they point into. Delete the views, or "
    "keep copies instead (numpy.array(field.x)), first",
    action, self->n_exports, self->n_exports == 1 ? "" : "s");
  return -1;
}

int py_sif_field_check_cartesian(sifFieldObject* self, const char* action) {
  if (self->field->units == SIF_COORDINATES_CARTESIAN)
    return 0;
  PyErr_Format(PyExc_ValueError,
    "cannot %s: the field holds sky coordinates (right ascension, "
    "declination, redshift), not positions; call convert_sky_coordinates() "
    "first",
    action);
  return -1;
}

/* --- Sky coordinates --- */

static PyObject* sifField_convert_sky_coordinates(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifFieldObject* self = (sifFieldObject*)self_obj;

  double omega_m;
  PyObject* omega_de = Py_None;
  double omega_r = 0.0, w0 = -1.0, wa = 0.0;
  static char* kwlist[] = {"omega_m", "omega_de", "omega_r", "w0", "wa", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "d|Oddd", kwlist, &omega_m, &omega_de, &omega_r, &w0, &wa))
    return NULL;

  sif_cosmology_t cosmo;
  if (py_sif_cosmology_from(omega_m, omega_de, omega_r, w0, wa, &cosmo) < 0)
    return NULL;

  sif_field_t* f = self->field;
  if (f->n_particles == 0 || !f->x) {
    PyErr_SetString(
      PyExc_ValueError, "the field holds no positions to convert");
    return NULL;
  }
  if (f->units != SIF_COORDINATES_SKY) {
    PyErr_SetString(PyExc_ValueError,
      "the field already holds Cartesian positions; converting them would "
      "read x as a right ascension. Set units='sky' for a field that holds "
      "sky coordinates");
    return NULL;
  }

  /* The largest redshift, for the message if the cosmology cannot reach it;
   * the C side works it out again, and checks every coordinate. */
  double z_max = 0.0;
  for (uint64_t i = 0; i < f->n_particles; i++)
    if ((double)f->z[i] > z_max)
      z_max = (double)f->z[i];

  int status;
  Py_BEGIN_ALLOW_THREADS status = sif_field_convert_sky_coordinates(f, &cosmo);
  Py_END_ALLOW_THREADS

    switch (status) {
  case SIF_OK:
    Py_RETURN_NONE;
  case SIF_ERR_ALLOC:
    return PyErr_NoMemory();
  case SIF_ERR_RANGE:
    return py_sif_cosmology_range_error(z_max);
  default:
    PyErr_SetString(PyExc_ValueError,
      "some coordinates cannot be converted -- a declination outside "
      "[-90, 90], a negative or non-finite redshift, or a non-finite right "
      "ascension; the log has the counts, and the field is unchanged. Are x, "
      "y, z the right ascension, the declination and the redshift, in "
      "degrees?");
    return NULL;
  }
}

static PyObject* sifField_get_units(PyObject* self_obj, void* closure) {
  (void)closure;
  return PyUnicode_FromString(
    ((sifFieldObject*)self_obj)->field->units == SIF_COORDINATES_SKY
      ? "sky"
      : "cartesian");
}

static int sifField_set_units(
  PyObject* self_obj, PyObject* value, void* closure) {
  (void)closure;
  const char* s =
    value && PyUnicode_Check(value) ? PyUnicode_AsUTF8(value) : NULL;
  if (!s || (strcmp(s, "cartesian") != 0 && strcmp(s, "sky") != 0)) {
    PyErr_Clear();
    PyErr_SetString(PyExc_ValueError, "units must be 'cartesian' or 'sky'");
    return -1;
  }
  ((sifFieldObject*)self_obj)->field->units =
    s[0] == 's' ? SIF_COORDINATES_SKY : SIF_COORDINATES_CARTESIAN;
  return 0;
}

#define EXPORT_CAPSULE "pysif.Field.view"

/* The view's base: holds the field alive, and counts as an export for as long
 * as the array does. */
static void export_release(PyObject* capsule) {
  sifFieldObject* owner = (sifFieldObject*)PyCapsule_GetContext(capsule);
  owner->n_exports--;
  Py_DECREF(owner);
}

/*
 * A read-only, zero-copy view of one column, or None if the field does not
 * carry it. Read-only because a write from Python would bypass what the field
 * keeps about its own data -- its cached bounds and its Morton order.
 */
static PyObject* column_view(sifFieldObject* self, sif_real* data) {
  if (!data || self->field->n_particles == 0)
    Py_RETURN_NONE;

  PyObject* guard = PyCapsule_New(data, EXPORT_CAPSULE, export_release);
  if (!guard)
    return NULL;
  if (PyCapsule_SetContext(guard, self) < 0) {
    /* No context yet, so the destructor must not run as a release. */
    PyCapsule_SetDestructor(guard, NULL);
    Py_DECREF(guard);
    return NULL;
  }
  Py_INCREF(self);
  self->n_exports++;

  npy_intp dims[1] = {(npy_intp)self->field->n_particles};
  PyObject* array = py_sif_wrap_borrowed(guard, 1, dims, data);
  Py_DECREF(guard); /* the array holds it now, or it is released here */
  return array;
}

#define COLUMN_GETTER(name)                                                    \
  static PyObject* sifField_get_##name(PyObject* self_obj, void* closure) {    \
    (void)closure;                                                             \
    sifFieldObject* self = (sifFieldObject*)self_obj;                          \
    return column_view(self, self->field->name);                               \
  }

COLUMN_GETTER(x)
COLUMN_GETTER(y)
COLUMN_GETTER(z)
COLUMN_GETTER(vx)
COLUMN_GETTER(vy)
COLUMN_GETTER(vz)
COLUMN_GETTER(weights)

#undef COLUMN_GETTER

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
  {"units", sifField_get_units, sifField_set_units,
    "str: What the positions are: 'cartesian', or 'sky' for right\n"
    "ascension and declination (degrees) and redshift held in x, y, z.\n"
    "A sky field is refused by everything that reads positions as lengths\n"
    "-- grids, meshes, octrees, wrap(), translate(), sort_morton() -- until\n"
    "convert_sky_coordinates() turns it into positions. Setting it only\n"
    "relabels the arrays; it does not convert them.",
    NULL},
  {"x", sifField_get_x, NULL,
    "ndarray: x positions, a read-only view of the field's own array (no\n"
    "copy). See the class docstring for what a live view prevents.",
    NULL},
  {"y", sifField_get_y, NULL, "ndarray: y positions; see x.", NULL},
  {"z", sifField_get_z, NULL, "ndarray: z positions; see x.", NULL},
  {"vx", sifField_get_vx, NULL,
    "ndarray or None: x velocities, a read-only view; see x.", NULL},
  {"vy", sifField_get_vy, NULL, "ndarray or None: y velocities; see vx.", NULL},
  {"vz", sifField_get_vz, NULL, "ndarray or None: z velocities; see vx.", NULL},
  {"weights", sifField_get_weights, NULL,
    "ndarray or None: per-particle weights, a read-only view; see x.", NULL},
  {NULL}};

/* --- Conversion --- */

static PyObject* sifField_to_numpy(PyObject* self_obj, PyObject* unused) {
  (void)unused;
  const sif_field_t* f = ((sifFieldObject*)self_obj)->field;
  if (f->n_particles > 0 && !f->x) {
    PyErr_SetString(PyExc_ValueError, "the field has no positions to export");
    return NULL;
  }

  /* Positions, then velocities, then weights: whichever the field has. */
  const sif_real* cols[7] = {f->x, f->y, f->z};
  int n_cols = 3;
  if (f->vx) {
    cols[n_cols++] = f->vx;
    cols[n_cols++] = f->vy;
    cols[n_cols++] = f->vz;
  }
  if (f->weights)
    cols[n_cols++] = f->weights;

  npy_intp dims[2] = {(npy_intp)f->n_particles, n_cols};
  PyObject* arr = PyArray_SimpleNew(
    2, dims, sizeof(sif_real) == 8 ? NPY_FLOAT64 : NPY_FLOAT32);
  if (!arr)
    return NULL;
  sif_real* data = PyArray_DATA((PyArrayObject*)arr);
  for (uint64_t i = 0; i < f->n_particles; i++)
    for (int k = 0; k < n_cols; k++)
      data[i * (uint64_t)n_cols + (uint64_t)k] = cols[k][i];
  return arr;
}

/* --- Method Definition Array --- */
static PyMethodDef sifField_methods[] = {
  {"to_numpy", (PyCFunction)sifField_to_numpy, METH_NOARGS,
    "to_numpy()\n"
    "--\n\n"
    "The field as one (N, k) NumPy array, a copy: a row per particle, with\n"
    "columns x, y, z (ra, dec, z on the sky), then vx, vy, vz if the field\n"
    "has velocities, then the weight if it has weights -- 3, 6, 4 or 7\n"
    "columns. Rows are in the field's current order, which sort_morton()\n"
    "changes.\n\n"
    "Raises:\n"
    "    ValueError: For a field with particles but no positions."},
  {"convert_sky_coordinates", (PyCFunction)sifField_convert_sky_coordinates,
    METH_VARARGS | METH_KEYWORDS,
    "convert_sky_coordinates(omega_m, omega_de=None, omega_r=0.0, w0=-1.0, "
    "wa=0.0)\n"
    "--\n\n"
    "Turn sky coordinates into comoving Cartesian positions, in place.\n\n"
    "The field has to hold right ascension, declination (both in degrees)\n"
    "and redshift in x, y and z, with units='sky'. Each tracer goes to the\n"
    "line-of-sight comoving distance of its redshift, along its direction,\n"
    "with the observer at the origin -- the convention of pyrecon's\n"
    "sky_to_cartesian(). Positions come out in Mpc/h, and units becomes\n"
    "'cartesian'. Weights and velocities are untouched.\n\n"
    "Every coordinate is checked first: on a declination outside [-90, 90]\n"
    "or a negative or non-finite redshift nothing is changed.\n\n"
    "Args:\n"
    "    omega_m: Matter density today.\n"
    "    omega_de: Dark-energy density today; None, the default, makes the\n"
    "        model flat.\n"
    "    omega_r: Radiation density today.\n"
    "    w0, wa: The dark-energy equation of state, w(a) = w0 + wa (1 - a):\n"
    "        -1 and 0 for a cosmological constant.\n\n"
    "Raises:\n"
    "    ValueError: For a field that is not in sky coordinates or holds\n"
    "        one out of range, a bad cosmology, or one with no expansion\n"
    "        history out to the largest redshift."},
  {"wrap", (PyCFunction)sifField_wrap, METH_VARARGS | METH_KEYWORDS,
    "wrap(box_length) -> dict\n\n"
    "Fold every coordinate into [0, box_length) periodically, in place.\n"
    "Returns {'boundary': n, 'wrapped': n}: 'boundary' counts coordinates\n"
    "that sat exactly on the box edge, which is the single-precision\n"
    "rounding artifact this exists for, and 'wrapped' counts coordinates\n"
    "that were genuinely outside. A large 'wrapped' means the box length is\n"
    "wrong and the folded field is meaningless -- check it rather than\n"
    "proceeding. Only correct for a field that is periodic in this box."},
  {"translate", (PyCFunction)sifField_translate, METH_VARARGS | METH_KEYWORDS,
    "translate(offset)\n"
    "--\n\n"
    "Shift every position by offset, in place.\n\n"
    "What moves a survey into the box pysif.finders.survey_box() chose for\n"
    "it: data and randoms both, by the same offset. The voids found there\n"
    "come back out with Catalogue.translate(-offset).\n\n"
    "Args:\n"
    "    offset: Three numbers, added to x, y and z."},
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
    "pysif.Field", /* Updated Namespace and Capitalised */
  .tp_basicsize = sizeof(sifFieldObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifField_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Field(capacity=0)\n"
    "--\n\n"
    "A particle field: positions, and optionally velocities and\n"
    "weights.\n\n"
    "The container every other structure is built from. Make one with\n"
    "pysif.field_from_numpy(), or read one off disk with\n"
    "pysif.io.read_field(), read_fits() or read_gadget(). Field() itself\n"
    "is empty.\n\n"
    "x, y, z, vx, vy, vz and weights are read-only NumPy views of the\n"
    "field's own arrays, in its current particle order -- which\n"
    "sort_morton() changes. They see in-place changes (wrap(),\n"
    "translate()). While any view is alive, whatever would reallocate or\n"
    "take the arrays raises BufferError instead: sort_morton(), building\n"
    "an Octree over an unsorted field, and ChainMesh(...,\n"
    "consume_field=True). Delete the views first, or keep copies\n"
    "(numpy.array(field.x)).\n\n"
    "units says what x, y and z hold: 'cartesian' positions, or 'sky' --\n"
    "right ascension, declination and redshift.\n\n"
    "Args:\n"
    "    capacity: Particles to make room for up front.",
  .tp_methods = sifField_methods,
  .tp_getset = sifField_getset,
  .tp_init = sifField_init,
  .tp_new = PyType_GenericNew,
};
