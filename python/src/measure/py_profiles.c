#include "py_profiles.h"

#include "structures/py_catalog.h"
#include "structures/py_field.h"
#include "structures/py_tessellation.h"
#include <numpy/arrayobject.h>

static void sifProfiles_dealloc(PyObject* self_obj) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (self->dens != NULL) {
    sif_density_profiles_free(self->dens);
    self->dens = NULL;
  }
  if (self->vel != NULL) {
    sif_velocity_profiles_free(self->vel);
    self->vel = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifProfiles_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  self->dens = NULL;
  self->vel = NULL;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifProfiles_get_has_density(
  PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  return PyBool_FromLong(self->dens != NULL);
}

static PyObject* sifProfiles_get_has_velocity(
  PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  return PyBool_FromLong(self->vel != NULL);
}

static PyObject* sifProfiles_get_n_voids(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (self->dens)
    return PyLong_FromUnsignedLongLong(self->dens->n_voids);
  if (self->vel)
    return PyLong_FromUnsignedLongLong(self->vel->n_voids);
  Py_RETURN_NONE;
}

static PyObject* sifProfiles_get_n_bins(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (self->dens)
    return PyLong_FromUnsignedLong(self->dens->n_bins);
  if (self->vel)
    return PyLong_FromUnsignedLong(self->vel->n_bins);
  Py_RETURN_NONE;
}

static PyObject* sifProfiles_get_ext(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (self->dens)
    return PyFloat_FromDouble((double)self->dens->ext);
  if (self->vel)
    return PyFloat_FromDouble((double)self->vel->ext);
  Py_RETURN_NONE;
}

static PyObject* sifProfiles_get_r_edges(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  real_t* edges =
    self->dens ? self->dens->r_edges : (self->vel ? self->vel->r_edges : NULL);
  uint32_t n_bins =
    self->dens ? self->dens->n_bins : (self->vel ? self->vel->n_bins : 0);

  if (!edges)
    Py_RETURN_NONE;

  npy_intp dims[1] = {n_bins + 1};
  PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, edges);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifProfiles_get_density(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (!self->dens || !self->dens->profiles)
    Py_RETURN_NONE;

  npy_intp dims[2] = {self->dens->n_voids, self->dens->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(2, dims, NPY_REAL_T, self->dens->profiles);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifProfiles_get_velocity(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (!self->vel || !self->vel->v_rad)
    Py_RETURN_NONE;

  npy_intp dims[2] = {self->vel->n_voids, self->vel->n_bins};
  PyObject* array =
    PyArray_SimpleNewFromData(2, dims, NPY_REAL_T, self->vel->v_rad);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyGetSetDef sifProfiles_getset[] = {
  {"has_density", sifProfiles_get_has_density, NULL,
    "True if density profile exists", NULL},
  {"has_velocity", sifProfiles_get_has_velocity, NULL,
    "True if velocity profile exists", NULL},
  {"n_voids", sifProfiles_get_n_voids, NULL, "Number of voids", NULL},
  {"n_bins", sifProfiles_get_n_bins, NULL, "Number of radial bins", NULL},
  {"ext", sifProfiles_get_ext, NULL, "Maximum scaling radius boundary", NULL},
  {"r_edges", sifProfiles_get_r_edges, NULL, "1D array of bin edges", NULL},
  {"density", sifProfiles_get_density, NULL, "2D array of density profiles",
    NULL},
  {"velocity", sifProfiles_get_velocity, NULL,
    "2D array of radial velocity profiles", NULL},
  {NULL}};

PyTypeObject sifProfilesType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.measure.Profiles",
  .tp_basicsize = sizeof(sifProfilesObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifProfiles_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Unified void structural profiles container.",
  .tp_getset = sifProfiles_getset,
  .tp_init = sifProfiles_init,
  .tp_new = PyType_GenericNew,
};

/* --- Functional API Implementation --- */

PyObject* py_sif_profiles(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* cat_obj;
  PyObject* field_obj;
  double box_length, ext;
  uint32_t n_bins;

  int compute_velocity = 0;
  int use_pbc = 1;
  const char* algorithm = "mesh";
  PyObject* tess_obj = NULL; /* New optional argument */

  /* Added 'tessellation' to the keyword list */
  static char* kwlist[] = {"catalog", "field", "box_length", "ext", "n_bins",
    "compute_velocity", "use_pbc", "algorithm", "tessellation", NULL};

  /* Format string updated: added 'O' at the end for the optional tess_obj */
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!ddI|ppsO", kwlist,
        &sifCatalogType, &cat_obj, &sifFieldType, &field_obj, &box_length, &ext,
        &n_bins, &compute_velocity, &use_pbc, &algorithm, &tess_obj)) {
    return NULL;
  }

  sifCatalogObject* cat = (sifCatalogObject*)cat_obj;
  sifFieldObject* field = (sifFieldObject*)field_obj;

  /* Test for the presence of velocities. This used to test an ownership flag
     instead, which rejected perfectly valid fields; the flags are gone and a
     field either has velocities or it does not. */
  if (compute_velocity && !field->field->vx) {
    PyErr_SetString(PyExc_ValueError,
      "Cannot compute velocity profiles: field missing raw velocities.");
    return NULL;
  }

  uint32_t options = use_pbc ? SIF_PBC_PERIODIC : SIF_PBC_OPEN;

  if (strcmp(algorithm, "mesh") == 0) {
    options |= SIF_PROFILES_ALGO_MESH;
  } else if (strcmp(algorithm, "voronoi") == 0) {
    options |= SIF_PROFILES_ALGO_VORONOI;
  } else {
    PyErr_Format(PyExc_ValueError,
      "Unknown profile mapping algorithm '%s'. Use 'mesh' or 'voronoi'.",
      algorithm);
    return NULL;
  }

  sif_density_profiles_t* dens_out = NULL;
  sif_velocity_profiles_t* vel_out = NULL;

  /* --- Python-Side Dispatcher --- */
  if ((options & __SIF_PROFILES_ALGO_MASK) == SIF_PROFILES_ALGO_VORONOI) {

    if (tess_obj == NULL || tess_obj == Py_None) {
      PyErr_SetString(PyExc_ValueError,
        "Algorithm 'voronoi' requires a valid tessellation "
        "object passed to the 'tessellation' argument.");
      return NULL;
    }

    if (!PyObject_TypeCheck(tess_obj, &sifTessellationType)) {
      PyErr_SetString(PyExc_TypeError,
        "The 'tessellation' argument must be a valid "
        "pysif.structures.Tessellation object.");
      return NULL;
    }

    sif_tessellation_t* c_tess = ((sifTessellationObject*)tess_obj)->tess;

    /* The GIL is released for the duration: this is a long, purely numeric
       run across every core and nothing below touches Python state. */
    Py_BEGIN_ALLOW_THREADS
    sif_profiles_voronoi(cat->catalog, field->field, c_tess,
      (real_t)box_length, (real_t)ext, n_bins, options, &dens_out,
      compute_velocity ? &vel_out : NULL);
    Py_END_ALLOW_THREADS

  } else {

    Py_BEGIN_ALLOW_THREADS
    sif_profiles_mesh(cat->catalog, field->field, (real_t)box_length,
      (real_t)ext, n_bins, options, &dens_out,
      compute_velocity ? &vel_out : NULL);
    Py_END_ALLOW_THREADS
  }

  if (!dens_out) {
    PyErr_SetString(
      PyExc_RuntimeError, "Backend profile engine execution failed.");
    return NULL;
  }

  sifProfilesObject* obj =
    (sifProfilesObject*)sifProfilesType.tp_alloc(&sifProfilesType, 0);
  if (!obj) {
    sif_density_profiles_free(dens_out);
    if (vel_out)
      sif_velocity_profiles_free(vel_out);
    return PyErr_NoMemory();
  }

  obj->dens = dens_out;
  obj->vel = vel_out;

  return (PyObject*)obj;
}
