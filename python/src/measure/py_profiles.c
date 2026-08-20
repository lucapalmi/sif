/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_profiles.h"

#include "structures/py_catalog.h"
#include "structures/py_chain_mesh.h"
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

static PyObject* sifProfiles_get_differential(
  PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (!self->dens)
    Py_RETURN_NONE;
  return PyBool_FromLong(self->dens->differential);
}

static PyObject* sifProfiles_get_r_bins(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  if (!self->dens)
    Py_RETURN_NONE;

  const uint32_t n_bins = self->dens->n_bins;
  npy_intp dims[1] = {n_bins};
  PyObject* array = PyArray_SimpleNew(1, dims, NPY_REAL_T);
  if (!array)
    return NULL;

  sif_real* out = (sif_real*)PyArray_DATA((PyArrayObject*)array);
  for (uint32_t b = 0; b < n_bins; b++)
    out[b] = sif_density_profiles_bin_radius(self->dens, b);

  return array;
}

static PyObject* sifProfiles_get_r_edges(PyObject* self_obj, void* closure) {
  sifProfilesObject* self = (sifProfilesObject*)self_obj;
  sif_real* edges =
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
  {"differential", sifProfiles_get_differential, NULL,
    "True when a density bin holds its own shell, False when it holds "
    "everything enclosed. None if there are no densities.",
    NULL},
  {"r_edges", sifProfiles_get_r_edges, NULL, "1D array of bin edges", NULL},
  {"r_bins", sifProfiles_get_r_bins, NULL,
    "Radius each density bin's value belongs at, in units of the void "
    "radius: the outer edge for a cumulative profile, the midpoint for a "
    "differential one. Plot against this, not against bin centres -- for a "
    "cumulative profile those differ by half a bin, enough to move a feature "
    "at r = R_v off that mark. None if there are no densities.",
    NULL},
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
  .tp_doc =
    "Profiles()\n"
    "--\n\n"
    "Stacked radial profiles of voids: density and radial velocity.\n\n"
    "Returned by pysif.measure.profiles(). One row per void, binned in\n"
    "radius scaled by that void's own radius, so profiles of\n"
    "different-sized voids stack directly.\n\n"
    "Not constructed directly.",
  .tp_getset = sifProfiles_getset,
  .tp_init = sifProfiles_init,
  .tp_new = PyType_GenericNew,
};

/* --- Functional API Implementation --- */

PyObject* py_sif_profiles_suggest_mesh_cells(
  PyObject* self, PyObject* args, PyObject* kwds) {

  unsigned long long n_particles;

  static char* kwlist[] = {"n_particles", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &n_particles))
    return NULL;

  /* The C helper folds a zero count into a resolution of 1, which is a legal
     mesh and so would not fail until the run was pointlessly slow. Here the
     caller gets told instead. */
  if (n_particles == 0) {
    PyErr_SetString(PyExc_ValueError, "need n_particles > 0");
    return NULL;
  }

  return PyLong_FromUnsignedLong(
    sif_profiles_suggest_mesh_cells((uint64_t)n_particles));
}

PyObject* py_sif_profiles(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* cat_obj = NULL;
  PyObject* mesh_obj = NULL;
  uint32_t n_bins;

  double ext = 0.0; /* 0 selects the library default */
  int compute_velocity = 0;
  int use_pbc = 1;
  int differential = 0;

  static char* kwlist[] = {"catalog", "mesh", "n_bins", "ext",
    "compute_velocity", "use_pbc", "differential", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!I|dppp", kwlist,
        &sifCatalogType, &cat_obj, &sifChainMeshType, &mesh_obj, &n_bins, &ext,
        &compute_velocity, &use_pbc, &differential)) {
    return NULL;
  }

  const sif_catalog_t* c_cat = ((sifCatalogObject*)cat_obj)->catalog;
  const sif_chain_mesh_t* c_mesh = ((sifChainMeshObject*)mesh_obj)->mesh;

  /* Checked here as well as in the C call so the message names the mesh the
     caller passed, rather than arriving as a generic backend failure. */
  if (compute_velocity && !c_mesh->vx) {
    PyErr_SetString(PyExc_ValueError,
      "Cannot compute velocity profiles: the mesh carries no velocities. "
      "Build it from a Field that has them.");
    return NULL;
  }

  const uint32_t options =
    (use_pbc ? SIF_PBC_PERIODIC : SIF_PBC_OPEN) |
    (differential ? SIF_PROFILES_DIFFERENTIAL : SIF_PROFILES_CUMULATIVE);

  sif_density_profiles_t* dens_out = NULL;
  sif_velocity_profiles_t* vel_out = NULL;

  /* The GIL is released for the duration: this is a long, purely numeric run
     across every core and nothing below touches Python state. The mesh is
     borrowed for it; mesh_obj is kept alive by the caller's reference. */
  int status = SIF_OK;
  Py_BEGIN_ALLOW_THREADS status = sif_profiles(c_cat, c_mesh, (sif_real)ext,
    n_bins, options, &dens_out, compute_velocity ? &vel_out : NULL);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) {
    PyErr_SetString(
      status == SIF_ERR_ALLOC ? PyExc_MemoryError : PyExc_ValueError,
      "Backend profile engine execution failed; see the log for the reason.");
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
