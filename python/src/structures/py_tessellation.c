/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_tessellation.h"
#include "py_chain_mesh.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifTessellation_dealloc(PyObject* self_obj) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (self->tess != NULL) {
    sif_tessellation_free(self->tess);
    self->tess = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifTessellation_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  PyObject* mesh_obj = NULL;
  uint32_t samples_per_tracer = SIF_TESSELLATION_DEFAULT_SAMPLES;
  unsigned long long seed = 0;
  int use_pbc = 1;

  static char* kwlist[] = {
    "mesh", "samples_per_tracer", "seed", "use_pbc", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|IKp", kwlist,
        &sifChainMeshType, &mesh_obj, &samples_per_tracer, &seed, &use_pbc)) {
    return -1;
  }

  sifChainMeshObject* mesh = (sifChainMeshObject*)mesh_obj;

  const sif_option opt = use_pbc ? SIF_PBC_PERIODIC : SIF_PBC_OPEN;

  sif_tessellation_t* tmp = NULL;
  Py_BEGIN_ALLOW_THREADS tmp =
    sif_tessellation_alloc(mesh->mesh, samples_per_tracer, (uint64_t)seed, opt);
  Py_END_ALLOW_THREADS

    if (!tmp) {
    PyErr_SetString(PyExc_ValueError,
      "Failed to build pysif.Tessellation; see the sif log for the reason");
    return -1;
  }

  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  sif_tessellation_free(self->tess);
  self->tess = tmp;
  return 0;
}

/* --- Properties (Getters) --- */

static PyObject* sifTessellation_get_n_tracers(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->tess->n_tracers);
}

static PyObject* sifTessellation_get_n_samples(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->tess->n_samples);
}

static PyObject* sifTessellation_get_n_empty(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->tess->n_empty);
}

static PyObject* sifTessellation_get_box_length(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  return PyFloat_FromDouble((double)self->tess->box_length);
}

static PyObject* sifTessellation_get_sample_volume(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  return PyFloat_FromDouble((double)self->tess->sample_volume);
}

static PyObject* sifTessellation_get_volumes(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  if (!self->tess->volumes)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->tess->n_tracers};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->tess->volumes);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifTessellation_get_consumed(
  PyObject* self_obj, void* closure) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;
  const int gone =
    (!self->tess->samples || self->tess->samples->n_particles == 0);
  return PyBool_FromLong(gone);
}

static PyGetSetDef sifTessellation_getset[] = {
  {"n_tracers", sifTessellation_get_n_tracers, NULL,
    "Tracers of the mesh this was built from", NULL},
  {"n_samples", sifTessellation_get_n_samples, NULL, "Samples thrown", NULL},
  {"n_empty", sifTessellation_get_n_empty, NULL,
    "Tracers no sample landed in. Non-zero means samples_per_tracer is too "
    "low: those cells read back with zero volume and their weight is missing "
    "from the samples.",
    NULL},
  {"box_length", sifTessellation_get_box_length, NULL, "Box length", NULL},
  {"sample_volume", sifTessellation_get_sample_volume, NULL,
    "Volume one sample stands for", NULL},
  {"volumes", sifTessellation_get_volumes, NULL,
    "Cell volume per tracer, in the mesh's order rather than the field's",
    NULL},
  {"consumed", sifTessellation_get_consumed, NULL,
    "True once mesh(consume=True) has taken the samples", NULL},
  {NULL}};

/* --- Methods --- */

static PyObject* sifTessellation_mesh(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifTessellationObject* self = (sifTessellationObject*)self_obj;

  uint32_t n_cells;
  int drop_indices = 1;
  int consume = 0;

  static char* kwlist[] = {"n_cells", "drop_indices", "consume", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "I|pp", kwlist, &n_cells, &drop_indices, &consume)) {
    return NULL;
  }

  const sif_option opt = drop_indices ? SIF_MESH_DROP_INDICES : SIF_DEFAULT;

  sif_chain_mesh_t* tmp = NULL;
  Py_BEGIN_ALLOW_THREADS tmp =
    consume
      ? sif_chain_mesh_alloc_tessellation_consume(self->tess, n_cells, opt)
      : sif_chain_mesh_alloc_tessellation(self->tess, n_cells, opt);
  Py_END_ALLOW_THREADS

    if (!tmp) {
    PyErr_SetString(PyExc_ValueError,
      "Failed to bin the samples into a ChainMesh; if consume=True was used "
      "before, the samples are already gone");
    return NULL;
  }

  sifChainMeshObject* obj =
    (sifChainMeshObject*)sifChainMeshType.tp_alloc(&sifChainMeshType, 0);
  if (!obj) {
    sif_chain_mesh_free(tmp);
    return PyErr_NoMemory();
  }

  obj->mesh = tmp;
  return (PyObject*)obj;
}

static PyMethodDef sifTessellation_methods[] = {
  {"mesh", (PyCFunction)sifTessellation_mesh, METH_VARARGS | METH_KEYWORDS,
    "mesh(n_cells, drop_indices=True, consume=False)\n"
    "--\n\n"
    "Bin the samples into a ChainMesh, ready to measure.\n\n"
    "What turns a tessellation into a profile: hand the result to\n"
    "measure.profiles() and the densities come out volume-weighted, since\n"
    "a sample stands for a piece of space rather than for a tracer.\n\n"
    "Args:\n"
    "    n_cells: Cells per side. Size it from n_samples, not from the\n"
    "        tracer count -- it is the samples being binned. See\n"
    "        measure.profiles_suggest_mesh_cells().\n"
    "    drop_indices: Release the index map; nothing downstream of here\n"
    "        needs to name a sample.\n"
    "    consume: Move the samples into the mesh instead of copying them.\n"
    "        The sample set is the largest thing in the process by a factor\n"
    "        of the sampling rate, so copying it means holding two. The\n"
    "        volumes are unaffected, but the samples cannot be handed out\n"
    "        again; consumed reads back as True.\n\n"
    "Returns:\n"
    "    ChainMesh: The samples, binned."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifTessellationType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.Tessellation",
  .tp_basicsize = sizeof(sifTessellationObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifTessellation_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "Tessellation(mesh, samples_per_tracer=32, seed=0, use_pbc=True)\n"
    "--\n\n"
    "Voronoi cell volumes, measured by sampling rather than constructed.\n\n"
    "Every point of space belongs to the tracer nearest it, so throwing\n"
    "uniform random points into the box and asking each one which tracer\n"
    "owns it measures the Voronoi partition directly: the fraction of the\n"
    "samples a tracer catches is the fraction of the box its cell occupies.\n"
    "No cell is ever built, the estimate is unbiased, and the total is\n"
    "conserved exactly since every sample lands in one cell.\n\n"
    "This buys two things. volumes is a density per tracer that adapts to\n"
    "the local sampling instead of to a fixed grid, which is what a sparse\n"
    "field wants. mesh() is the more useful one: each sample carries a\n"
    "share of its owner's weight, so binning the samples inside a sphere\n"
    "counts the fraction of every cell the sphere covers -- the geometric\n"
    "intersection that building the cells would otherwise be needed for.\n\n"
    "For fields too sparse to profile by counting tracers. The sample set\n"
    "is samples_per_tracer times the field, in memory and in the cost of\n"
    "everything measured from it, and a field dense enough to count has\n"
    "nothing to gain here.\n\n"
    "Args:\n"
    "    mesh: ChainMesh of the tracers. Weights and velocities ride along\n"
    "        into the samples if it carries them. drop_indices=True is\n"
    "        fine; nothing here names a tracer in field order.\n"
    "    samples_per_tracer: Samples thrown per tracer, which sets both the\n"
    "        accuracy and the size of the result. A cell's volume is\n"
    "        measured to about 1/sqrt(this).\n"
    "    seed: Any integer. The samples do not depend on the thread count.\n"
    "    use_pbc: Whether a sample near a face may be owned by a tracer\n"
    "        across it.",
  .tp_new = PyType_GenericNew,
  .tp_init = sifTessellation_init,
  .tp_getset = sifTessellation_getset,
  .tp_methods = sifTessellation_methods,
};
