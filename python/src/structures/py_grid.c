/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_grid.h"
#include "py_field.h"
#include "py_tessellation.h"
#include <stdio.h>

/* --- Lifecycle Methods --- */

static void sifGrid_dealloc(PyObject* self_obj) {
  sifGridObject* self = (sifGridObject*)self_obj;
  if (self->grid != NULL) {
    sif_grid_free(self->grid);
    self->grid = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifGrid_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  unsigned int n_cells;
  double box_length_in;

  static char* kwlist[] = {"n_cells", "box_length", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "Id", kwlist, &n_cells, &box_length_in)) {
    return -1;
  }

  sif_grid_t* tmp = sif_grid_alloc(n_cells, (sif_real)box_length_in);
  if (!tmp) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sif.grid");
    return -1;
  }

  sifGridObject* self = (sifGridObject*)self_obj;
  self->grid = tmp;
  return 0;
}

/* --- Pythonic Methods --- */

/*
 * A CIC deposit's status as an exception, or 0 for success. The C side names
 * the reason in its log; this is what makes it reach a Python caller at all.
 */
static int raise_for_cic_status(int status) {
  switch (status) {
  case SIF_OK:
    return 0;
  case SIF_ERR_RANGE:
    PyErr_SetString(PyExc_ValueError,
      "some particles lie outside [0, box_length) and cannot be deposited: "
      "fold a periodic snapshot with Field.wrap(), or move a survey into its "
      "box with Field.translate() (see pysif.finders.survey_box())");
    return -1;
  case SIF_ERR_ALLOC:
    PyErr_SetString(
      PyExc_MemoryError, "out of memory depositing the field onto the grid");
    return -1;
  default:
    PyErr_SetString(PyExc_ValueError,
      "nothing to deposit: the field holds no particles (a ChainMesh built "
      "with consume_field=True empties it)");
    return -1;
  }
}

static PyObject* sifGrid_assign_cic(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifGridObject* self = (sifGridObject*)self_obj;
  PyObject* field_obj = NULL;

  static char* kwlist[] = {"field", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!", kwlist, &sifFieldType, &field_obj)) {
    return NULL;
  }

  sifFieldObject* py_field = (sifFieldObject*)field_obj;

  /* A full pass over the field and a scatter over the grid: worth the GIL at
   * the particle counts this is for, as for the tessellation deposit. */
  int status;
  Py_BEGIN_ALLOW_THREADS status =
    sif_grid_assign_cic(self->grid, py_field->field);
  Py_END_ALLOW_THREADS

    if (raise_for_cic_status(status) < 0) return NULL;

  Py_RETURN_NONE;
}

static PyObject* sifGrid_assign_cic_tessellation(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifGridObject* self = (sifGridObject*)self_obj;
  PyObject* tess_obj = NULL;

  static char* kwlist[] = {"tessellation", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!", kwlist, &sifTessellationType, &tess_obj)) {
    return NULL;
  }

  sifTessellationObject* py_tess = (sifTessellationObject*)tess_obj;

  /* The C entry point reports its refusals by logging, which a Python caller
     cannot see, so the two that a caller can actually cause are checked here
     and raised. */
  if (!py_tess->tess->samples || py_tess->tess->samples->n_particles == 0) {
    PyErr_SetString(PyExc_ValueError,
      "these samples have already been consumed by mesh(consume=True); there "
      "is nothing left to deposit");
    return NULL;
  }

  if (self->grid->box_length != py_tess->tess->box_length) {
    /* PyErr_Format has no float conversion, so anything carrying a double has
       to be rendered before it gets there. */
    char detail[256];
    snprintf(detail, sizeof(detail),
      "the grid spans a box of %g but the tessellation was built in one of %g",
      (double)self->grid->box_length, (double)py_tess->tess->box_length);
    PyErr_SetString(PyExc_ValueError, detail);
    return NULL;
  }

  int status;
  Py_BEGIN_ALLOW_THREADS status =
    sif_grid_assign_cic_tessellation(self->grid, py_tess->tess);
  Py_END_ALLOW_THREADS

    if (raise_for_cic_status(status) < 0) return NULL;

  Py_RETURN_NONE;
}

static PyObject* sifGrid_to_density_contrast(
  PyObject* self_obj, PyObject* args) {
  sifGridObject* self = (sifGridObject*)self_obj;

  /* Checked here as well as in C so the message can say which of the two
   * refusals it was: the status alone does not separate a second conversion
   * from a grid with nothing in it. */
  if (self->grid->content == SIF_GRID_DENSITY_CONTRAST) {
    PyErr_SetString(PyExc_ValueError,
      "this grid already holds a density contrast; converting again would "
      "renormalize by a mean of zero");
    return NULL;
  }

  if (sif_grid_to_density_contrast(self->grid) != SIF_OK) {
    PyErr_SetString(PyExc_ValueError,
      "the grid's mean density is not positive, so there is nothing to "
      "normalize by: deposit a field onto it with assign_cic() first");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef sifGrid_methods[] = {
  {"assign_cic", (PyCFunction)sifGrid_assign_cic, METH_VARARGS | METH_KEYWORDS,
    "assign_cic(field)\n"
    "--\n\n"
    "Deposit a particle field onto the grid by Cloud-In-Cell assignment.\n\n"
    "Each particle contributes to the eight cells around it. Weights are\n"
    "used when the field carries them, otherwise every particle counts as\n"
    "one. Afterwards the cells hold a density, not a density contrast.\n\n"
    "Every coordinate must lie in [0, box_length); use Field.wrap() first\n"
    "if a periodic snapshot has drifted onto the boundary, or\n"
    "Field.translate() to move a survey into its box.\n\n"
    "Args:\n"
    "    field: The particle field to deposit.\n\n"
    "Raises:\n"
    "    ValueError: If a particle lies outside the box, or the field holds\n"
    "        none. The grid is left as it was.\n"
    "    MemoryError: If the deposit could not allocate its scratch."},
  {"assign_cic_tessellation", (PyCFunction)sifGrid_assign_cic_tessellation,
    METH_VARARGS | METH_KEYWORDS,
    "assign_cic_tessellation(tessellation)\n"
    "--\n\n"
    "Deposit a Tessellation onto the grid by Cloud-In-Cell assignment.\n\n"
    "The tessellation's density rather than the tracers': every sample\n"
    "carries a share of its owner's weight and lands where the owner's cell\n"
    "actually reaches, so a region with few tracers is filled by whatever\n"
    "large cells cover it instead of being left empty. Depositing sparse\n"
    "tracers directly leaves most cells holding nothing and the rest holding\n"
    "shot noise, and no smoothing afterwards puts back a field that was\n"
    "never sampled.\n\n"
    "The result is a density on the same scale and with the same mean as\n"
    "assign_cic() of the tracers, so to_density_contrast() and everything\n"
    "downstream treat it identically.\n\n"
    "Costs the sampling rate times what depositing the tracers would.\n\n"
    "Args:\n"
    "    tessellation: The Tessellation to deposit. Its samples must still\n"
    "        be present -- not one whose mesh(consume=True) has taken them."},
  {"to_density_contrast", (PyCFunction)sifGrid_to_density_contrast, METH_NOARGS,
    "to_density_contrast()\n"
    "--\n\n"
    "Convert the cells in place from a density to a density contrast.\n\n"
    "Replaces each cell with delta = rho / rho_mean - 1, so the field\n"
    "averages to zero and is bounded below by -1. This is the form the\n"
    "finders and pysif.measure expect.\n\n"
    "Calling it twice is refused: the second pass would renormalize by a\n"
    "mean that is now zero."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */
PyTypeObject sifGridType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.Grid", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifGridObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifGrid_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Grid(n_cells, box_length)\n"
            "--\n\n"
            "A cubic grid of n_cells**3 cells over a periodic box.\n\n"
            "Deposit a field onto it with assign_cic(), then convert it with\n"
            "to_density_contrast() before handing it to a finder or to\n"
            "pysif.measure.\n\n"
            "Args:\n"
            "    n_cells: Cells per side.\n"
            "    box_length: Physical side length of the box.",
  .tp_methods = sifGrid_methods,
  .tp_init = sifGrid_init,
  .tp_new = PyType_GenericNew,
};