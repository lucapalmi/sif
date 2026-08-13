/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_grid.h"
#include "py_field.h"

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

  sif_grid_assign_cic(self->grid, py_field->field);

  Py_RETURN_NONE;
}

static PyObject* sifGrid_to_density_contrast(
  PyObject* self_obj, PyObject* args) {
  sifGridObject* self = (sifGridObject*)self_obj;

  /* The C entry point returns void and refuses a second conversion by
   * logging, which a Python caller cannot see. Check here so the refusal
   * arrives as an exception instead of as a silent no-op. */
  if (self->grid->content == SIF_GRID_DENSITY_CONTRAST) {
    PyErr_SetString(PyExc_ValueError,
      "this grid already holds a density contrast; converting again would "
      "renormalize by a mean of zero");
    return NULL;
  }

  sif_grid_to_density_contrast(self->grid);

  Py_RETURN_NONE;
}

static PyMethodDef sifGrid_methods[] = {
  {"assign_cic", (PyCFunction)sifGrid_assign_cic, METH_VARARGS | METH_KEYWORDS,
    "assign_cic(field)\n"
    "--\n\n"
    "Deposit a particle field onto the grid by Cloud-In-Cell assignment.\n\n"
    "Each particle contributes to the eight cells around it. Masses are\n"
    "used when the field carries them, otherwise every particle counts as\n"
    "one. Afterwards the cells hold mass, not density contrast.\n\n"
    "Every coordinate must lie in [0, box_length); use Field.wrap() first\n"
    "if a periodic snapshot has drifted onto the boundary.\n\n"
    "Args:\n"
    "    field: The particle field to deposit."},
  {"to_density_contrast", (PyCFunction)sifGrid_to_density_contrast, METH_NOARGS,
    "to_density_contrast()\n"
    "--\n\n"
    "Convert the cells in place from mass to density contrast.\n\n"
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