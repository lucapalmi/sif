/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_grid.h"
#include "py_field.h"
#include "py_tessellation.h"
#include <stdio.h>
#include <string.h>

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

  sif_error_clear();
  sif_grid_t* tmp = sif_grid_alloc(n_cells, (sif_real)box_length_in);
  if (!tmp) {
    py_sif_raise(sif_error_status(), "Grid");
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
  if (py_sif_field_check_cartesian(py_field, "deposit it") < 0)
    return NULL;

  /* A full pass over the field and a scatter over the grid: worth the GIL at
   * the particle counts this is for, as for the tessellation deposit. */
  int status;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status =
    sif_grid_assign_cic(self->grid, py_field->field);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) return py_sif_field_raise_binning(status, "assign_cic");

  Py_RETURN_NONE;
}

static const char* const scheme_names[] = {NULL, "ngp", "cic", "tsc", "pcs"};

static PyObject* sifGrid_assign(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifGridObject* self = (sifGridObject*)self_obj;
  PyObject* field_obj = NULL;
  const char* scheme_name = "cic";
  int interlace = 0;

  static char* kwlist[] = {"field", "scheme", "interlace", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|s$p", kwlist, &sifFieldType,
        &field_obj, &scheme_name, &interlace)) {
    return NULL;
  }

  sif_assignment_t scheme = SIF_ASSIGNMENT_UNKNOWN;
  for (int s = SIF_ASSIGNMENT_NGP; s <= SIF_ASSIGNMENT_PCS; s++)
    if (strcmp(scheme_name, scheme_names[s]) == 0)
      scheme = (sif_assignment_t)s;
  if (scheme == SIF_ASSIGNMENT_UNKNOWN) {
    PyErr_Format(PyExc_ValueError,
      "scheme must be 'ngp', 'cic', 'tsc' or 'pcs', not '%s'", scheme_name);
    return NULL;
  }

  sifFieldObject* py_field = (sifFieldObject*)field_obj;
  if (py_sif_field_check_cartesian(py_field, "deposit it") < 0)
    return NULL;

  int status;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status = sif_grid_assign(self->grid, py_field->field,
    scheme, interlace ? SIF_GRID_INTERLACE : 0);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) return py_sif_field_raise_binning(status, "assign");

  Py_RETURN_NONE;
}

static PyObject* sifGrid_get_assignment(PyObject* self_obj, void* closure) {
  (void)closure;
  const sif_assignment_t a = ((sifGridObject*)self_obj)->grid->assignment;
  if (a >= SIF_ASSIGNMENT_NGP && a <= SIF_ASSIGNMENT_PCS)
    return PyUnicode_FromString(scheme_names[a]);
  Py_RETURN_NONE;
}

static PyObject* sifGrid_get_interlaced(PyObject* self_obj, void* closure) {
  (void)closure;
  return PyBool_FromLong(((sifGridObject*)self_obj)->grid->interlaced != 0);
}

static PyGetSetDef sifGrid_getset[] = {
  {"assignment", sifGrid_get_assignment, NULL,
    "The scheme the grid was assigned with -- 'ngp', 'cic', 'tsc' or 'pcs' --\n"
    "or None if nothing has been deposited, or the grid was read from a file\n"
    "that does not record it. The finders and pysif.measure undo this\n"
    "scheme's window (CIC's when it is None).",
    NULL},
  {"interlaced", sifGrid_get_interlaced, NULL,
    "Whether the grid was assigned with interlace=True.", NULL},
  {NULL, NULL, NULL, NULL, NULL}};

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
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status =
    sif_grid_assign_cic_tessellation(self->grid, py_tess->tess);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) return py_sif_field_raise_binning(
      status, "assign_cic_tessellation");

  Py_RETURN_NONE;
}

static PyObject* sifGrid_to_density_contrast(
  PyObject* self_obj, PyObject* args) {
  sifGridObject* self = (sifGridObject*)self_obj;

  sif_error_clear();
  const int status = sif_grid_to_density_contrast(self->grid);
  if (status != SIF_OK)
    return py_sif_raise(status, "to_density_contrast");

  Py_RETURN_NONE;
}

static PyMethodDef sifGrid_methods[] = {
  {"assign_cic", (PyCFunction)sifGrid_assign_cic, METH_VARARGS | METH_KEYWORDS,
    "assign_cic($self, field)\n"
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
  {"assign", (PyCFunction)sifGrid_assign, METH_VARARGS | METH_KEYWORDS,
    "assign($self, field, scheme='cic', *, interlace=False)\n"
    "--\n\n"
    "Deposit a particle field onto the grid with a mass-assignment scheme.\n\n"
    "The schemes spread each particle over 1, 2, 3 or 4 cells per axis:\n"
    "nearest grid point, cloud-in-cell, triangular-shaped cloud and\n"
    "piecewise cubic spline. A higher order folds less power from above the\n"
    "grid's Nyquist frequency back onto the resolved modes, at the cost of a\n"
    "deposit that touches more cells (27 for TSC, 64 for PCS, against 8).\n\n"
    "interlace=True deposits the field a second time shifted by half a cell\n"
    "along each axis and averages the two in Fourier space, which cancels\n"
    "the leading aliasing images outright. It costs a second deposit and a\n"
    "forward and backward FFT.\n\n"
    "assign(field) is assign_cic(field). The grid records the scheme, and\n"
    "the finders and pysif.measure undo that scheme's window.\n\n"
    "Args:\n"
    "    field: The particle field to deposit.\n"
    "    scheme: 'ngp', 'cic', 'tsc' or 'pcs'.\n"
    "    interlace: Average with the half-cell-shifted deposit.\n\n"
    "Raises:\n"
    "    ValueError: On an unknown scheme, a grid of fewer than 4 cells per\n"
    "        side, a particle outside the box, or a field that holds none.\n"
    "        The grid is left as it was.\n"
    "    MemoryError: If the deposit could not allocate its scratch."},
  {"assign_cic_tessellation", (PyCFunction)sifGrid_assign_cic_tessellation,
    METH_VARARGS | METH_KEYWORDS,
    "assign_cic_tessellation($self, tessellation)\n"
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
    "to_density_contrast($self)\n"
    "--\n\n"
    "Convert the cells in place from a density to a density contrast.\n\n"
    "Replaces each cell with delta = rho / rho_mean - 1, so the field\n"
    "averages to zero and is bounded below by -1. This is the form the\n"
    "finders and pysif.measure expect.\n\n"
    "Calling it twice is refused: the second pass would renormalise by a\n"
    "mean that is now zero."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */
PyTypeObject sifGridType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name =
    "pysif.Grid", /* Updated Namespace and Capitalised */
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
  .tp_getset = sifGrid_getset,
  .tp_init = sifGrid_init,
  .tp_new = PyType_GenericNew,
};