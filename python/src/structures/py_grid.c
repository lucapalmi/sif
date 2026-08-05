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

  sif_grid_t* tmp = sif_grid_alloc(n_cells, (real_t)box_length_in);
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

static PyObject* sifGrid_compute_overdensity(
  PyObject* self_obj, PyObject* args) {
  sifGridObject* self = (sifGridObject*)self_obj;

  sif_grid_compute_overdensity(self->grid);

  Py_RETURN_NONE;
}

static PyMethodDef sifGrid_methods[] = {
  {"assign_cic", (PyCFunction)sifGrid_assign_cic, METH_VARARGS | METH_KEYWORDS,
    "Assign particles from an sif.field to the grid using Cloud-in-Cell "
    "interpolation."},
  {"compute_overdensity", (PyCFunction)sifGrid_compute_overdensity, METH_NOARGS,
    "Compute the density contrast (delta) for the grid cells."},
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */
PyTypeObject sifGridType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pysif.structures.Grid", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifGridObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifGrid_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF cubic grid object.",
  .tp_methods = sifGrid_methods,
  .tp_init = sifGrid_init,
  .tp_new = PyType_GenericNew,
};