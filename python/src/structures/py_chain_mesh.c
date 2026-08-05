#include "py_chain_mesh.h"
#include "py_field.h"
#include <numpy/arrayobject.h>

/* --- Lifecycle Methods --- */

static void sifChainMesh_dealloc(PyObject* self_obj) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  if (self->mesh != NULL) {
    sif_chain_mesh_free(self->mesh);
    self->mesh = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifChainMesh_init(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  uint32_t n_cells;
  /* "d" writes a full double, so this must not be a real_t. */
  double box_length_in;
  PyObject* field_obj = NULL;
  int allocate_masses = 0;
  int allocate_velocities = 0;

  int allocate_original_idx = 1;

  static char* kwlist[] = {"n_cells", "box_length", "field",
    "allocate_masses", "allocate_velocities", "allocate_original_idx", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "IdO!|ppp", kwlist, &n_cells,
        &box_length_in, &sifFieldType, &field_obj, &allocate_masses, &allocate_velocities, &allocate_original_idx)) {
    return -1;
  }

  sifFieldObject* field = (sifFieldObject*)field_obj;

  sif_chain_mesh_t* tmp = sif_chain_mesh_alloc(n_cells, (real_t)box_length_in,
    field->field, (bool)allocate_masses, (bool)allocate_velocities,
    (bool)allocate_original_idx);

  if (!tmp) {
    PyErr_SetString(PyExc_ValueError,
      "Failed to build sif.chain_mesh: check n_cells/box_length and that every "
      "particle lies in [0, box_length)");
    return -1;
  }

  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  self->mesh = tmp;
  return 0;
}
/* --- Properties (Getters) --- */

static PyObject* sifChainMesh_get_n_cells(PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  return PyLong_FromUnsignedLong(self->mesh->n_cells);
}

static PyObject* sifChainMesh_get_total_cells(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->mesh->total_cells);
}

static PyObject* sifChainMesh_get_box_length(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  return PyFloat_FromDouble((double)self->mesh->box_length);
}

static PyObject* sifChainMesh_get_cell_length(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  return PyFloat_FromDouble((double)self->mesh->cell_length);
}

static PyObject* sifChainMesh_get_n_particles(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  return PyLong_FromUnsignedLongLong(self->mesh->n_particles);
}

/* Helper macro for extracting 1D float arrays */
#define GET_1D_REAL_ARRAY(member_name)                                         \
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;                    \
  if (!self->mesh->member_name)                                                \
    Py_RETURN_NONE;                                                            \
  npy_intp dims[1] = {self->mesh->n_particles};                                \
  PyObject* array =                                                            \
    PyArray_SimpleNewFromData(1, dims, NPY_REAL_T, self->mesh->member_name);   \
  if (!array)                                                                  \
    return NULL;                                                               \
  Py_INCREF(self_obj);                                                         \
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);                      \
  return array;

static PyObject* sifChainMesh_get_x(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(x)
}
static PyObject* sifChainMesh_get_y(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(y)
}
static PyObject* sifChainMesh_get_z(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(z)
}
static PyObject* sifChainMesh_get_vx(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(vx)
}
static PyObject* sifChainMesh_get_vy(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(vy)
}
static PyObject* sifChainMesh_get_vz(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(vz)
}
static PyObject* sifChainMesh_get_masses(PyObject* self_obj, void* closure) {
  GET_1D_REAL_ARRAY(masses)
}

static PyObject* sifChainMesh_get_original_idx(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  if (!self->mesh->original_idx)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->mesh->n_particles};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_UINT64, self->mesh->original_idx);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyObject* sifChainMesh_get_cell_offsets(
  PyObject* self_obj, void* closure) {
  sifChainMeshObject* self = (sifChainMeshObject*)self_obj;
  if (!self->mesh->cell_offsets)
    Py_RETURN_NONE;

  npy_intp dims[1] = {self->mesh->total_cells + 1};
  PyObject* array =
    PyArray_SimpleNewFromData(1, dims, NPY_UINT64, self->mesh->cell_offsets);
  if (!array)
    return NULL;

  Py_INCREF(self_obj);
  PyArray_SetBaseObject((PyArrayObject*)array, self_obj);
  return array;
}

static PyGetSetDef sifChainMesh_getset[] = {
  {"n_cells", sifChainMesh_get_n_cells, NULL, "Number of cells", NULL},
  {"total_cells", sifChainMesh_get_total_cells, NULL, "Total cells", NULL},
  {"box_length", sifChainMesh_get_box_length, NULL, "Box length", NULL},
  {"cell_length", sifChainMesh_get_cell_length, NULL, "Cell length", NULL},
  {"n_particles", sifChainMesh_get_n_particles, NULL, "Number of particles",
    NULL},
  {"x", sifChainMesh_get_x, NULL, "X coordinates", NULL},
  {"y", sifChainMesh_get_y, NULL, "Y coordinates", NULL},
  {"z", sifChainMesh_get_z, NULL, "Z coordinates", NULL},
  {"vx", sifChainMesh_get_vx, NULL, "X velocities", NULL},
  {"vy", sifChainMesh_get_vy, NULL, "Y velocities", NULL},
  {"vz", sifChainMesh_get_vz, NULL, "Z velocities", NULL},
  {"masses", sifChainMesh_get_masses, NULL, "Masses", NULL},
  {"original_idx", sifChainMesh_get_original_idx, NULL, "Original indices",
    NULL},
  {"cell_offsets", sifChainMesh_get_cell_offsets, NULL, "Cell offsets", NULL},
  {NULL}};

/* --- Methods --- */

static PyMethodDef sifChainMesh_methods[] = {
  {NULL, NULL, 0, NULL}};

/* --- Type Object --- */

PyTypeObject sifChainMeshType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pysif.structures.ChainMesh", /* Updated Namespace and Capitalized */
  .tp_basicsize = sizeof(sifChainMeshObject),
  .tp_itemsize = 0,
  .tp_dealloc = sifChainMesh_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "SIF chain mesh object.",
  .tp_methods = sifChainMesh_methods,
  .tp_getset = sifChainMesh_getset,
  .tp_init = sifChainMesh_init,
  .tp_new = PyType_GenericNew,
};