#define Py_MODULE_HEAD_UNIFIED
#include "io/py_io.h"

static PyMethodDef io_methods[] = {
  {"write_field", (PyCFunction)pysif_write_field, METH_VARARGS | METH_KEYWORDS, 
   "Writes arrays to a highly optimized .xfield binary format."},
   
  {"read_field",  (PyCFunction)pysif_read_field,  METH_VARARGS | METH_KEYWORDS,
   "read_field(filepath, wrap=False) -> Field\n\n"
   "Reads an .xfield binary directly into memory. wrap=True folds every\n"
   "coordinate into [0, box_length) using the box the file itself declares,\n"
   "which is the short way to clear the single-precision rounding that puts\n"
   "a handful of particles exactly on the box edge. It is off by default:\n"
   "folding is only correct for a genuinely periodic field. Use\n"
   "Field.wrap(box_length) instead when you want the counts back."},

  {"read_field_header", (PyCFunction)pysif_read_field_header,
   METH_VARARGS | METH_KEYWORDS,
   "read_field_header(filepath) -> dict\n\n"
   "Reads only the 64-byte .xfield header: n_particles, box_length,\n"
   "has_masses, has_velocities, version. Costs one small read rather than\n"
   "the whole file, so it can size a grid or a chain mesh before committing\n"
   "to loading the tracers. box_length is stored in the file and not in the\n"
   "field itself, so this is the only way to recover it."},


  {"write_grid",  (PyCFunction)pysif_write_grid,  METH_VARARGS | METH_KEYWORDS, 
   "Serializes a discrete density grid to an .xgrid binary."},
   
  {"read_grid",   (PyCFunction)pysif_read_grid,   METH_VARARGS | METH_KEYWORDS, 
   "Loads an .xgrid binary into memory."},
   
  {"read_field_ascii",    (PyCFunction)pysif_read_field_ascii,    METH_VARARGS | METH_KEYWORDS, 
   "Parses an ASCII particle field using custom delimitations."},
   
  {"write_catalog_ascii", (PyCFunction)pysif_write_catalog_ascii, METH_VARARGS | METH_KEYWORDS, 
   "Dumps void catalog properties to a human-readable ASCII format."},
   
  {"read_catalog_ascii",  (PyCFunction)pysif_read_catalog_ascii,  METH_VARARGS | METH_KEYWORDS, 
   "Loads an ASCII catalog into the core tracking structures."},
   
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef io_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif.io",
  .m_doc = "SIF sub-module for high-performance binary and ASCII serialization.",
  .m_size = -1,
  .m_methods = io_methods
};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_io(void) {
  return PyModule_Create(&io_module);
}