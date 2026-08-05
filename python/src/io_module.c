#define Py_MODULE_HEAD_UNIFIED
#include "io/py_io.h"

static PyMethodDef io_methods[] = {
  {"write_field", (PyCFunction)pysif_write_field, METH_VARARGS | METH_KEYWORDS, 
   "Writes arrays to a highly optimized .xfield binary format."},
   
  {"read_field",  (PyCFunction)pysif_read_field,  METH_VARARGS | METH_KEYWORDS, 
   "Reads an .xfield binary directly into memory."},
   
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