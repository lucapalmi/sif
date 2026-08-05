#include "finders/py_finders.h"

static PyMethodDef finders_methods[] = {
  {"rescaled_spherical", (PyCFunction)py_sif_finder_rescaled_spherical, METH_VARARGS | METH_KEYWORDS,
   "Executes the high-performance rescaled spherical threshold void finder algorithm."},
   
  {"spherical", (PyCFunction)py_sif_finder_spherical, METH_VARARGS | METH_KEYWORDS,
   "Executes the pure geometric spherical void finder algorithm."},
   
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef finders_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif.finders",
  .m_doc = "SIF sub-module for cosmic void identification engines.",
  .m_size = -1,
  .m_methods = finders_methods
};

PyObject* py_sif_init_finders(void) {
  return PyModule_Create(&finders_module);
}