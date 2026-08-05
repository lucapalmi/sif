#define Py_MODULE_HEAD_UNIFIED
#include "measure/py_delta.h"
#include "measure/py_profiles.h"
#include "measure/py_size_function.h"

static PyMethodDef measure_methods[] = {
  {"size_function_catalog", (PyCFunction)py_sif_size_function_catalog,
    METH_VARARGS | METH_KEYWORDS,
    "Measures the Void Size Function from a void catalog."},

  {"size_function_combine", (PyCFunction)py_sif_size_function_combine,
    METH_VARARGS | METH_KEYWORDS, "Combines multiple void size functions."},

  {"profiles", (PyCFunction)py_sif_profiles, METH_VARARGS | METH_KEYWORDS,
    "Measures unified overdensity and dynamic velocity tracking shell profiles "
    "around void structures."},

  {"delta_distribution_grid", (PyCFunction)py_sif_delta_distribution_grid,
    METH_VARARGS | METH_KEYWORDS,
    "Measures the PDF of the smoothed density contrast on a grid.\n"
    "Pass shuffle='phases' or 'gaussian' to measure the same field with its "
    "Fourier phases randomized, which preserves the power spectrum and "
    "destroys everything above it."},

  {"delta_moments_grid", (PyCFunction)py_sif_delta_moments_grid,
    METH_VARARGS | METH_KEYWORDS,
    "Measures the spectral moments sigma_0..sigma_order of the smoothed "
    "density contrast from a gridded field.\n"
    "Pass window='gaussian' when order >= 2 has to be meaningful: the top-hat "
    "sum is cut off by the grid, not the field."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef measure_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.measure",
  .m_doc = "SIF sub-module for structural feature measurement metrics.",
  .m_size = -1, .m_methods = measure_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_measure(void) {
  if (PyType_Ready(&sifProfilesType) < 0)
    return NULL;

  PyObject* m = PyModule_Create(&measure_module);
  if (!m)
    return NULL;

  Py_INCREF(&sifProfilesType);
  PyModule_AddObject(m, "Profiles", (PyObject*)&sifProfilesType);

  return m;
}
