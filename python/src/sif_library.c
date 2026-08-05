#define PYSIF_MAIN_MODULE
#include "py_common.h"

#include "sif/core/settings.h"
#include "sif/core/system.h"

/* --- Submodule Initialization Hooks --- */
extern PyObject* py_sif_init_structures(void);
extern PyObject* py_sif_init_io(void);
extern PyObject* py_sif_init_measure(void);
extern PyObject* py_sif_init_model(void);
extern PyObject* py_sif_init_finders(void);

/* ============================================================================
 * GLOBAL LIBRARY FUNCTIONS
 * ========================================================================== */

static PyObject* py_sif_init(PyObject* module, PyObject* args, PyObject* kwds) {
  /* Default values */
  int verbose = 0;
  int threads = 0;
  int skip_tuning = 0;
  int save_memory = 0;
  int log_level = 2; /* SIF_LOG_LEVEL_INFO */

  static char* kwlist[] = {
    "verbose", "threads", "skip_tuning", "save_memory", "log_level", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pippi", kwlist, &verbose,
        &threads, &skip_tuning, &save_memory, &log_level)) {
    return NULL;
  }

  sif_fft_config_t fft_cfg = {
    .skip_tuning = (bool)skip_tuning};

  sif_omp_config_t omp_cfg = {.n_threads = (uint32_t)threads};

  sif_config_t config = {.fft_config = &fft_cfg,
    .omp_config = &omp_cfg,
    .verbose = (bool)verbose,
    .log_level = (uint8_t)log_level};

  sif_init(&config);

  Py_RETURN_NONE;
}

static PyObject* py_sif_finalize(PyObject* module, PyObject* args) {
  sif_finalize();
  Py_RETURN_NONE;
}

static PyObject* py_sif_setting_set(
  PyObject* module, PyObject* args, PyObject* kwds) {
  const char* key;
  const char* value;

  static char* kwlist[] = {"key", "value", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist, &key, &value)) {
    return NULL;
  }

  sif_setting_set(key, value);

  Py_RETURN_NONE;
}

static PyObject* py_sif_setting_get(
  PyObject* module, PyObject* args, PyObject* kwds) {
  const char* key;
  const char* fallback = "";

  static char* kwlist[] = {"key", "fallback", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|s", kwlist, &key, &fallback)) {
    return NULL;
  }

  const char* result = sif_setting_get(key, fallback);

  if (!result) {
    Py_RETURN_NONE;
  }

  return PyUnicode_FromString(result);
}

/* ============================================================================
 * MODULE REGISTRATION
 * ========================================================================== */

static PyMethodDef sif_module_methods[] = {
  {"init", (PyCFunction)py_sif_init, METH_VARARGS | METH_KEYWORDS,
    "Initialize the SIF backend hardware configuration (OpenMP/FFTW)."},
  {"finalize", (PyCFunction)py_sif_finalize, METH_NOARGS,
    "Finalize the SIF library and safely free global memory architectures."},
  {"set_setting", (PyCFunction)py_sif_setting_set, METH_VARARGS | METH_KEYWORDS,
    "Dynamically override an internal C library runtime setting."},
  {"get_setting", (PyCFunction)py_sif_setting_get, METH_VARARGS | METH_KEYWORDS,
    "Retrieve the current state of an internal C library runtime setting."},
  {NULL, NULL, 0, NULL}};

static struct PyModuleDef sif_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif",
  .m_doc = "SIF: Extreme Void Library Python Bindings",
  .m_size = -1,
  .m_methods = sif_module_methods,
};

PyMODINIT_FUNC PyInit_pysif(void) {
  /* CRITICAL: Initialize NumPy C-API */
  import_array();

  /* 1. Create the Main Module */
  PyObject* m = PyModule_Create(&sif_module);
  if (m == NULL) {
    return NULL;
  }

  PyObject* sys_modules = PyImport_GetModuleDict();

  /* 2. Initialize and Attach Submodules */
  PyObject* mod_structures = py_sif_init_structures();
  if (mod_structures) {
    PyDict_SetItemString(sys_modules, "pysif.structures", mod_structures);
    PyModule_AddObject(m, "structures", mod_structures);
  } else {
    return NULL; /* Fail hard if core structures fail to load */
  }

  PyObject* mod_io = py_sif_init_io();
  if (mod_io) {
    PyDict_SetItemString(sys_modules, "pysif.io", mod_io);
    PyModule_AddObject(m, "io", mod_io);
  }

  PyObject* mod_measure = py_sif_init_measure();
  if (mod_measure) {
    PyDict_SetItemString(sys_modules, "pysif.measure", mod_measure);
    PyModule_AddObject(m, "measure", mod_measure);
  }

  PyObject* mod_model = py_sif_init_model();
  if (mod_model) {
    PyDict_SetItemString(sys_modules, "pysif.model", mod_model);
    PyModule_AddObject(m, "model", mod_model);
  }

  PyObject* mod_finders = py_sif_init_finders();
  if (mod_finders) {
    PyDict_SetItemString(sys_modules, "pysif.finders", mod_finders);
    PyModule_AddObject(m, "finders", mod_finders);
  }

  return m;
}
