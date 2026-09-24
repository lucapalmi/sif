/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#define PYSIF_MAIN_MODULE
#include "py_common.h"

#include "sif/core/settings.h"
#include "sif/core/system.h"

/* --- Submodule Initialization Hooks --- */
extern int py_sif_register_types(PyObject* module);
extern PyObject* py_sif_init_io(void);
extern PyObject* py_sif_init_measure(void);
extern PyObject* py_sif_init_model(void);
extern PyObject* py_sif_init_finders(void);

/* ============================================================================
 * GLOBAL LIBRARY FUNCTIONS
 * ========================================================================== */

static PyObject* py_sif_init(PyObject* module, PyObject* args, PyObject* kwds) {
  /* Default values */
  int threads = 0;
  int skip_tuning = 0;
  int log_level = 2; /* SIF_LOG_LEVEL_INFO */

  static char* kwlist[] = {"threads", "skip_tuning", "log_level", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|ipi", kwlist, &threads, &skip_tuning, &log_level)) {
    return NULL;
  }

  sif_fft_config_t fft_cfg = {.skip_tuning = (bool)skip_tuning};

  sif_omp_config_t omp_cfg = {.n_threads = (uint32_t)threads};

  sif_config_t config = {.fft_config = &fft_cfg,
    .omp_config = &omp_cfg,
    .log_level = (uint8_t)log_level};

  /* Where the C library used to end the process, and now reports instead:
   * the interpreter survives a failed init and can say why. */
  if (sif_init(&config) != SIF_OK) {
    PyErr_SetString(PyExc_MemoryError,
      "sif could not initialize: the library state or FFTW could not be "
      "allocated");
    return NULL;
  }

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
    "init(threads=0, skip_tuning=False, log_level=2)\n"
    "--\n\n"
    "Start the library. Call this before anything else.\n\n"
    "Brings up the logger, the thread ceiling, the settings table and the\n"
    "FFTW plan cache. Calling it twice warns and does nothing.\n\n"
    "Args:\n"
    "    threads: Thread ceiling; 0 leaves OpenMP's own default.\n"
    "    skip_tuning: Take FFTW's estimated plan instead of tuning. Faster\n"
    "        to start, slower to transform; worth setting for short runs.\n"
    "    log_level: 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 none.\n"
    "        0 is the verbose mode: everything, timings included."},
  {"finalize", (PyCFunction)py_sif_finalize, METH_NOARGS,
    "finalize()\n"
    "--\n\n"
    "Shut the library down, releasing everything init() acquired.\n\n"
    "Saves the settings table if it changed and tears down the FFTW plan\n"
    "cache. Nothing else may be called afterwards without init()."},
  {"set_setting", (PyCFunction)py_sif_setting_set, METH_VARARGS | METH_KEYWORDS,
    "set_setting(key, value)\n"
    "--\n\n"
    "Set a persistent runtime setting.\n\n"
    "Settings live in a file under $HOME/.sif and survive between runs.\n"
    "Values may reference environment variables ($HOME, ${USER}), which are\n"
    "expanded when set rather than when read.\n\n"
    "Args:\n"
    "    key: Setting name.\n"
    "    value: Value to store."},
  {"get_setting", (PyCFunction)py_sif_setting_get, METH_VARARGS | METH_KEYWORDS,
    "get_setting(key, fallback=None)\n"
    "--\n\n"
    "Read a persistent runtime setting.\n\n"
    "If the key is absent and a fallback is given, the fallback is stored\n"
    "under that key and returned, so the next call finds it.\n\n"
    "Args:\n"
    "    key: Setting name.\n"
    "    fallback: Value to install and return if the key is absent.\n\n"
    "Returns:\n"
    "    str: The setting's value, or None if absent with no fallback."},
  {NULL, NULL, 0, NULL}};

static struct PyModuleDef sif_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif",
  .m_doc = "sif: cosmic void finding and analysis.\n\n"
           "Call init() before anything else and finalize() when done.\n\n"
           "The data structures live here in the package root -- Field, Grid,\n"
           "Catalog and the rest -- and the operations on them are grouped\n"
           "into submodules: io for reading and writing, finders for void\n"
           "identification, measure for measurements taken from data, and\n"
           "model for theoretical predictions.\n\n"
           "pysif.real is the NumPy scalar type matching the precision the\n"
           "library was built with, so np.zeros(n, dtype=pysif.real) gives\n"
           "arrays the bindings take without a conversion copy.",
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

  /* The NumPy scalar type matching sif_real, so callers can write
   * np.zeros(n, dtype=pysif.real) and get arrays the bindings take without a
   * conversion copy. It is the type object (np.float32 / np.float64) rather
   * than a dtype instance, which also makes pysif.real(value) work as a cast.
   */
  PyObject* real_type = PyArray_TypeObjectFromType(NPY_REAL_T);
  if (!real_type) {
    Py_DECREF(m);
    return NULL;
  }
  if (PyModule_AddObject(m, "real", real_type) < 0) {
    Py_DECREF(real_type);
    Py_DECREF(m);
    return NULL;
  }

  PyObject* sys_modules = PyImport_GetModuleDict();

  /* 2. The data structures live in the package root rather than a submodule.
   * They are the nouns the whole library is written in terms of, and every
   * submodule below takes or returns them, so pysif.Field is where a user
   * expects to find it. The submodules that follow are namespaces of
   * operations, which is a different thing and stays separate. */
  if (py_sif_register_types(m) < 0) {
    Py_DECREF(m);
    return NULL;
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
