/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The .sdf container, from Python.
 *
 * Two things are deliberately not mirrored from C. The status argument does
 * not survive the crossing -- Python has exceptions, and a status a caller can
 * ignore is not the same promise -- so every call raises, with the exception
 * type chosen to say what kind of mistake it was: a missing key is a KeyError,
 * a value of the wrong type a TypeError, a damaged file an OSError. And a
 * metadata table is a dict, because Python already has the dictionary this
 * format spent a header file rebuilding.
 */

#include "py_sdf.h"

#include "measure/py_profiles.h"
#include "structures/py_catalog.h"
#include "structures/py_size_function.h"

/* --- errors --- */

/*
 * Turns a status into the exception a Python caller would expect to catch.
 *
 * The distinctions are the ones a script can act on: a file that is not there
 * can be produced, a key that is not there can be defaulted, and a file that
 * is damaged can only be reported. Everything that means "this file is not
 * what it says" lands on OSError together, since there is nothing different to
 * do about any of them.
 */
static void* sdf_raise(sif_sdf_status_t status, const char* what) {
  PyObject* type;

  switch (status) {
  case SIF_SDF_ERR_NOT_FOUND:
    type = PyExc_FileNotFoundError;
    break;
  case SIF_SDF_ERR_PERMISSION:
    type = PyExc_PermissionError;
    break;
  case SIF_SDF_ERR_EXISTS:
    type = PyExc_FileExistsError;
    break;
  case SIF_SDF_ERR_ABSENT:
    type = PyExc_KeyError;
    break;
  case SIF_SDF_ERR_TYPE:
    type = PyExc_TypeError;
    break;
  case SIF_SDF_ERR_ALLOC:
    return PyErr_NoMemory();
  case SIF_SDF_ERR_INVALID:
  case SIF_SDF_ERR_MODE:
  case SIF_SDF_ERR_NAME_TAKEN:
  case SIF_SDF_ERR_CATALOG_MISMATCH:
    type = PyExc_ValueError;
    break;
  default:
    type = PyExc_OSError;
    break;
  }

  PyErr_Format(type, "%s: %s", what, sif_sdf_strerror(status));
  return NULL;
}

/* The handle, or NULL with a TypeError set. */
static sif_sdf_t* live(sifSDFObject* self) {
  if (!self->file) {
    PyErr_SetString(PyExc_ValueError, "operation on a closed .sdf file");
    return NULL;
  }
  return self->file;
}

/* --- the object --- */

static void sifSDF_dealloc(PyObject* self_obj) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (self->file) {
    /* No status: a file being collected has nobody left to tell. A caller who
     * cares about the flush calls close(). */
    sif_sdf_close(self->file, NULL);
    self->file = NULL;
  }
  Py_TYPE(self)->tp_free(self_obj);
}

static int sifSDF_init(PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  const char* filepath;
  const char* mode = "r";
  static char* kwlist[] = {"filepath", "mode", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s", kwlist, &filepath, &mode))
    return -1;

  sif_sdf_mode_t open_mode;
  if (strcmp(mode, "r") == 0) {
    open_mode = SIF_SDF_READ;
  } else if (strcmp(mode, "a") == 0) {
    open_mode = SIF_SDF_APPEND;
  } else {
    PyErr_Format(PyExc_ValueError,
      "mode must be 'r' or 'a', not '%s' -- an .sdf file is read or added to, "
      "never rewritten",
      mode);
    return -1;
  }

  if (self->file) {
    sif_sdf_close(self->file, NULL);
    self->file = NULL;
  }

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_t* file = NULL;

  Py_BEGIN_ALLOW_THREADS file = sif_sdf_open(filepath, open_mode, &status);
  Py_END_ALLOW_THREADS

    if (!file) {
    sdf_raise(status, filepath);
    return -1;
  }

  self->file = file;
  return 0;
}

static PyObject* sifSDF_create(PyObject* cls, PyObject* args, PyObject* kwds) {
  const char* filepath;
  double box_length;
  PyObject* cat_obj;
  int overwrite = 0;
  static char* kwlist[] = {
    "filepath", "box_length", "catalog", "overwrite", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sdO!|p", kwlist, &filepath,
        &box_length, &sifCatalogType, &cat_obj, &overwrite))
    return NULL;

  sif_catalog_t* cat = ((sifCatalogObject*)cat_obj)->catalog;
  if (!cat) {
    PyErr_SetString(PyExc_ValueError, "the catalog holds nothing");
    return NULL;
  }

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_t* file = NULL;
  const sif_option flags = overwrite ? SIF_SDF_OVERWRITE : 0;

  Py_BEGIN_ALLOW_THREADS file =
    sif_sdf_create(filepath, box_length, cat, flags, &status);
  Py_END_ALLOW_THREADS

    if (!file) return sdf_raise(status, filepath);

  sifSDFObject* self = (sifSDFObject*)sifSDFType.tp_alloc(&sifSDFType, 0);
  if (!self) {
    sif_sdf_close(file, NULL);
    return PyErr_NoMemory();
  }

  self->file = file;
  return (PyObject*)self;
}

static PyObject* sifSDF_close(PyObject* self_obj, PyObject* Py_UNUSED(args)) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;

  /* Here the flush is reported, unlike in dealloc: a caller who wrote this
   * call wants to know whether the last block reached the disk. */
  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_close(self->file, &status);
  self->file = NULL;

  if (status != SIF_SDF_OK)
    return sdf_raise(status, "close");

  Py_RETURN_NONE;
}

static PyObject* sifSDF_enter(PyObject* self_obj, PyObject* Py_UNUSED(args)) {
  Py_INCREF(self_obj);
  return self_obj;
}

static PyObject* sifSDF_exit(PyObject* self_obj, PyObject* args) {
  return sifSDF_close(self_obj, args);
}

/* --- reading --- */

static PyObject* sifSDF_catalog(PyObject* self_obj, PyObject* Py_UNUSED(a)) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = NULL;

  Py_BEGIN_ALLOW_THREADS cat = sif_sdf_catalog(file, &status);
  Py_END_ALLOW_THREADS

    if (!cat) return sdf_raise(status, sif_sdf_path(file));

  sifCatalogObject* obj =
    (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!obj) {
    sif_catalog_free(cat);
    return PyErr_NoMemory();
  }

  obj->catalog = cat;
  return (PyObject*)obj;
}

static PyObject* sifSDF_size_function(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  const char* name = NULL;
  static char* kwlist[] = {"name", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|z", kwlist, &name))
    return NULL;

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_size_function_t* vsf = NULL;

  Py_BEGIN_ALLOW_THREADS vsf = sif_sdf_size_function(file, name, &status);
  Py_END_ALLOW_THREADS

    if (!vsf) return sdf_raise(status, sif_sdf_path(file));

  return py_sif_wrap_size_function(vsf);
}

static PyObject* sifSDF_profiles(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  const char* name = NULL;
  static char* kwlist[] = {"name", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|z", kwlist, &name))
    return NULL;

  /* One Profiles carries both kinds, and a file may hold either or both under
   * a name, so each is asked for on its own status: a set that is not there is
   * an absence to be worked around here, not a failure of the call. */
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  sif_sdf_status_t dens_status = SIF_SDF_OK;
  sif_sdf_status_t vel_status = SIF_SDF_OK;

  Py_BEGIN_ALLOW_THREADS if (sif_sdf_contains(file,
                               SIF_SDF_BLOCK_DENSITY_PROFILES, name)) dens =
    sif_sdf_density_profiles(file, name, &dens_status);
  if (sif_sdf_contains(file, SIF_SDF_BLOCK_VELOCITY_PROFILES, name))
    vel = sif_sdf_velocity_profiles(file, name, &vel_status);
  Py_END_ALLOW_THREADS

    if (dens_status != SIF_SDF_OK || vel_status != SIF_SDF_OK) {
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    return sdf_raise(
      dens_status != SIF_SDF_OK ? dens_status : vel_status, sif_sdf_path(file));
  }

  if (!dens && !vel)
    return sdf_raise(SIF_SDF_ERR_ABSENT, sif_sdf_path(file));

  sifProfilesObject* obj =
    (sifProfilesObject*)sifProfilesType.tp_alloc(&sifProfilesType, 0);
  if (!obj) {
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    return PyErr_NoMemory();
  }

  obj->dens = dens;
  obj->vel = vel;
  return (PyObject*)obj;
}

/* --- writing --- */

static PyObject* sifSDF_append_size_function(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  PyObject* vsf_obj;
  const char* name = NULL;
  static char* kwlist[] = {"size_function", "name", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!|z", kwlist, &sifSizeFunctionType, &vsf_obj, &name))
    return NULL;

  sif_size_function_t* vsf = ((sifSizeFunctionObject*)vsf_obj)->vsf;
  if (!vsf) {
    PyErr_SetString(PyExc_ValueError, "the size function holds nothing");
    return NULL;
  }

  sif_sdf_status_t status = SIF_SDF_OK;
  Py_BEGIN_ALLOW_THREADS sif_sdf_append_size_function(file, vsf, name, &status);
  Py_END_ALLOW_THREADS

    if (status != SIF_SDF_OK) return sdf_raise(status, sif_sdf_path(file));

  Py_RETURN_NONE;
}

static PyObject* sifSDF_append_profiles(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  PyObject* prof_obj;
  const char* name = NULL;
  static char* kwlist[] = {"profiles", "name", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!|z", kwlist, &sifProfilesType, &prof_obj, &name))
    return NULL;

  sifProfilesObject* profs = (sifProfilesObject*)prof_obj;
  if (!profs->dens && !profs->vel) {
    PyErr_SetString(PyExc_ValueError, "the profiles hold nothing");
    return NULL;
  }

  /* Both under one name: they are different kinds of block, and a density set
   * and a velocity set measured together are one measurement. */
  sif_sdf_status_t status = SIF_SDF_OK;
  Py_BEGIN_ALLOW_THREADS if (profs->dens)
    sif_sdf_append_density_profiles(file, profs->dens, name, &status);
  if (profs->vel)
    sif_sdf_append_velocity_profiles(file, profs->vel, name, &status);
  Py_END_ALLOW_THREADS

    if (status != SIF_SDF_OK) return sdf_raise(status, sif_sdf_path(file));

  Py_RETURN_NONE;
}

/* --- metadata, as a dict --- */

/* One value out of the table, as the Python object it should be. */
static PyObject* meta_value(const sif_sdf_meta_t* meta, const char* key) {
  const uint32_t length = sif_sdf_meta_length(meta, key);

  switch (sif_sdf_meta_type(meta, key)) {
  case SIF_SDF_META_I64: {
    if (length == 1) {
      int64_t value = 0;
      if (sif_sdf_meta_get_i64(meta, key, &value) != SIF_SDF_OK)
        break;
      return PyLong_FromLongLong((long long)value);
    }
    const int64_t* values = NULL;
    uint32_t n = 0;
    if (sif_sdf_meta_get_i64v(meta, key, &values, &n) != SIF_SDF_OK)
      break;
    PyObject* list = PyList_New(n);
    if (!list)
      return NULL;
    for (uint32_t i = 0; i < n; i++)
      PyList_SET_ITEM(list, i, PyLong_FromLongLong((long long)values[i]));
    return list;
  }
  case SIF_SDF_META_F64: {
    if (length == 1) {
      double value = 0.0;
      if (sif_sdf_meta_get_f64(meta, key, &value) != SIF_SDF_OK)
        break;
      return PyFloat_FromDouble(value);
    }
    const double* values = NULL;
    uint32_t n = 0;
    if (sif_sdf_meta_get_f64v(meta, key, &values, &n) != SIF_SDF_OK)
      break;
    PyObject* list = PyList_New(n);
    if (!list)
      return NULL;
    for (uint32_t i = 0; i < n; i++)
      PyList_SET_ITEM(list, i, PyFloat_FromDouble(values[i]));
    return list;
  }
  case SIF_SDF_META_STR: {
    const char* text = NULL;
    if (sif_sdf_meta_get_str(meta, key, &text) != SIF_SDF_OK)
      break;
    return PyUnicode_FromString(text);
  }
  default:
    break;
  }

  PyErr_Format(PyExc_OSError, "the metadata key '%s' could not be read", key);
  return NULL;
}

static PyObject* sifSDF_metadata(PyObject* self_obj, PyObject* Py_UNUSED(a)) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_meta_t* meta = sif_sdf_meta_read(file, &status);
  if (!meta)
    return sdf_raise(status, sif_sdf_path(file));

  PyObject* dict = PyDict_New();
  if (!dict) {
    sif_sdf_meta_free(meta);
    return NULL;
  }

  for (uint32_t i = 0; i < sif_sdf_meta_count(meta); i++) {
    const char* key = sif_sdf_meta_key(meta, i);
    PyObject* value = meta_value(meta, key);
    if (!value || PyDict_SetItemString(dict, key, value) < 0) {
      Py_XDECREF(value);
      Py_DECREF(dict);
      sif_sdf_meta_free(meta);
      return NULL;
    }
    Py_DECREF(value);
  }

  sif_sdf_meta_free(meta);
  return dict;
}

/* A sequence of numbers, as one metadata entry. */
static int meta_put_sequence(
  sif_sdf_meta_t* meta, const char* key, PyObject* value) {

  PyObject* fast = PySequence_Fast(value, "expected a sequence");
  if (!fast)
    return -1;

  const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
  if (n > UINT32_MAX) {
    Py_DECREF(fast);
    PyErr_Format(PyExc_ValueError, "'%s' has too many values", key);
    return -1;
  }

  /* Integers only if every element is one: a list of counts stays a list of
   * counts, and one float in it makes the whole thing measurements. */
  int all_integers = 1;
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* item = PySequence_Fast_GET_ITEM(fast, i);
    if (!PyLong_Check(item) || PyBool_Check(item)) {
      all_integers = 0;
      break;
    }
  }

  int failed = 0;

  if (all_integers) {
    int64_t* values = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(int64_t));
    if (!values) {
      PyErr_NoMemory();
      failed = 1;
    } else {
      for (Py_ssize_t i = 0; i < n && !failed; i++) {
        values[i] =
          (int64_t)PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred())
          failed = 1;
      }
      if (!failed)
        sif_sdf_meta_put_i64v(meta, key, values, (uint32_t)n);
      PyMem_Free(values);
    }
  } else {
    double* values = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    if (!values) {
      PyErr_NoMemory();
      failed = 1;
    } else {
      for (Py_ssize_t i = 0; i < n && !failed; i++) {
        values[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred())
          failed = 1;
      }
      if (!failed)
        sif_sdf_meta_put_f64v(meta, key, values, (uint32_t)n);
      PyMem_Free(values);
    }
  }

  Py_DECREF(fast);
  return failed ? -1 : 0;
}

static PyObject* sifSDF_write_metadata(
  PyObject* self_obj, PyObject* args, PyObject* kwds) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  PyObject* dict;
  static char* kwlist[] = {"metadata", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!", kwlist, &PyDict_Type, &dict))
    return NULL;

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta)
    return PyErr_NoMemory();

  PyObject *key_obj, *value;
  Py_ssize_t pos = 0;

  while (PyDict_Next(dict, &pos, &key_obj, &value)) {
    if (!PyUnicode_Check(key_obj)) {
      sif_sdf_meta_free(meta);
      PyErr_SetString(PyExc_TypeError, "metadata keys must be strings");
      return NULL;
    }

    const char* key = PyUnicode_AsUTF8(key_obj);
    if (!key) {
      sif_sdf_meta_free(meta);
      return NULL;
    }

    /* Checked before PyLong, since a bool is one. A flag is stored as an
     * integer of 0 or 1, which is what the format uses for its own. */
    if (PyBool_Check(value)) {
      sif_sdf_meta_put_i64(meta, key, (value == Py_True) ? 1 : 0);
    } else if (PyLong_Check(value)) {
      const long long as_int = PyLong_AsLongLong(value);
      if (PyErr_Occurred()) {
        sif_sdf_meta_free(meta);
        return NULL;
      }
      sif_sdf_meta_put_i64(meta, key, (int64_t)as_int);
    } else if (PyFloat_Check(value)) {
      sif_sdf_meta_put_f64(meta, key, PyFloat_AsDouble(value));
    } else if (PyUnicode_Check(value)) {
      const char* text = PyUnicode_AsUTF8(value);
      if (!text) {
        sif_sdf_meta_free(meta);
        return NULL;
      }
      sif_sdf_meta_put_str(meta, key, text);
    } else if (PySequence_Check(value)) {
      if (meta_put_sequence(meta, key, value) < 0) {
        sif_sdf_meta_free(meta);
        return NULL;
      }
    } else {
      sif_sdf_meta_free(meta);
      PyErr_Format(PyExc_TypeError,
        "'%s' is a %s; metadata holds integers, floats, strings and "
        "sequences of numbers",
        key, Py_TYPE(value)->tp_name);
      return NULL;
    }
  }

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_meta_write(file, meta, &status);
  sif_sdf_meta_free(meta);

  if (status != SIF_SDF_OK)
    return sdf_raise(status, sif_sdf_path(file));

  Py_RETURN_NONE;
}

/* --- what is in the file --- */

static const char* block_kind(sif_sdf_block_type_t type) {
  switch (type) {
  case SIF_SDF_BLOCK_CATALOG:
    return "catalog";
  case SIF_SDF_BLOCK_META:
    return "metadata";
  case SIF_SDF_BLOCK_DENSITY_PROFILES:
    return "density_profiles";
  case SIF_SDF_BLOCK_VELOCITY_PROFILES:
    return "velocity_profiles";
  case SIF_SDF_BLOCK_SIZE_FUNCTION:
    return "size_function";
  default:
    return (type >= SIF_SDF_BLOCK_PRIVATE) ? "private" : "unknown";
  }
}

static PyObject* sifSDF_blocks(PyObject* self_obj, PyObject* Py_UNUSED(args)) {
  sif_sdf_t* file = live((sifSDFObject*)self_obj);
  if (!file)
    return NULL;

  const uint32_t n = sif_sdf_n_blocks(file);
  PyObject* list = PyList_New(n);
  if (!list)
    return NULL;

  for (uint32_t i = 0; i < n; i++) {
    sif_sdf_block_info_t info;
    sif_sdf_info(file, i, &info);

    PyObject* entry =
      Py_BuildValue("{s:s,s:s,s:K}", "kind", block_kind(info.type), "name",
        info.name, "n_items", (unsigned long long)info.n_items);
    if (!entry) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, entry);
  }

  return list;
}

/* --- properties --- */

static PyObject* sifSDF_get_path(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyUnicode_FromString(sif_sdf_path(self->file));
}

static PyObject* sifSDF_get_box_length(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyFloat_FromDouble(sif_sdf_box_length(self->file));
}

static PyObject* sifSDF_get_n_voids(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(sif_sdf_n_voids(self->file));
}

static PyObject* sifSDF_get_n_blocks(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLong(sif_sdf_n_blocks(self->file));
}

static PyObject* sifSDF_get_created(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(sif_sdf_created(self->file));
}

static PyObject* sifSDF_get_writer(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  if (!self->file)
    Py_RETURN_NONE;
  return PyUnicode_FromString(sif_sdf_writer(self->file));
}

static PyObject* sifSDF_get_closed(PyObject* self_obj, void* closure) {
  sifSDFObject* self = (sifSDFObject*)self_obj;
  return PyBool_FromLong(self->file == NULL);
}

static PyGetSetDef sifSDF_getsetters[] = {
  {"path", sifSDF_get_path, NULL, "Where the file is.", NULL},
  {"box_length", sifSDF_get_box_length, NULL,
    "Simulation box size, shared by everything in the file.", NULL},
  {"n_voids", sifSDF_get_n_voids, NULL,
    "Voids in the catalogue, without reading it.", NULL},
  {"n_blocks", sifSDF_get_n_blocks, NULL,
    "Blocks in the file, the catalogue included.", NULL},
  {"created", sifSDF_get_created, NULL,
    "When the file was created, in seconds since the epoch.", NULL},
  {"writer", sifSDF_get_writer, NULL, "The sif version that created the file.",
    NULL},
  {"closed", sifSDF_get_closed, NULL, "Whether close() has been called.", NULL},
  {NULL}};

static PyMethodDef sifSDF_methods[] = {
  {"create", (PyCFunction)sifSDF_create,
    METH_VARARGS | METH_KEYWORDS | METH_STATIC,
    "create(filepath, box_length, catalog, overwrite=False)\n"
    "--\n\n"
    "Create a file around a catalogue, open for appending.\n\n"
    "The catalogue is not optional and cannot be added later: a file is a\n"
    "catalogue and the measurements made from it, and writing both in one\n"
    "call is what makes a file without one impossible rather than merely\n"
    "discouraged.\n\n"
    "Raises:\n"
    "    FileExistsError: The path is taken and overwrite is False."},

  {"close", sifSDF_close, METH_NOARGS,
    "close()\n--\n\nFlush and close. Raises OSError if the last block never\n"
    "reached the disk."},
  {"__enter__", sifSDF_enter, METH_NOARGS, "Enter a with block."},
  {"__exit__", sifSDF_exit, METH_VARARGS, "Close on leaving a with block."},

  {"catalog", sifSDF_catalog, METH_NOARGS,
    "catalog()\n--\n\nRead the catalogue the file is built around.\n\n"
    "A fresh Catalog every call; the file keeps none."},

  {"size_function", (PyCFunction)sifSDF_size_function,
    METH_VARARGS | METH_KEYWORDS,
    "size_function(name=None)\n"
    "--\n\n"
    "Read a size function.\n\n"
    "Args:\n"
    "    name: Which one, or None for the first in the file.\n\n"
    "Raises:\n"
    "    KeyError: The file holds no such block."},

  {"profiles", (PyCFunction)sifSDF_profiles, METH_VARARGS | METH_KEYWORDS,
    "profiles(name=None)\n"
    "--\n\n"
    "Read a profile set: density, velocity, or both under one name.\n\n"
    "Raises:\n"
    "    KeyError: The file holds neither under that name."},

  {"append_size_function", (PyCFunction)sifSDF_append_size_function,
    METH_VARARGS | METH_KEYWORDS,
    "append_size_function(size_function, name=None)\n"
    "--\n\n"
    "Append a measured size function.\n\n"
    "Raises:\n"
    "    ValueError: It was measured from a different catalogue than the\n"
    "        file holds, or from none at all, or the name is taken."},

  {"append_profiles", (PyCFunction)sifSDF_append_profiles,
    METH_VARARGS | METH_KEYWORDS,
    "append_profiles(profiles, name=None)\n"
    "--\n\n"
    "Append whichever profile sets the Profiles carries, under one name.\n\n"
    "Raises:\n"
    "    ValueError: As append_size_function()."},

  {"metadata", sifSDF_metadata, METH_NOARGS,
    "metadata()\n"
    "--\n\n"
    "The file's notes, as a dict.\n\n"
    "Every metadata block in the file, merged in file order, so a value\n"
    "written twice reads as the later of the two. Empty for a file that\n"
    "carries none."},

  {"write_metadata", (PyCFunction)sifSDF_write_metadata,
    METH_VARARGS | METH_KEYWORDS,
    "write_metadata(metadata)\n"
    "--\n\n"
    "Append a dict of notes to the file.\n\n"
    "Keys are strings. Values are int, float, str, bool (stored as 0 or 1)\n"
    "or a sequence of numbers. Keys beginning 'sif.' are reserved.\n\n"
    "A key already in the file is superseded rather than replaced: the file\n"
    "grows, and a later read sees the new value."},

  {"blocks", sifSDF_blocks, METH_NOARGS,
    "blocks()\n"
    "--\n\n"
    "What the file holds, as a list of dicts with 'kind', 'name' and\n"
    "'n_items'. Reads nothing: the block table was built when the file was\n"
    "opened."},

  {NULL, NULL, 0, NULL}};

PyTypeObject sifSDFType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pysif.io.SDF",
  .tp_doc = "SDF(filepath, mode='r')\n"
            "--\n\n"
            "An open .sdf file: a void catalogue and the things measured\n"
            "from it.\n\n"
            "Args:\n"
            "    filepath: Path to an existing file.\n"
            "    mode: 'r' to read, 'a' to read and append. There is no\n"
            "        write mode: a file grows by appending whole blocks and\n"
            "        nothing already in it is ever modified.\n\n"
            "Use SDF.create() to make a new one. Works as a context "
            "manager.",
  .tp_basicsize = sizeof(sifSDFObject),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_new = PyType_GenericNew,
  .tp_init = sifSDF_init,
  .tp_dealloc = sifSDF_dealloc,
  .tp_methods = sifSDF_methods,
  .tp_getset = sifSDF_getsetters,
};

/* --- checking and recovering, which take a path --- */

PyObject* pysif_sdf_verify(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  sif_sdf_status_t status = SIF_SDF_OK;
  uint64_t damaged = 0;

  Py_BEGIN_ALLOW_THREADS damaged = sif_sdf_verify(filepath, &status);
  Py_END_ALLOW_THREADS

    if (status != SIF_SDF_OK) return sdf_raise(status, filepath);

  return PyLong_FromUnsignedLongLong(damaged);
}

PyObject* pysif_sdf_repair(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath))
    return NULL;

  sif_sdf_status_t status = SIF_SDF_OK;
  uint64_t dropped = 0;

  Py_BEGIN_ALLOW_THREADS dropped = sif_sdf_repair(filepath, &status);
  Py_END_ALLOW_THREADS

    if (status != SIF_SDF_OK) return sdf_raise(status, filepath);

  return PyLong_FromUnsignedLongLong(dropped);
}
