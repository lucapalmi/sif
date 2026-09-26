/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * GADGET snapshots from Python. The C reader takes one enumeration per
 * choice, so that a C call spells out everything; here the same choices are
 * keyword arguments with defaults, spelled as strings, and translated to the
 * enumerations before anything is read. A value that does not translate is a
 * ValueError naming the argument and what it accepts.
 *
 * The GIL is held throughout: an HDF5 snapshot goes through the HDF5 library,
 * which as commonly built is not thread-safe.
 */

#include "io/py_io.h"

#include "structures/py_field.h"

#include "sif/io/gadget_io.h"

#include <stdio.h>
#include <string.h>

/* --- argument translation --- */

/* "auto", "hdf5", or the SnapFormat number as an int or a string. */
static int parse_format(PyObject* obj, sif_gadget_format_t* out) {
  if (!obj || obj == Py_None) {
    *out = SIF_GADGET_FORMAT_AUTO;
    return 0;
  }

  if (PyLong_Check(obj)) {
    const long v = PyLong_AsLong(obj);
    if (v == 1 || v == 2) {
      *out = v == 1 ? SIF_GADGET_FORMAT_1 : SIF_GADGET_FORMAT_2;
      return 0;
    }
  } else if (PyUnicode_Check(obj)) {
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s)
      return -1;
    if (strcmp(s, "auto") == 0) {
      *out = SIF_GADGET_FORMAT_AUTO;
      return 0;
    }
    if (strcmp(s, "1") == 0 || strcmp(s, "2") == 0) {
      *out = s[0] == '1' ? SIF_GADGET_FORMAT_1 : SIF_GADGET_FORMAT_2;
      return 0;
    }
    if (strcmp(s, "hdf5") == 0) {
      *out = SIF_GADGET_FORMAT_HDF5;
      return 0;
    }
  }

  PyErr_SetString(PyExc_ValueError, "format must be 'auto', 1, 2 or 'hdf5'");
  return -1;
}

static PyObject* format_value(sif_gadget_format_t format) {
  switch (format) {
  case SIF_GADGET_FORMAT_1:
    return PyLong_FromLong(1);
  case SIF_GADGET_FORMAT_2:
    return PyLong_FromLong(2);
  default:
    return PyUnicode_FromString("hdf5");
  }
}

/* One of a fixed set of strings, to its enumerator. */
static int parse_choice(const char* arg, const char* s,
  const char* const* names, const int* values, int n, const char* accepted,
  int* out) {
  for (int i = 0; i < n; i++) {
    if (strcmp(s, names[i]) == 0) {
      *out = values[i];
      return 0;
    }
  }
  PyErr_Format(PyExc_ValueError, "%s must be %s, not '%s'", arg, accepted, s);
  return -1;
}

/* After a failed call: the exception that says which kind of failure it was.
 * The reason itself is in the sif log. */
static PyObject* status_error(int status, const char* path) {
  switch (status) {
  case SIF_ERR_INVALID:
    return PyErr_Format(PyExc_ValueError,
      "cannot read %s as asked; see the log for the reason", path);
  case SIF_ERR_ALLOC:
    return PyErr_NoMemory();
  case SIF_ERR_UNSUPPORTED:
    return PyErr_Format(PyExc_RuntimeError,
      "cannot read %s: it is an HDF5 snapshot, and pysif was built without "
      "HDF5 support (rebuild with -C cmake.define.SIF_HDF5_SUPPORT=ON)",
      path);
  default:
    return PyErr_Format(
      PyExc_OSError, "failed to read %s; see the log for the reason", path);
  }
}

/* --- header --- */

static PyObject* u64_list(const uint64_t* v, uint32_t n) {
  PyObject* list = PyList_New(n);
  for (uint32_t i = 0; list && i < n; i++) {
    PyObject* item = PyLong_FromUnsignedLongLong(v[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

static PyObject* f64_list(const double* v, uint32_t n) {
  PyObject* list = PyList_New(n);
  for (uint32_t i = 0; list && i < n; i++) {
    PyObject* item = PyFloat_FromDouble(v[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

static PyObject* header_dict(const sif_gadget_header_t* h) {
  const uint32_t n_named =
    h->n_blocks < SIF_GADGET_MAX_BLOCKS ? h->n_blocks : SIF_GADGET_MAX_BLOCKS;
  PyObject* blocks = NULL;
  if (h->n_blocks > 0 && h->blocks[0][0] != '\0') {
    blocks = PyList_New(n_named);
    for (uint32_t b = 0; blocks && b < n_named; b++) {
      PyObject* s = PyUnicode_FromString(h->blocks[b]);
      if (!s) {
        Py_CLEAR(blocks);
        break;
      }
      PyList_SET_ITEM(blocks, b, s);
    }
    if (!blocks)
      return NULL;
  } else {
    blocks = Py_None;
    Py_INCREF(blocks);
  }

  PyObject* cosmology;
  if (h->has_cosmology) {
    cosmology = Py_BuildValue("{s:d,s:d,s:d}", "omega0", h->omega0,
      "omega_lambda", h->omega_lambda, "hubble_param", h->hubble_param);
    if (!cosmology) {
      Py_DECREF(blocks);
      return NULL;
    }
  } else {
    cosmology = Py_None;
    Py_INCREF(cosmology);
  }

  PyObject* unit = h->unit_length_in_cm > 0
                     ? PyFloat_FromDouble(h->unit_length_in_cm)
                     : (Py_INCREF(Py_None), Py_None);

  PyObject* format = format_value(h->format);
  PyObject* n_file = u64_list(h->n_part_file, h->n_types);
  PyObject* n_total = u64_list(h->n_part_total, h->n_types);
  PyObject* masses = f64_list(h->mass_table, h->n_types);

  PyObject* out = NULL;
  if (format && n_file && n_total && masses && unit) {
    const int binary = h->format != SIF_GADGET_FORMAT_HDF5;
    PyObject* swapped = binary ? PyBool_FromLong(h->is_swapped) : Py_None;
    PyObject* legacy = binary ? PyBool_FromLong(h->is_legacy_header) : Py_None;
    if (!binary) {
      Py_INCREF(Py_None);
      Py_INCREF(Py_None);
    }

    /* "N" steals each reference, so nothing below needs releasing on
     * success; on failure Py_BuildValue releases them itself. */
    out = Py_BuildValue(
      "{s:N,s:N,s:N,s:I,s:I,s:N,s:N,s:N,s:d,s:d,s:d,s:I,s:N,s:N,s:I,s:N}",
      "format", format, "swapped", swapped, "legacy_header", legacy,
      "precision", (unsigned)h->precision, "n_types", (unsigned)h->n_types,
      "n_part_file", n_file, "n_part_total", n_total, "mass_table", masses,
      "time", h->time, "redshift", h->redshift, "box_size", h->box_size,
      "n_files", (unsigned)h->n_files, "cosmology", cosmology,
      "unit_length_in_cm", unit, "n_blocks", (unsigned)h->n_blocks, "blocks",
      blocks);
    return out;
  }

  Py_XDECREF(format);
  Py_XDECREF(n_file);
  Py_XDECREF(n_total);
  Py_XDECREF(masses);
  Py_XDECREF(unit);
  Py_DECREF(cosmology);
  Py_DECREF(blocks);
  return NULL;
}

PyObject* pysif_gadget_header(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* path;
  PyObject* format_obj = NULL;
  static char* kwlist[] = {"path", "format", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|O", kwlist, &path, &format_obj))
    return NULL;

  sif_gadget_format_t format;
  if (parse_format(format_obj, &format) < 0)
    return NULL;

  sif_gadget_header_t h;
  const int status = sif_gadget_read_header(path, format, &h);
  if (status != SIF_OK)
    return status_error(status, path);

  return header_dict(&h);
}

/*
 * Printed through sys.stdout rather than the C stdout, which a notebook does
 * not show: the C printer writes into a temporary file, and its text is
 * handed to Python.
 */
PyObject* pysif_inspect_gadget(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* path;
  PyObject* format_obj = NULL;
  static char* kwlist[] = {"path", "format", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "s|O", kwlist, &path, &format_obj))
    return NULL;

  sif_gadget_format_t format;
  if (parse_format(format_obj, &format) < 0)
    return NULL;

  sif_gadget_header_t h;
  const int status = sif_gadget_read_header(path, format, &h);
  if (status != SIF_OK)
    return status_error(status, path);

  FILE* tmp = tmpfile();
  if (!tmp)
    return PyErr_SetFromErrno(PyExc_OSError);
  fprintf(tmp, "%s\n", path);
  sif_gadget_print_header(&h, tmp);
  const long len = ftell(tmp);
  rewind(tmp);

  PyObject* text = NULL;
  char* buf = len > 0 ? PyMem_Malloc((size_t)len + 1) : NULL;
  if (buf && fread(buf, 1, (size_t)len, tmp) == (size_t)len) {
    buf[len] = '\0';
    text = PyUnicode_FromString(buf);
  }
  PyMem_Free(buf);
  fclose(tmp);
  if (!text)
    return PyErr_Occurred() ? NULL : PyErr_NoMemory();

  PyObject* out = PySys_GetObject("stdout"); /* borrowed */
  int rc = -1;
  if (out && out != Py_None)
    rc = PyFile_WriteObject(text, out, Py_PRINT_RAW);
  Py_DECREF(text);

  if (rc < 0) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError, "sys.stdout is not available");
    return NULL;
  }
  Py_RETURN_NONE;
}

/* --- reader --- */

PyObject* pysif_read_gadget(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* path;
  int ptype = 1;
  PyObject* velocities_obj = Py_None;
  int masses = 0;
  const char* length_s = "kpc";
  double fraction = 1.0;
  unsigned long long seed = 0;
  PyObject* format_obj = NULL;

  static char* kwlist[] = {"path", "ptype", "velocities", "masses", "length",
    "fraction", "seed", "format", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|$iOpsdKO", kwlist, &path,
        &ptype, &velocities_obj, &masses, &length_s, &fraction, &seed,
        &format_obj))
    return NULL;

  sif_gadget_format_t format;
  if (parse_format(format_obj, &format) < 0)
    return NULL;

  if (ptype < 0 || ptype > 5) {
    return PyErr_Format(PyExc_ValueError,
      "ptype must be a particle type from 0 to 5, not %d", ptype);
  }

  /* The C side refuses these too, but only says why in the log. */
  if (!(fraction > 0 && fraction <= 1)) {
    char detail[96];
    snprintf(
      detail, sizeof(detail), "fraction must be in (0, 1], not %g", fraction);
    PyErr_SetString(PyExc_ValueError, detail);
    return NULL;
  }

  /* None or False skips them; a string names the convention. */
  int velocity = SIF_GADGET_VELOCITY_SKIP;
  if (velocities_obj != Py_None && velocities_obj != Py_False) {
    const char* s =
      PyUnicode_Check(velocities_obj) ? PyUnicode_AsUTF8(velocities_obj) : NULL;
    if (!s) {
      PyErr_Clear();
      PyErr_SetString(
        PyExc_ValueError, "velocities must be None, 'raw' or 'peculiar'");
      return NULL;
    }
    static const char* const names[] = {"raw", "peculiar"};
    static const int values[] = {
      SIF_GADGET_VELOCITY_RAW, SIF_GADGET_VELOCITY_PECULIAR};
    if (parse_choice("velocities", s, names, values, 2,
          "None, 'raw' or 'peculiar'", &velocity) < 0)
      return NULL;
  }

  int length;
  {
    static const char* const names[] = {"kpc", "mpc", "auto"};
    static const int values[] = {
      SIF_GADGET_LENGTH_KPC, SIF_GADGET_LENGTH_MPC, SIF_GADGET_LENGTH_AUTO};
    if (parse_choice("length", length_s, names, values, 3,
          "'kpc', 'mpc' or 'auto'", &length) < 0)
      return NULL;
  }

  sif_field_t* field = NULL;
  double box_length = 0.0;
  const int status = sif_field_read_gadget(path, format,
    (sif_gadget_ptype_t)(SIF_GADGET_PTYPE_0 + ptype),
    (sif_gadget_velocity_t)velocity,
    masses ? SIF_GADGET_MASS_READ : SIF_GADGET_MASS_SKIP,
    (sif_gadget_length_t)length, fraction, (uint64_t)seed, &field, &box_length);

  if (status != SIF_OK)
    return status_error(status, path);

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }
  obj->field = field;

  return Py_BuildValue("(Nd)", (PyObject*)obj, box_length);
}
