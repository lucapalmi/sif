/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * pysif.io.describe(), check() and inspect(): the library's file
 * descriptions and checks (file_info.h, file_check.h) as dicts, or printed
 * as sif-inspect prints them.
 */

#include "py_io.h"

#include "sif/io/file_check.h"
#include "sif/io/file_info.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* --- arguments --- */

/* format=None or one of sif_file_format_name()'s names; -1 with a
 * ValueError for another. */
static int parse_format(const char* name, sif_file_format_t* out) {
  *out = SIF_FILE_FORMAT_UNKNOWN;
  if (!name)
    return 0;
  if (sif_file_format_parse(name, out) == SIF_OK)
    return 0;
  PyErr_Format(PyExc_ValueError,
    "format must be 'ascii', 'fits', 'hdf5', 'gadget', 'xfield', 'xgrid', "
    "'checkpoint' or None, not '%s'",
    name);
  return -1;
}

/* --- building the dicts --- */

/* Sets d[key] = value, stealing value; -1 on failure, value released. */
static int put(PyObject* d, const char* key, PyObject* value) {
  if (!value)
    return -1;
  const int rc = PyDict_SetItemString(d, key, value);
  Py_DECREF(value);
  return rc;
}

static PyObject* meta_dict(const sif_file_meta_t* meta, uint32_t n) {
  PyObject* d = PyDict_New();
  for (uint32_t i = 0; d && i < n; i++) {
    PyObject* v = meta[i].kind == SIF_FILE_META_INT
                    ? PyLong_FromLongLong((long long)meta[i].i)
                  : meta[i].kind == SIF_FILE_META_REAL
                    ? PyFloat_FromDouble(meta[i].r)
                    : PyUnicode_DecodeFSDefault(meta[i].s ? meta[i].s : "");
    if (put(d, meta[i].key, v) < 0)
      Py_CLEAR(d);
  }
  return d;
}

/* A range, or None where there is none. */
static PyObject* bound(double v) {
  if (isnan(v))
    Py_RETURN_NONE;
  return PyFloat_FromDouble(v);
}

static PyObject* stats_dict(const sif_file_stats_t* s) {
  return Py_BuildValue("{s:K,s:K,s:N,s:N}", "n", (unsigned long long)s->n,
    "n_nonfinite", (unsigned long long)s->n_nonfinite, "min", bound(s->min),
    "max", bound(s->max));
}

/* The description; with a check's statistics, each column and array also
 * carries its own. */
static PyObject* info_dict(const sif_file_info_t* info,
  sif_file_stats_t* const* column_stats, const sif_file_stats_t* array_stats) {
  PyObject* d = PyDict_New();
  if (!d)
    return NULL;
  if (put(d, "path", PyUnicode_DecodeFSDefault(info->path)) < 0 ||
      put(d, "format",
        PyUnicode_FromString(sif_file_format_name(info->format))) < 0 ||
      put(d, "detail", PyUnicode_FromString(info->detail ? info->detail : "")) <
        0 ||
      put(d, "n_files", PyLong_FromUnsignedLong(info->n_files)) < 0 ||
      put(d, "size_bytes",
        PyLong_FromUnsignedLongLong((unsigned long long)info->size_bytes)) <
        0 ||
      put(d, "meta", meta_dict(info->meta, info->n_meta)) < 0)
    goto fail;

  /* Held by the dict as they are filled; the local reference goes once
   * they are. */
  PyObject* tables = PyList_New(info->n_tables);
  if (!tables || PyDict_SetItemString(d, "tables", tables) < 0) {
    Py_XDECREF(tables);
    goto fail;
  }
  for (uint32_t t = 0; t < info->n_tables; t++) {
    const sif_file_table_t* tb = &info->tables[t];
    PyObject* columns = PyList_New(tb->n_columns);
    PyObject* table = columns
                        ? Py_BuildValue("{s:s,s:K,s:N,s:N}", "name", tb->name,
                            "n_rows", (unsigned long long)tb->n_rows, "columns",
                            columns, "meta", meta_dict(tb->meta, tb->n_meta))
                        : NULL;
    if (!table) {
      Py_DECREF(tables);
      goto fail;
    }
    PyList_SET_ITEM(tables, t, table);
    for (uint32_t c = 0; c < tb->n_columns; c++) {
      const sif_file_column_t* col = &tb->columns[c];
      PyObject* cd =
        Py_BuildValue("{s:s,s:s,s:K,s:z}", "name", col->name, "type", col->type,
          "width", (unsigned long long)col->width, "unit", col->unit);
      if (cd && column_stats &&
          put(cd, "stats", stats_dict(&column_stats[t][c])) < 0)
        Py_CLEAR(cd);
      if (!cd) {
        Py_DECREF(tables);
        goto fail;
      }
      PyList_SET_ITEM(columns, c, cd);
    }
  }
  Py_DECREF(tables);

  PyObject* arrays = PyList_New(info->n_arrays);
  if (!arrays || PyDict_SetItemString(d, "arrays", arrays) < 0) {
    Py_XDECREF(arrays);
    goto fail;
  }
  for (uint32_t i = 0; i < info->n_arrays; i++) {
    const sif_file_array_t* a = &info->arrays[i];
    PyObject* shape = PyTuple_New(a->rank);
    for (uint32_t k = 0; shape && k < a->rank; k++)
      PyTuple_SET_ITEM(
        shape, k, PyLong_FromUnsignedLongLong((unsigned long long)a->dims[k]));
    PyObject* ad =
      shape ? Py_BuildValue("{s:s,s:s,s:N,s:N}", "name", a->name, "type",
                a->type, "shape", shape, "meta", meta_dict(a->meta, a->n_meta))
            : NULL;
    if (ad && array_stats && put(ad, "stats", stats_dict(&array_stats[i])) < 0)
      Py_CLEAR(ad);
    if (!ad) {
      Py_DECREF(arrays);
      goto fail;
    }
    PyList_SET_ITEM(arrays, i, ad);
  }
  Py_DECREF(arrays);
  return d;

fail:
  Py_DECREF(d);
  return NULL;
}

static PyObject* check_dict(const sif_file_check_t* c) {
  PyObject* info = c->info ? info_dict(c->info, c->column_stats, c->array_stats)
                           : (Py_INCREF(Py_None), Py_None);
  if (!info)
    return NULL;
  PyObject* problems = PyList_New(c->n_problems);
  for (uint32_t i = 0; problems && i < c->n_problems; i++) {
    PyObject* p = Py_BuildValue("{s:s,s:s}", "severity",
      c->problems[i].severity == SIF_FILE_PROBLEM_ERROR ? "error" : "warning",
      "message", c->problems[i].message);
    if (!p) {
      Py_CLEAR(problems);
      break;
    }
    PyList_SET_ITEM(problems, i, p);
  }
  if (!problems) {
    Py_DECREF(info);
    return NULL;
  }
  return Py_BuildValue("{s:N,s:s,s:K,s:K,s:N,s:O}", "info", info, "content",
    c->content, "n_errors", (unsigned long long)c->n_errors, "n_warnings",
    (unsigned long long)c->n_warnings, "problems", problems, "ok",
    c->n_errors == 0 ? Py_True : Py_False);
}

/* --- the entry points --- */

/* The description, or NULL with the reason raised. */
static sif_file_info_t* read_info(const char* path, sif_file_format_t format) {
  sif_file_info_t* info;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS;
  info = sif_file_info_read(path, format);
  Py_END_ALLOW_THREADS;
  if (!info)
    py_sif_raise(sif_error_status(), path);
  return info;
}

/* The check, or NULL with MemoryError raised: a file that cannot be read is
 * a check that says so. */
static sif_file_check_t* run_check(const char* path, sif_file_format_t format) {
  sif_file_check_t* c;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS;
  c = sif_file_check(path, format);
  Py_END_ALLOW_THREADS;
  if (!c)
    py_sif_raise(SIF_ERR_ALLOC, path);
  return c;
}

static PyObject* describe_impl(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  const char* path;
  const char* format_name = NULL;
  static char* kwlist[] = {"path", "format", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&|z", kwlist, py_sif_path_arg, &path, &format_name))
    return NULL;
  sif_file_format_t format;
  if (parse_format(format_name, &format) < 0)
    return NULL;

  sif_file_info_t* info = read_info(path, format);
  if (!info)
    return NULL;
  PyObject* d = info_dict(info, NULL, NULL);
  sif_file_info_free(info);
  return d;
}

static PyObject* check_impl(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  const char* path;
  const char* format_name = NULL;
  static char* kwlist[] = {"path", "format", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&|z", kwlist, py_sif_path_arg, &path, &format_name))
    return NULL;
  sif_file_format_t format;
  if (parse_format(format_name, &format) < 0)
    return NULL;

  sif_file_check_t* c = run_check(path, format);
  if (!c)
    return NULL;
  PyObject* d = check_dict(c);
  sif_file_check_free(c);
  return d;
}

/*
 * Printed through sys.stdout rather than the C stdout, which a notebook does
 * not show: the C printer writes into a temporary file, and its text is
 * handed to Python.
 */
static PyObject* inspect_impl(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  const char* path;
  const char* format_name = NULL;
  int check = 0;
  static char* kwlist[] = {"path", "format", "check", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|z$p", kwlist,
        py_sif_path_arg, &path, &format_name, &check))
    return NULL;
  sif_file_format_t format;
  if (parse_format(format_name, &format) < 0)
    return NULL;

  FILE* tmp = tmpfile();
  if (!tmp)
    return PyErr_SetFromErrno(PyExc_OSError);
  int status;
  if (check) {
    sif_file_check_t* c = run_check(path, format);
    if (!c) {
      fclose(tmp);
      return NULL;
    }
    status = sif_file_check_print(c, tmp);
    sif_file_check_free(c);
  } else {
    sif_file_info_t* info = read_info(path, format);
    if (!info) {
      fclose(tmp);
      return NULL;
    }
    status = sif_file_info_print(info, tmp);
    sif_file_info_free(info);
  }
  if (status != SIF_OK) {
    fclose(tmp);
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  const long len = ftell(tmp);
  rewind(tmp);
  PyObject* text = NULL;
  char* buf = len > 0 ? PyMem_Malloc((size_t)len + 1) : NULL;
  if (buf && fread(buf, 1, (size_t)len, tmp) == (size_t)len) {
    buf[len] = '\0';
    text = PyUnicode_DecodeFSDefault(buf);
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

PyObject* pysif_describe(PyObject* self, PyObject* args, PyObject* kwds) {
  return py_sif_call_with_path(describe_impl, self, args, kwds, "path");
}

PyObject* pysif_check(PyObject* self, PyObject* args, PyObject* kwds) {
  return py_sif_call_with_path(check_impl, self, args, kwds, "path");
}

PyObject* pysif_inspect(PyObject* self, PyObject* args, PyObject* kwds) {
  return py_sif_call_with_path(inspect_impl, self, args, kwds, "path");
}
