/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * FITS catalogues from Python. The C reader takes its columns as a struct and
 * its files as an array; here they are keyword arguments and a path or a
 * list of paths, translated before anything is read.
 *
 * The C reader says only whether it worked, with the reason in the log. The
 * exception is chosen here: a path that does not exist is the OSError the
 * operating system gives for it, found before reading; after a failed read, a
 * file whose structure cannot be read is an OSError, and anything else -- a
 * column, an HDU or an expression the file does not have -- is the request's,
 * a ValueError.
 *
 * The GIL is held throughout: cfitsio is not thread-safe unless it was built
 * to be, and nothing here can tell.
 */

#include "io/py_io.h"

#include "measure/py_profiles.h"
#include "structures/py_catalog.h"
#include "structures/py_field.h"
#include "structures/py_size_function.h"

#include "sif/io/fits_io.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#ifdef SIF_HAVE_FITS

/* Whether a file's structure reads: what tells an unreadable file from a
 * request the file cannot satisfy. The summary is thrown away. */
static int structure_reads(const char* path) {
  FILE* sink = tmpfile();
  if (!sink)
    return 1;
  const int status = sif_fits_print_summary(path, sink);
  fclose(sink);
  return status != SIF_ERR_IO;
}

/* An HDU as C takes it: NULL for None, an EXTNAME, or an extension number in
 * decimal, written into `buf`. -1 with an exception set. */
static int hdu_string(PyObject* obj, char* buf, size_t len, const char** out) {
  *out = NULL;
  if (obj == Py_None)
    return 0;
  if (PyLong_Check(obj)) {
    const long ext = PyLong_AsLong(obj);
    if (ext < 0) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_ValueError, "hdu must not be negative");
      return -1;
    }
    snprintf(buf, len, "%ld", ext);
    *out = buf;
    return 0;
  }
  if (PyUnicode_Check(obj)) {
    *out = PyUnicode_AsUTF8(obj);
    return *out ? 0 : -1;
  }
  PyErr_SetString(PyExc_TypeError, "hdu must be None, an EXTNAME or a number");
  return -1;
}

/* --- reader --- */

PyObject* pysif_read_fits(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* paths_obj;
  const char *x = NULL, *y = NULL, *z = NULL, *ra = NULL, *dec = NULL;
  const char *vx = NULL, *vy = NULL, *vz = NULL, *w = NULL;
  const char* where = NULL;
  PyObject* hdu_obj = Py_None;
  double fraction = 1.0;
  unsigned long long seed = 0;

  static char* kwlist[] = {"paths", "x", "y", "z", "ra", "dec", "vx", "vy",
    "vz", "w", "where", "hdu", "fraction", "seed", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$zzzzzzzzzzOdK", kwlist,
        &paths_obj, &x, &y, &z, &ra, &dec, &vx, &vy, &vz, &w, &where, &hdu_obj,
        &fraction, &seed))
    return NULL;

  /* The C side refuses all of these too, but only says why in the log. */
  if ((x || y) && (ra || dec))
    return PyErr_Format(PyExc_ValueError,
      "give x, y and z for positions, or ra, dec and z for sky coordinates, "
      "not some of each");
  if ((ra || dec) ? !(ra && dec && z) : !(x && y && z))
    return PyErr_Format(PyExc_TypeError,
      "read_fits() needs the columns x, y and z -- or ra, dec and z, for sky "
      "coordinates");
  if ((!!vx + !!vy + !!vz) % 3 != 0)
    return PyErr_Format(
      PyExc_ValueError, "vx, vy and vz are given all three or not at all");
  if (!(fraction > 0 && fraction <= 1)) {
    char detail[96];
    snprintf(
      detail, sizeof(detail), "fraction must be in (0, 1], not %g", fraction);
    PyErr_SetString(PyExc_ValueError, detail);
    return NULL;
  }

  char hdu_buf[32];
  const char* hdu;
  if (hdu_string(hdu_obj, hdu_buf, sizeof hdu_buf, &hdu) < 0)
    return NULL;

  PyObject* list = py_sif_paths_list(paths_obj);
  if (!list)
    return NULL;
  const Py_ssize_t n = PyList_GET_SIZE(list);
  if (n > (Py_ssize_t)UINT32_MAX) {
    Py_DECREF(list);
    return PyErr_Format(PyExc_ValueError, "too many paths");
  }

  const char** paths = PyMem_Malloc((size_t)n * sizeof(char*));
  if (!paths) {
    Py_DECREF(list);
    return PyErr_NoMemory();
  }
  for (Py_ssize_t i = 0; i < n; i++)
    paths[i] = PyBytes_AS_STRING(PyList_GET_ITEM(list, i));

  /* A missing file is the operating system's error, with its errno: a
   * FileNotFoundError, as open() would raise. */
  PyObject* result = NULL;
  for (Py_ssize_t i = 0; i < n; i++)
    if (py_sif_require_file(paths[i]) < 0)
      goto done;

  const sif_field_columns_t columns = {.x = x,
    .y = y,
    .ra = ra,
    .dec = dec,
    .z = z,
    .vx = vx,
    .vy = vy,
    .vz = vz,
    .w = w};
  sif_field_t* field = sif_field_read_fits(
    paths, (uint32_t)n, hdu, &columns, where, fraction, (uint64_t)seed);

  if (!field) {
    for (Py_ssize_t i = 0; i < n; i++) {
      if (!structure_reads(paths[i])) {
        PyErr_Format(PyExc_OSError,
          "cannot read %s as a FITS file; see the log for the reason",
          paths[i]);
        goto done;
      }
    }
    PyErr_Format(PyExc_ValueError,
      "cannot read %s%s as asked; see the log for the reason", paths[0],
      n > 1 ? " and the rest" : "");
    goto done;
  }

  sifFieldObject* obj =
    (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    PyErr_NoMemory();
    goto done;
  }
  obj->field = field;
  result = (PyObject*)obj;

done:
  PyMem_Free(paths);
  Py_DECREF(list);
  return result;
}

/* --- inspection --- */

/*
 * Printed through sys.stdout rather than the C stdout, which a notebook does
 * not show: the C printer writes into a temporary file, and its text is
 * handed to Python.
 */
PyObject* pysif_inspect_fits(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  static char* kwlist[] = {"path", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&", kwlist, PyUnicode_FSConverter, &path_obj))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_obj);

  if (py_sif_require_file(path) < 0) {
    Py_DECREF(path_obj);
    return NULL;
  }

  FILE* tmp = tmpfile();
  if (!tmp) {
    Py_DECREF(path_obj);
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  const int status = sif_fits_print_summary(path, tmp);
  if (status != SIF_OK) {
    fclose(tmp);
    PyErr_Format(PyExc_OSError,
      "cannot read %s as a FITS file; see the log for the reason", path);
    Py_DECREF(path_obj);
    return NULL;
  }
  Py_DECREF(path_obj);

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

/* --- catalogues --- */

PyObject* pysif_write_catalog_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  PyObject* cat_obj;
  static char* kwlist[] = {"filepath", "catalog", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!", kwlist,
        PyUnicode_FSConverter, &path_obj, &sifCatalogType, &cat_obj))
    return NULL;

  const char* path = PyBytes_AS_STRING(path_obj);
  const int status =
    sif_catalog_write_fits(path, ((sifCatalogObject*)cat_obj)->catalog);
  if (status != SIF_OK) {
    PyErr_Format(
      PyExc_OSError, "failed to write %s; see the log for the reason", path);
    Py_DECREF(path_obj);
    return NULL;
  }
  Py_DECREF(path_obj);
  Py_RETURN_NONE;
}

PyObject* pysif_read_catalog_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  static char* kwlist[] = {"filepath", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&", kwlist, PyUnicode_FSConverter, &path_obj))
    return NULL;

  const char* path = PyBytes_AS_STRING(path_obj);
  PyObject* result = NULL;
  if (py_sif_require_file(path) == 0) {
    sif_catalog_t* cat = sif_catalog_read_fits(path);
    if (!cat) {
      PyErr_Format(PyExc_ValueError,
        "cannot read %s as a void catalogue; see the log for the reason", path);
    } else {
      sifCatalogObject* obj =
        (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
      if (!obj) {
        sif_catalog_free(cat);
        PyErr_NoMemory();
      } else {
        obj->catalog = cat;
        result = (PyObject*)obj;
      }
    }
  }
  Py_DECREF(path_obj);
  return result;
}

/* --- profiles and size functions --- */

/* After a failed write: a file sif did not write, or one that cannot be
 * written. */
static PyObject* product_write_error(int status, const char* path) {
  if (status == SIF_ERR_INVALID)
    return PyErr_Format(PyExc_ValueError,
      "cannot write %s as asked; see the log for the reason", path);
  return PyErr_Format(PyExc_OSError,
    "failed to write %s -- a FITS file sif did not write is never written "
    "into; see the log",
    path);
}

PyObject* pysif_write_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  PyObject* prof_obj;
  static char* kwlist[] = {"filepath", "profiles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!", kwlist,
        PyUnicode_FSConverter, &path_obj, &sifProfilesType, &prof_obj))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_obj);
  const sifProfilesObject* prof = (const sifProfilesObject*)prof_obj;
  const int status = sif_profiles_write_fits(path, prof->dens, prof->vel);
  PyObject* result =
    status == SIF_OK ? Py_NewRef(Py_None) : product_write_error(status, path);
  Py_DECREF(path_obj);
  return result;
}

PyObject* pysif_read_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  static char* kwlist[] = {"filepath", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&", kwlist, PyUnicode_FSConverter, &path_obj))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_obj);
  PyObject* result = NULL;

  /* Whatever sets the file holds, which the C reader will not guess. */
  int has_dens = 0, has_vel = 0;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  if (py_sif_require_file(path) < 0)
    goto done;
  if (sif_profiles_read_header_fits(path, &has_dens, &has_vel) != SIF_OK) {
    PyErr_Format(PyExc_OSError, "cannot read %s as a FITS file", path);
    goto done;
  }
  if (!has_dens && !has_vel) {
    PyErr_Format(PyExc_ValueError, "%s holds no profiles", path);
    goto done;
  }
  if (sif_profiles_read_fits(
        path, has_dens ? &dens : NULL, has_vel ? &vel : NULL) != SIF_OK) {
    PyErr_Format(PyExc_ValueError,
      "cannot read the profiles of %s; see the log for the reason", path);
    goto done;
  }
  sifProfilesObject* prof =
    (sifProfilesObject*)sifProfilesType.tp_alloc(&sifProfilesType, 0);
  if (!prof) {
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    PyErr_NoMemory();
    goto done;
  }
  prof->dens = dens;
  prof->vel = vel;
  result = (PyObject*)prof;

done:
  Py_DECREF(path_obj);
  return result;
}

PyObject* pysif_write_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  PyObject* vsf_obj;
  static char* kwlist[] = {"filepath", "size_function", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O!", kwlist,
        PyUnicode_FSConverter, &path_obj, &sifSizeFunctionType, &vsf_obj))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_obj);
  const int status =
    sif_size_function_write_fits(path, ((sifSizeFunctionObject*)vsf_obj)->vsf);
  PyObject* result =
    status == SIF_OK ? Py_NewRef(Py_None) : product_write_error(status, path);
  Py_DECREF(path_obj);
  return result;
}

PyObject* pysif_read_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  static char* kwlist[] = {"filepath", NULL};
  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&", kwlist, PyUnicode_FSConverter, &path_obj))
    return NULL;
  const char* path = PyBytes_AS_STRING(path_obj);
  PyObject* result = NULL;
  if (py_sif_require_file(path) == 0) {
    sif_size_function_t* vsf = sif_size_function_read_fits(path);
    if (!vsf) {
      PyErr_Format(PyExc_ValueError,
        "cannot read the size function of %s; see the log for the reason",
        path);
    } else {
      sifSizeFunctionObject* obj =
        (sifSizeFunctionObject*)sifSizeFunctionType.tp_alloc(
          &sifSizeFunctionType, 0);
      if (!obj) {
        sif_size_function_free(vsf);
        PyErr_NoMemory();
      } else {
        obj->vsf = vsf;
        result = (PyObject*)obj;
      }
    }
  }
  Py_DECREF(path_obj);
  return result;
}

/* --- keywords --- */

PyObject* pysif_fits_key(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  const char* key;
  PyObject* hdu_obj = Py_None;
  static char* kwlist[] = {"path", "key", "hdu", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&s|O", kwlist,
        PyUnicode_FSConverter, &path_obj, &key, &hdu_obj))
    return NULL;

  const char* path = PyBytes_AS_STRING(path_obj);
  char hdu_buf[32];
  const char* hdu;
  PyObject* result = NULL;
  if (hdu_string(hdu_obj, hdu_buf, sizeof hdu_buf, &hdu) < 0 ||
      py_sif_require_file(path) < 0)
    goto done;

  /* Typed as the header has it; the C getters give a number or text, and
   * the text of an integer keeps every digit a double would not. */
  char text[4096];
  switch (sif_fits_key_kind(path, hdu, key)) {
  case SIF_FITS_KEY_MISSING:
    result = Py_NewRef(Py_None);
    break;
  case SIF_FITS_KEY_LOGICAL:
    result = PyBool_FromLong(sif_fits_get_key_real(path, hdu, key) != 0.0);
    break;
  case SIF_FITS_KEY_INT:
    sif_fits_get_key_string(path, hdu, key, text, sizeof text);
    result = PyLong_FromString(text, NULL, 10);
    break;
  case SIF_FITS_KEY_REAL:
    result = PyFloat_FromDouble(sif_fits_get_key_real(path, hdu, key));
    break;
  case SIF_FITS_KEY_STRING:
    sif_fits_get_key_string(path, hdu, key, text, sizeof text);
    result = PyUnicode_DecodeLatin1(text, (Py_ssize_t)strlen(text), NULL);
    break;
  }

done:
  Py_DECREF(path_obj);
  return result;
}

PyObject* pysif_set_fits_key(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  PyObject* path_obj = NULL;
  const char* key;
  PyObject* value;
  PyObject* hdu_obj = Py_None;
  static char* kwlist[] = {"path", "key", "value", "hdu", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&sO|O", kwlist,
        PyUnicode_FSConverter, &path_obj, &key, &value, &hdu_obj))
    return NULL;

  const char* path = PyBytes_AS_STRING(path_obj);
  char hdu_buf[32];
  const char* hdu;
  PyObject* result = NULL;
  if (hdu_string(hdu_obj, hdu_buf, sizeof hdu_buf, &hdu) < 0 ||
      py_sif_require_file(path) < 0)
    goto done;

  int status;
  if (PyLong_Check(value)) { /* bool included, as 1 or 0 */
    const long long v = PyLong_AsLongLong(value);
    if (v == -1 && PyErr_Occurred())
      goto done;
    status = sif_fits_set_key_int(path, hdu, key, (int64_t)v);
  } else if (PyFloat_Check(value)) {
    status = sif_fits_set_key_real(path, hdu, key, PyFloat_AS_DOUBLE(value));
  } else if (PyUnicode_Check(value)) {
    const char* v = PyUnicode_AsUTF8(value);
    if (!v)
      goto done;
    status = sif_fits_set_key_string(path, hdu, key, v);
  } else {
    PyErr_SetString(PyExc_TypeError, "value must be an int, a float or a str");
    goto done;
  }

  if (status == SIF_ERR_INVALID)
    PyErr_Format(PyExc_ValueError,
      "cannot set %s in %s: see the log for the reason", key, path);
  else if (status != SIF_OK)
    PyErr_Format(PyExc_OSError, "failed to write %s; see the log", path);
  else
    result = Py_NewRef(Py_None);

done:
  Py_DECREF(path_obj);
  return result;
}

#else /* !SIF_HAVE_FITS */

static PyObject* no_fits(void) {
  return PyErr_Format(PyExc_RuntimeError,
    "pysif was built without FITS support (rebuild with "
    "-C cmake.define.SIF_FITS_SUPPORT=ON)");
}

PyObject* pysif_read_fits(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_inspect_fits(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_write_catalog_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_read_catalog_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_fits_key(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_set_fits_key(PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_write_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_read_profiles_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_write_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

PyObject* pysif_read_size_function_fits(
  PyObject* self, PyObject* args, PyObject* kwds) {
  (void)self;
  (void)args;
  (void)kwds;
  return no_fits();
}

#endif /* SIF_HAVE_FITS */
