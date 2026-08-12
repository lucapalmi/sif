#include "py_common.h"

#include "structures/py_field.h"
#include "structures/py_grid.h"
#include "structures/py_catalog.h"

#include "sif/io/field_io.h"
#include "sif/io/grid_io.h"
#include "sif/io/catalog_io.h"

#include <numpy/arrayobject.h>
#include <string.h>

/* --- Field I/O --- */

PyObject* pysif_write_field(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  double box_length;
  PyObject *x_obj, *y_obj, *z_obj;
  PyObject *vx_obj = NULL, *vy_obj = NULL, *vz_obj = NULL;
  PyObject *masses_obj = NULL;

  static char* kwlist[] = {"filepath", "box_length", "x", "y", "z", 
                           "vx", "vy", "vz", "masses", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sdOOO|OOOO", kwlist,
        &filepath, &box_length, &x_obj, &y_obj, &z_obj, 
        &vx_obj, &vy_obj, &vz_obj, &masses_obj)) {
    return NULL;
  }

  /* 1. Safely cast Python objects to contiguous NumPy arrays without copying */
  PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  
  if (!x_arr || !y_arr || !z_arr) {
    Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr);
    return PyErr_Format(PyExc_TypeError, "Positions must be 1D numpy arrays of type float32/64");
  }

  uint64_t n_particles = PyArray_SIZE(x_arr);

  /* 2. Temporarily wrap the NumPy pointers into our C struct (Zero-Copy) */
  sif_field_t temp_field;
  memset(&temp_field, 0, sizeof(sif_field_t));
  temp_field.n_particles = n_particles;
  temp_field.x = (real_t*)PyArray_DATA(x_arr);
  temp_field.y = (real_t*)PyArray_DATA(y_arr);
  temp_field.z = (real_t*)PyArray_DATA(z_arr);

  PyArrayObject *vx_arr = NULL, *vy_arr = NULL, *vz_arr = NULL, *m_arr = NULL;
  
  if (vx_obj && vx_obj != Py_None && vy_obj && vy_obj != Py_None && vz_obj && vz_obj != Py_None) {
    vx_arr = (PyArrayObject*)PyArray_FROM_OTF(vx_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vy_arr = (PyArrayObject*)PyArray_FROM_OTF(vy_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    vz_arr = (PyArrayObject*)PyArray_FROM_OTF(vz_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (vx_arr && vy_arr && vz_arr) {
      temp_field.vx = (real_t*)PyArray_DATA(vx_arr);
      temp_field.vy = (real_t*)PyArray_DATA(vy_arr);
      temp_field.vz = (real_t*)PyArray_DATA(vz_arr);
    }
  }

  if (masses_obj && masses_obj != Py_None) {
    m_arr = (PyArrayObject*)PyArray_FROM_OTF(masses_obj, NPY_REAL_T, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (m_arr) temp_field.masses = (real_t*)PyArray_DATA(m_arr);
  }

  /* 3. Release the GIL so Python can do other things while NVMe writes */
  int status = 0;
  Py_BEGIN_ALLOW_THREADS
  status = sif_field_write(filepath, &temp_field, box_length);
  Py_END_ALLOW_THREADS

  /* 4. Cleanup NumPy references */
  Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr);
  Py_XDECREF(vx_arr); Py_XDECREF(vy_arr); Py_XDECREF(vz_arr); Py_XDECREF(m_arr);

  if (status != 0) {
    return PyErr_Format(PyExc_IOError, "Failed to write .xfield to %s", filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_field(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  double box_length = 0.0;
  sif_field_t* field = NULL;

  /* Drop GIL while 144 threads hit the disk via parallel pread() */
  Py_BEGIN_ALLOW_THREADS
  field = sif_field_read(filepath, &box_length);
  Py_END_ALLOW_THREADS

  if (!field) {
    return PyErr_Format(PyExc_IOError, "Failed to read .xfield from %s", filepath);
  }

  sifFieldObject* obj = (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }
  
  obj->field = field;
  return (PyObject*)obj;
}

PyObject* pysif_read_field_header(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    return PyErr_Format(PyExc_IOError, "Failed to open %s", filepath);
  }

  sif_xfield_header_t header;
  const size_t got = fread(&header, sizeof(header), 1, file);
  fclose(file);

  if (got != 1) {
    return PyErr_Format(
      PyExc_IOError, "%s is too short to hold an .xfield header", filepath);
  }

  if (strncmp(header.magic, __SIF_XFIELD_MAGIC, 4) != 0) {
    return PyErr_Format(PyExc_ValueError, "%s is not an .xfield file", filepath);
  }

  /* box_length is the reason this exists: it lives in the file, not in
   * sif_field_t, so reading the field is not a way to recover it. */
  return Py_BuildValue("{s:K,s:d,s:O,s:O,s:I}",
    "n_particles", (unsigned long long)header.n_particles,
    "box_length", header.box_length,
    "has_masses", header.has_masses ? Py_True : Py_False,
    "has_velocities", header.has_velocities ? Py_True : Py_False,
    "version", (unsigned int)header.version);
}

/* --- Grid I/O --- */

PyObject* pysif_write_grid(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* grid_obj;

  static char* kwlist[] = {"filepath", "grid", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!", kwlist, 
        &filepath, &sifGridType, &grid_obj)) {
    return NULL;
  }

  sifGridObject* grid_wrap = (sifGridObject*)grid_obj;

  int status = 0;
  Py_BEGIN_ALLOW_THREADS
  status = sif_grid_write(filepath, grid_wrap->grid);
  Py_END_ALLOW_THREADS

  if (status != 0) {
    return PyErr_Format(PyExc_IOError, "Failed to write .xgrid to %s", filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_grid(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  sif_grid_t* grid = NULL;

  Py_BEGIN_ALLOW_THREADS
  grid = sif_grid_read(filepath);
  Py_END_ALLOW_THREADS

  if (!grid) {
    return PyErr_Format(PyExc_IOError, "Failed to read .xgrid from %s", filepath);
  }

  sifGridObject* obj = (sifGridObject*)sifGridType.tp_alloc(&sifGridType, 0);
  if (!obj) {
    sif_grid_free(grid);
    return PyErr_NoMemory();
  }
  
  obj->grid = grid;
  return (PyObject*)obj;
}

/* --- ASCII I/O --- */

PyObject* pysif_read_field_ascii(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  const char* format = "x y z";
  char delimiter = ' ';
  int skip_lines = 1;
  const char* delim_str = " ";

  static char* kwlist[] = {"filepath", "format", "delimiter", "skip_lines", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|ssi", kwlist, &filepath,
        &format, &delim_str, &skip_lines)) {
    return NULL;
  }

  delimiter = delim_str[0];

  /* Allocate an empty field struct. The C ASCII reader will auto-detect 
   * the particle count and allocate the NUMA blocks internally. */
  sif_field_t* field = sif_field_alloc(0);
  if (!field) return PyErr_NoMemory();

  int status = 0;
  Py_BEGIN_ALLOW_THREADS
  status = sif_field_read_ascii(field, filepath, format, delimiter, skip_lines);
  Py_END_ALLOW_THREADS

  if (status != 0) {
    sif_field_free(field);
    return PyErr_Format(PyExc_IOError, "Failed to read ASCII field from %s", filepath);
  }

  sifFieldObject* obj = (sifFieldObject*)sifFieldType.tp_alloc(&sifFieldType, 0);
  if (!obj) {
    sif_field_free(field);
    return PyErr_NoMemory();
  }
  
  obj->field = field;
  return (PyObject*)obj;
}

PyObject* pysif_write_catalog_ascii(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  PyObject* cat_obj;

  static char* kwlist[] = {"filepath", "catalog", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!", kwlist, 
        &filepath, &sifCatalogType, &cat_obj)) {
    return NULL;
  }

  sifCatalogObject* cat = (sifCatalogObject*)cat_obj;

  int status = 0;
  Py_BEGIN_ALLOW_THREADS
  status = sif_catalog_write_ascii(cat->catalog, filepath);
  Py_END_ALLOW_THREADS

  if (status != 0) {
    return PyErr_Format(PyExc_IOError, "Failed to write ASCII catalog to %s", filepath);
  }

  Py_RETURN_NONE;
}

PyObject* pysif_read_catalog_ascii(PyObject* self, PyObject* args, PyObject* kwds) {
  const char* filepath;
  static char* kwlist[] = {"filepath", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
    return NULL;
  }

  sif_catalog_t* cat = NULL;

  Py_BEGIN_ALLOW_THREADS
  cat = sif_catalog_read_ascii(filepath);
  Py_END_ALLOW_THREADS

  if (!cat) {
    return PyErr_Format(PyExc_IOError, "Failed to read ASCII catalog from %s", filepath);
  }

  sifCatalogObject* obj = (sifCatalogObject*)sifCatalogType.tp_alloc(&sifCatalogType, 0);
  if (!obj) {
    sif_catalog_free(cat);
    return PyErr_NoMemory();
  }
  
  obj->catalog = cat;
  return (PyObject*)obj;
}