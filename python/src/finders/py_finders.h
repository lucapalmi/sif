#ifndef __SIF_PY_FINDERS_H__
#define __SIF_PY_FINDERS_H__

#include "py_common.h"

/* Module-level functional entry points */
PyObject* py_sif_finder_rescaled_spherical(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_sif_finder_spherical(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_sif_finder_suggest_mesh_cells(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* __SIF_PY_FINDERS_H__ */