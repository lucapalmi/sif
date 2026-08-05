#ifndef __SIF_PY_TESSELLATION_H__
#define __SIF_PY_TESSELLATION_H__

#include "py_common.h"
#include "sif/structures/tessellation.h"

typedef struct {
  PyObject_HEAD 
  sif_tessellation_t* tess;
} sifTessellationObject;

extern PyTypeObject sifTessellationType;

/* Expose the free function for the module */
PyObject* py_sif_tessellation_build(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* __SIF_PY_TESSELLATION_H__ */