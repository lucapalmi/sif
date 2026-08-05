#ifndef __SIF_PY_SIZEFUNCTION_H__
#define __SIF_PY_SIZEFUNCTION_H__

#include "py_common.h"
#include "sif/structures/sizefunction.h"

typedef struct {
  PyObject_HEAD sif_size_function_t* vsf;
} sifSizeFunctionObject;

extern PyTypeObject sifSizeFunctionType;

/*
 * @brief Wraps a freshly computed size function, taking ownership.
 *
 * Shared by the measure-side and (eventually) model-side producers.
 */
PyObject* py_sif_wrap_size_function(sif_size_function_t* vsf);

#endif /* __SIF_PY_SIZEFUNCTION_H__ */
