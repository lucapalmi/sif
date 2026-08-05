#ifndef __SIF_PY_DELTA_MOMENTS_H__
#define __SIF_PY_DELTA_MOMENTS_H__

#include "py_common.h"
#include "sif/structures/deltamoments.h"

typedef struct {
  PyObject_HEAD sif_delta_moments_t* moments;
} sifDeltaMomentsObject;

extern PyTypeObject sifDeltaMomentsType;

/*
 * @brief Wraps a freshly computed moment set, taking ownership.
 *
 * Shared by the measure-side and model-side producers.
 */
PyObject* py_sif_wrap_moments(sif_delta_moments_t* moments);

#endif /* __SIF_PY_DELTA_MOMENTS_H__ */
