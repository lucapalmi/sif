#ifndef __SIF_PY_DELTA_DISTRIBUTION_H__
#define __SIF_PY_DELTA_DISTRIBUTION_H__

#include "py_common.h"
#include "sif/structures/deltadistribution.h"

typedef struct {
  PyObject_HEAD sif_delta_distribution_t* dist;
} sifDeltaDistributionObject;

extern PyTypeObject sifDeltaDistributionType;

PyObject* py_sif_wrap_distribution(sif_delta_distribution_t* dist);

#endif /* __SIF_PY_DELTA_DISTRIBUTION_H__ */
