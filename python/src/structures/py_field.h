#ifndef __SIF_PY_FIELD_H__
#define __SIF_PY_FIELD_H__

#include "py_common.h"
#include "sif/structures/field.h"

typedef struct {
  PyObject_HEAD sif_field_t* field;
} sifFieldObject;

extern PyTypeObject sifFieldType;

#endif /* __SIF_PY_FIELD_H__ */