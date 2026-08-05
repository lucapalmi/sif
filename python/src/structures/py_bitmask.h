#ifndef __SIF_PY_BITMASK_H__
#define __SIF_PY_BITMASK_H__

#include "py_common.h"
#include "sif/structures/bitmask.h"

/*
 * @brief Python object wrapping the C sif_bitmask_t struct
 */
typedef struct {
  PyObject_HEAD sif_bitmask_t* bitmask;
} sifBitmaskObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifBitmaskType;

#endif /* __SIF_PY_BITMASK_H__ */
